from __future__ import annotations

from io import BytesIO
from pathlib import Path
import zipfile

import pytest

from app.services import ingestion
from app.services.ingestion import (
    ArchiveCorruptError,
    ArchiveLimits,
    ArchivePasswordError,
    IngestionError,
    RarArchiveExtractor,
    detect_archive_type,
    extract_archive,
    ingest_folder,
)


def _rar_fixture(path: Path, signature: bytes) -> Path:
    path.write_bytes(signature + b"synthetic test archive")
    return path


class FakeRarInfo:
    def __init__(self, filename: str, data: bytes = b"source", *, password: bool = False, symlink: bool = False, compressed_size: int | None = None):
        self.filename = filename
        self.file_size = len(data)
        self.compress_size = compressed_size if compressed_size is not None else max(1, len(data))
        self._data = data
        self._password = password
        self._symlink = symlink

    def is_dir(self) -> bool:
        return self.filename.endswith("/")

    def is_symlink(self) -> bool:
        return self._symlink

    def needs_password(self) -> bool:
        return self._password


class FakeRarFile:
    infos: list[FakeRarInfo] = []

    def __init__(self, _path: Path):
        self._infos = self.infos

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def infolist(self):
        return self._infos

    def needs_password(self):
        return False

    def open(self, info):
        return BytesIO(info._data)


def patch_rar(monkeypatch, infos: list[FakeRarInfo]) -> None:
    FakeRarFile.infos = infos
    assert ingestion.rarfile is not None
    monkeypatch.setattr(ingestion.rarfile, "RarFile", FakeRarFile)


def test_rar4_and_rar5_signatures_and_octet_stream(tmp_path: Path) -> None:
    rar4 = _rar_fixture(tmp_path / "patient.rar", ingestion.RAR4_SIGNATURE)
    rar5 = _rar_fixture(tmp_path / "patient5.bin", ingestion.RAR5_SIGNATURE)
    assert detect_archive_type(rar4, "application/octet-stream") == "rar"
    assert detect_archive_type(rar5, "application/octet-stream") == "rar"
    assert detect_archive_type(rar4, "application/vnd.rar") == "rar"
    assert detect_archive_type(rar5, "application/x-rar-compressed") == "rar"


def test_rar_root_and_nested_files_preserve_relative_paths(monkeypatch, tmp_path: Path) -> None:
    archive = _rar_fixture(tmp_path / "patient.rar", ingestion.RAR5_SIGNATURE)
    patch_rar(monkeypatch, [FakeRarInfo("page1.jpg", b"jpg"), FakeRarInfo("Scans/Visit 1/lab.pdf", b"pdf")])
    extracted = extract_archive(archive, tmp_path / "incoming")
    try:
        assert extracted.archive_type == "rar"
        assert (extracted.root / "page1.jpg").read_bytes() == b"jpg"
        assert (extracted.root / "Scans" / "Visit 1" / "lab.pdf").read_bytes() == b"pdf"
        ingested = ingest_folder("P1", extracted.root, tmp_path / "data", archive_filename="patient.rar", archive_type="rar")
        assert {item.source_file.original_relative_path for item in ingested} == {"page1.jpg", "Scans/Visit 1/lab.pdf"}
        assert all(item.source_file.archive_filename == "patient.rar" for item in ingested)
    finally:
        import shutil
        shutil.rmtree(extracted.root, ignore_errors=True)


def test_rar_skips_unsupported_nested_archives_and_duplicates(monkeypatch, tmp_path: Path) -> None:
    archive = _rar_fixture(tmp_path / "patient.rar", ingestion.RAR4_SIGNATURE)
    patch_rar(monkeypatch, [
        FakeRarInfo("page1.jpg", b"same"),
        FakeRarInfo("duplicate.png", b"same"),
        FakeRarInfo("notes.doc", b"unsupported"),
        FakeRarInfo("inside.zip", b"nested"),
    ])
    extracted = RarArchiveExtractor().extract(archive, tmp_path / "incoming")
    assert extracted.extracted_count == 2
    assert extracted.unsupported_count == 1
    assert extracted.nested_archive_count == 1
    assert "Nested archive detected and skipped." in extracted.warnings
    ingested = ingest_folder("P1", extracted.root, tmp_path / "data", archive_filename="patient.rar", archive_type="rar")
    assert len(ingested) == 1  # checksum deduplication happens after extraction


def test_zip_limits_and_nested_archive_warning(tmp_path: Path) -> None:
    archive = tmp_path / "patient.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("patient/page1.jpg", b"abc")
        handle.writestr("patient/inside.rar", b"nested")
    with pytest.raises(IngestionError, match="uncompressed-size"):
        extract_archive(archive, tmp_path / "incoming", limits=ArchiveLimits(max_uncompressed_bytes=2))
    extracted = extract_archive(archive, tmp_path / "incoming2")
    assert extracted.nested_archive_count == 1
    assert "Nested archive detected and skipped." in extracted.warnings


def test_password_corrupt_and_traversal_rar_fail_safely(monkeypatch, tmp_path: Path) -> None:
    password_archive = _rar_fixture(tmp_path / "password.rar", ingestion.RAR4_SIGNATURE)
    patch_rar(monkeypatch, [FakeRarInfo("page.jpg", password=True)])
    with pytest.raises(ArchivePasswordError):
        extract_archive(password_archive, tmp_path / "incoming")

    corrupt_archive = _rar_fixture(tmp_path / "corrupt.rar", ingestion.RAR5_SIGNATURE)
    assert ingestion.rarfile is not None
    monkeypatch.setattr(ingestion.rarfile, "RarFile", lambda _path: (_ for _ in ()).throw(ingestion.rarfile.BadRarFile("bad")))
    with pytest.raises(ArchiveCorruptError):
        extract_archive(corrupt_archive, tmp_path / "incoming2")

    traversal_archive = _rar_fixture(tmp_path / "traversal.rar", ingestion.RAR4_SIGNATURE)
    patch_rar(monkeypatch, [FakeRarInfo("../escape.jpg", b"bad")])
    with pytest.raises(IngestionError, match="traversal"):
        extract_archive(traversal_archive, tmp_path / "incoming3")
    assert not (tmp_path / "escape.jpg").exists()


def test_invalid_rar_extension_signature_is_rejected(tmp_path: Path) -> None:
    invalid = (tmp_path / "not-really.rar")
    invalid.write_bytes(b"not a rar")
    with pytest.raises(IngestionError, match="invalid RAR signature"):
        detect_archive_type(invalid, "application/octet-stream")
