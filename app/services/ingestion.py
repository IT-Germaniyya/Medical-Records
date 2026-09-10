from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path, PurePosixPath
import shutil
import stat
import time
from shutil import copy2
from uuid import uuid4
import zipfile
import re

from pypdf import PdfReader

try:  # RAR support is optional in old local environments until dependencies are installed.
    import rarfile
except ImportError:  # pragma: no cover
    rarfile = None  # type: ignore[assignment]

from app.schemas import SourceFile

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".pdf"}
ARCHIVE_SUFFIXES = {".zip", ".rar"}
ZIP_MIME_TYPES = {"application/zip", "application/x-zip-compressed", "application/octet-stream"}
RAR_MIME_TYPES = {"application/vnd.rar", "application/x-rar-compressed", "application/octet-stream"}
ZIP_SIGNATURES = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
RAR4_SIGNATURE = b"Rar!\x1a\x07\x00"
RAR5_SIGNATURE = b"Rar!\x1a\x07\x01\x00"


class IngestionError(ValueError):
    pass


class ArchivePasswordError(IngestionError):
    """The archive is encrypted and no password is accepted by this workflow."""


class ArchiveCorruptError(IngestionError):
    """The archive cannot be read safely."""


@dataclass(frozen=True)
class ArchiveLimits:
    max_files: int = 1000
    max_uncompressed_bytes: int = 500 * 1024 * 1024
    max_single_file_bytes: int = 100 * 1024 * 1024
    max_compression_ratio: float = 1000.0
    extraction_timeout_seconds: float = 120.0
    allow_nested_archives: bool = False


@dataclass
class ArchiveExtraction:
    root: Path
    archive_filename: str
    archive_type: str
    warnings: list[str] = field(default_factory=list)
    unsupported_count: int = 0
    nested_archive_count: int = 0
    extracted_count: int = 0


@dataclass
class IngestedFile:
    source_file: SourceFile
    input_path: Path
    warnings: list[str] = field(default_factory=list)


def checksum_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_member_name(name: str) -> str:
    """Return a portable relative path or raise before touching the filesystem."""
    if not name or "\x00" in name:
        raise IngestionError("archive contains an invalid member path")
    portable = name.replace("\\", "/")
    path = PurePosixPath(portable)
    if path.is_absolute() or portable.startswith("/"):
        raise IngestionError("archive contains an unsafe absolute path")
    if len(portable) >= 2 and portable[1] == ":":
        raise IngestionError("archive contains an unsafe absolute path")
    if ".." in path.parts:
        raise IngestionError("archive contains an unsafe path traversal")
    return "/".join(part for part in path.parts if part not in ("", "."))


def _destination_for_member(target: Path, name: str) -> Path:
    relative = _normalise_member_name(name)
    if not relative:
        return target
    resolved = (target / Path(*PurePosixPath(relative).parts)).resolve()
    if not resolved.is_relative_to(target.resolve()):
        raise IngestionError("archive contains an unsafe path traversal")
    return resolved


def _is_nested_archive(name: str) -> bool:
    return Path(name).suffix.lower() in ARCHIVE_SUFFIXES


def _check_member_limits(name: str, file_size: int, compressed_size: int, count: int, total: int, limits: ArchiveLimits) -> int:
    if count >= limits.max_files:
        raise IngestionError(f"archive exceeds the {limits.max_files} file limit")
    if file_size < 0 or file_size > limits.max_single_file_bytes:
        raise IngestionError(f"archive member exceeds the {limits.max_single_file_bytes} byte single-file limit")
    new_total = total + file_size
    if new_total > limits.max_uncompressed_bytes:
        raise IngestionError(f"archive exceeds the {limits.max_uncompressed_bytes} byte uncompressed-size limit")
    if compressed_size > 0 and file_size / compressed_size > limits.max_compression_ratio:
        raise IngestionError("archive member exceeds the decompression-ratio safety limit")
    return new_total


def _safe_copy_stream(source, destination: Path, *, expected_size: int, deadline: float, limits: ArchiveLimits) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with destination.open("xb") as output:
        while True:
            if time.monotonic() > deadline:
                raise IngestionError("archive extraction timed out")
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if written > limits.max_single_file_bytes:
                raise IngestionError(f"archive member exceeds the {limits.max_single_file_bytes} byte single-file limit")
            output.write(chunk)
    if expected_size >= 0 and written != expected_size:
        raise ArchiveCorruptError("archive member size did not match its header")


class ArchiveExtractor:
    archive_type: str = ""

    def can_handle(self, file_path: Path, mime_type: str | None = None) -> bool:
        return detect_archive_type(file_path, mime_type, raise_on_unknown=False) == self.archive_type

    def extract(self, file_path: Path, destination: Path, limits: ArchiveLimits | None = None) -> ArchiveExtraction:
        raise NotImplementedError


class ZipArchiveExtractor(ArchiveExtractor):
    archive_type = "zip"

    def extract(self, file_path: Path, destination: Path, limits: ArchiveLimits | None = None) -> ArchiveExtraction:
        limits = limits or ArchiveLimits()
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / f"zip-{uuid4()}"
        target.mkdir(parents=True, exist_ok=False)
        result = ArchiveExtraction(target, file_path.name, self.archive_type)
        deadline = time.monotonic() + limits.extraction_timeout_seconds
        total_size = 0
        file_count = 0
        try:
            with zipfile.ZipFile(file_path) as archive:
                for member in archive.infolist():
                    if time.monotonic() > deadline:
                        raise IngestionError("archive extraction timed out")
                    name = _normalise_member_name(member.filename)
                    destination_path = _destination_for_member(target, name)
                    if member.is_dir():
                        destination_path.mkdir(parents=True, exist_ok=True)
                        continue
                    mode = (member.external_attr >> 16) & 0xFFFF
                    if stat.S_ISLNK(mode):
                        raise IngestionError("archive contains a symbolic link")
                    if member.flag_bits & 0x1:
                        raise ArchivePasswordError("archive is password protected")
                    file_count += 1
                    total_size = _check_member_limits(name, member.file_size, member.compress_size, file_count - 1, total_size, limits)
                    if _is_nested_archive(name) and not limits.allow_nested_archives:
                        result.nested_archive_count += 1
                        continue
                    if Path(name).suffix.lower() not in SUPPORTED_SUFFIXES:
                        result.unsupported_count += 1
                        continue
                    with archive.open(member, "r") as source:
                        _safe_copy_stream(source, destination_path, expected_size=member.file_size, deadline=deadline, limits=limits)
                    result.extracted_count += 1
        except ArchivePasswordError:
            shutil.rmtree(target, ignore_errors=True)
            raise
        except IngestionError:
            shutil.rmtree(target, ignore_errors=True)
            raise
        except (zipfile.BadZipFile, EOFError, OSError) as exc:
            shutil.rmtree(target, ignore_errors=True)
            raise ArchiveCorruptError("archive could not be read") from exc
        if result.unsupported_count:
            result.warnings.append(f"{result.unsupported_count} unsupported files were skipped.")
        if result.nested_archive_count:
            result.warnings.append("Nested archive detected and skipped.")
        if result.extracted_count == 0:
            shutil.rmtree(target, ignore_errors=True)
            raise IngestionError("archive contains no supported JPG, PNG, or PDF files")
        return result


class RarArchiveExtractor(ArchiveExtractor):
    archive_type = "rar"

    def extract(self, file_path: Path, destination: Path, limits: ArchiveLimits | None = None) -> ArchiveExtraction:
        if rarfile is None:
            raise IngestionError("RAR support is unavailable; install rarfile and a native RAR backend")
        limits = limits or ArchiveLimits()
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / f"rar-{uuid4()}"
        target.mkdir(parents=True, exist_ok=False)
        result = ArchiveExtraction(target, file_path.name, self.archive_type)
        deadline = time.monotonic() + limits.extraction_timeout_seconds
        total_size = 0
        file_count = 0
        try:
            with rarfile.RarFile(file_path) as archive:
                # RAR5 may encrypt the headers, so infolist() can be empty while
                # the archive still clearly requires a password.
                if archive.needs_password():
                    raise ArchivePasswordError("archive is password protected")
                for member in archive.infolist():
                    if time.monotonic() > deadline:
                        raise IngestionError("archive extraction timed out")
                    name = _normalise_member_name(member.filename)
                    destination_path = _destination_for_member(target, name)
                    if member.is_dir():
                        destination_path.mkdir(parents=True, exist_ok=True)
                        continue
                    if member.is_symlink() or getattr(member, "file_redir", None):
                        raise IngestionError("archive contains a symbolic link")
                    if member.needs_password():
                        raise ArchivePasswordError("archive is password protected")
                    file_count += 1
                    file_size = int(getattr(member, "file_size", 0) or 0)
                    compressed_size = int(getattr(member, "compress_size", 0) or 0)
                    total_size = _check_member_limits(name, file_size, compressed_size, file_count - 1, total_size, limits)
                    if _is_nested_archive(name) and not limits.allow_nested_archives:
                        result.nested_archive_count += 1
                        continue
                    if Path(name).suffix.lower() not in SUPPORTED_SUFFIXES:
                        result.unsupported_count += 1
                        continue
                    try:
                        with archive.open(member) as source:
                            _safe_copy_stream(source, destination_path, expected_size=file_size, deadline=deadline, limits=limits)
                    except (rarfile.PasswordRequired, rarfile.RarWrongPassword) as exc:
                        raise ArchivePasswordError("archive is password protected") from exc
                    except (rarfile.BadRarFile, rarfile.NeedFirstVolume, OSError) as exc:
                        raise ArchiveCorruptError("archive could not be read") from exc
                    result.extracted_count += 1
        except ArchivePasswordError:
            shutil.rmtree(target, ignore_errors=True)
            raise
        except ArchiveCorruptError:
            shutil.rmtree(target, ignore_errors=True)
            raise
        except IngestionError:
            shutil.rmtree(target, ignore_errors=True)
            raise
        except rarfile.RarCannotExec as exc:
            shutil.rmtree(target, ignore_errors=True)
            raise IngestionError("RAR support is unavailable; install a native RAR backend") from exc
        except rarfile.Error as exc:
            shutil.rmtree(target, ignore_errors=True)
            raise ArchiveCorruptError("archive could not be read") from exc
        except (rarfile.BadRarFile, rarfile.NotRarFile, rarfile.NeedFirstVolume, OSError) as exc:
            shutil.rmtree(target, ignore_errors=True)
            raise ArchiveCorruptError("archive could not be read") from exc
        if result.unsupported_count:
            result.warnings.append(f"{result.unsupported_count} unsupported files were skipped.")
        if result.nested_archive_count:
            result.warnings.append("Nested archive detected and skipped.")
        if result.extracted_count == 0:
            shutil.rmtree(target, ignore_errors=True)
            raise IngestionError("archive contains no supported JPG, PNG, or PDF files")
        return result


def _signature(file_path: Path) -> str | None:
    with file_path.open("rb") as handle:
        header = handle.read(8)
    if header.startswith(ZIP_SIGNATURES):
        return "zip"
    if header.startswith(RAR5_SIGNATURE) or header.startswith(RAR4_SIGNATURE):
        return "rar"
    return None


def detect_archive_type(file_path: Path, mime_type: str | None = None, *, raise_on_unknown: bool = True) -> str | None:
    """Detect ZIP/RAR from signature plus extension/MIME, never MIME alone."""
    suffix = file_path.suffix.lower()
    signature = _signature(file_path)
    mime = (mime_type or "").split(";", 1)[0].strip().lower()
    mime_kind = "rar" if mime in RAR_MIME_TYPES and mime != "application/octet-stream" else "zip" if mime in ZIP_MIME_TYPES and mime != "application/octet-stream" else None
    extension_kind = suffix.lstrip(".") if suffix in ARCHIVE_SUFFIXES else None
    if extension_kind and signature is None:
        if raise_on_unknown:
            raise IngestionError(f"{suffix} file has an invalid {suffix.lstrip('.').upper()} signature")
        return None
    if extension_kind and signature and extension_kind != signature:
        if raise_on_unknown:
            raise IngestionError(f"{suffix} file has an invalid {suffix.lstrip('.').upper()} signature")
        return None
    kind = signature or extension_kind
    if kind is None and mime_kind and raise_on_unknown:
        raise IngestionError("archive signature is missing or invalid")
    if kind is None and raise_on_unknown:
        raise IngestionError("unsupported or invalid archive")
    if mime_kind and kind and mime_kind != kind and mime != "application/octet-stream":
        if raise_on_unknown:
            raise IngestionError("archive MIME type does not match its file signature")
        return None
    return kind


def archive_extractor_for(file_path: Path, mime_type: str | None = None) -> ArchiveExtractor | None:
    kind = detect_archive_type(file_path, mime_type, raise_on_unknown=False)
    if kind == "zip":
        return ZipArchiveExtractor()
    if kind == "rar":
        return RarArchiveExtractor()
    return None


def extract_archive(file_path: Path, destination: Path, mime_type: str | None = None, limits: ArchiveLimits | None = None) -> ArchiveExtraction:
    extractor = archive_extractor_for(file_path, mime_type)
    if extractor is None:
        detect_archive_type(file_path, mime_type)
        raise IngestionError("unsupported or invalid archive")
    return extractor.extract(file_path, destination, limits)


def safely_extract_zip(zip_path: Path, destination: Path) -> Path:
    """Compatibility helper retained for trusted local CLI callers."""
    return ZipArchiveExtractor().extract(zip_path, destination).root


def patient_files(patient_folder: Path) -> list[Path]:
    if not patient_folder.is_dir():
        raise IngestionError(f"patient folder does not exist: {patient_folder}")
    files = [path for path in patient_folder.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES]
    if not files:
        raise IngestionError("patient folder contains no JPG, PNG, or PDF source files")
    # Natural path ordering keeps page_2 ahead of page_10 while retaining the
    # archive's relative-path order for files without an explicit sequence.
    def order_key(path: Path) -> tuple:
        parts = []
        for token in re.split(r"(\d+)", path.relative_to(patient_folder).as_posix().casefold()):
            parts.append((0, int(token)) if token.isdigit() else (1, token))
        return tuple(parts)

    return sorted(files, key=order_key)


def page_count(path: Path) -> tuple[int | None, list[str]]:
    if path.suffix.lower() != ".pdf":
        return 1, []
    try:
        return len(PdfReader(str(path)).pages), []
    except Exception:
        return None, ["PDF could not be parsed; original preserved for manual review"]


def ingest_folder(
    patient_id: str,
    patient_folder: Path,
    storage_root: Path,
    skip_checksums: set[str] | None = None,
    *,
    archive_filename: str | None = None,
    archive_type: str | None = None,
) -> list[IngestedFile]:
    original_dir = storage_root / patient_id / "original"
    original_dir.mkdir(parents=True, exist_ok=True)
    ingested: list[IngestedFile] = []
    seen: set[str] = set()
    for source_order, input_path in enumerate(patient_files(patient_folder)):
        checksum = checksum_file(input_path)
        if checksum in seen or (skip_checksums and checksum in skip_checksums):
            continue
        seen.add(checksum)
        file_id = str(uuid4())
        relative_name = input_path.relative_to(patient_folder).as_posix()
        destination = original_dir / f"{file_id}-{Path(relative_name).name}"
        copy2(input_path, destination)
        pages, warnings = page_count(destination)
        ingested.append(IngestedFile(
            source_file=SourceFile(
                file_id=file_id,
                original_filename=relative_name,
                file_type=input_path.suffix.lower().lstrip("."),
                page_count=pages,
                checksum=checksum,
                processing_status="ingested",
                original_path=str(destination),
                archive_filename=archive_filename,
                archive_type=archive_type,
                original_relative_path=relative_name if archive_filename else None,
                extracted_filename=relative_name if archive_filename else None,
                source_order=source_order,
            ),
            input_path=input_path,
            warnings=warnings,
        ))
    return ingested
