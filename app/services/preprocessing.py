from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from shutil import copy2

from PIL import Image, ImageChops, ImageOps, UnidentifiedImageError

from app.services.ingestion import IngestedFile


@dataclass
class PreprocessResult:
    processed_path: Path
    warnings: list[str] = field(default_factory=list)


def _crop_uniform_margin(image: Image.Image) -> Image.Image:
    """Trim only an outer area matching the top-left background color in the derivative."""
    background = Image.new(image.mode, image.size, image.getpixel((0, 0)))
    bounding_box = ImageChops.difference(image, background).getbbox()
    if not bounding_box:
        return image
    left, top, right, bottom = bounding_box
    width, height = image.size
    # Refuse aggressive crops: edge handwriting or stamps must remain in the derived image.
    if left > width * 0.12 or top > height * 0.12 or right < width * 0.88 or bottom < height * 0.88:
        return image
    return image.crop(bounding_box)


def _possible_glare(image: Image.Image) -> bool:
    """A cautious clipping heuristic; a flag asks for review and does not change pixels."""
    sample = image.convert("RGB").resize((160, 160))
    pixels = list(sample.getdata())
    bright = sum(1 for red, green, blue in pixels if red > 252 and green > 252 and blue > 252) / len(pixels)
    non_white = sum(1 for red, green, blue in pixels if min(red, green, blue) < 220) / len(pixels)
    return 0.08 <= bright <= 0.65 and non_white >= 0.10


def preprocess(source: IngestedFile, processed_root: Path) -> PreprocessResult:
    """Create a derived file only; never mutate a forensic original."""
    destination = processed_root / f"{source.source_file.file_id}-{source.source_file.original_filename}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = source.input_path.suffix.lower()
    if suffix == ".pdf":
        copy2(source.source_file.original_path, destination)
        return PreprocessResult(destination)
    try:
        with Image.open(source.source_file.original_path) as image:
            image = ImageOps.exif_transpose(image)
            width, height = image.size
            warnings: list[str] = []
            if min(width, height) < 800:
                warnings.append("low-resolution image; extraction requires verification")
            # Conservative contrast enhancement preserves pixels spatially and leaves the original untouched.
            enhanced = ImageOps.autocontrast(image.convert("RGB"), cutoff=0.2)
            enhanced = _crop_uniform_margin(enhanced)
            if _possible_glare(enhanced):
                warnings.append("possible glare detected; verify affected text against original")
            enhanced.save(destination)
            return PreprocessResult(destination, warnings)
    except (UnidentifiedImageError, OSError):
        copy2(source.source_file.original_path, destination)
        return PreprocessResult(destination, ["image could not be preprocessed; requires manual review"])
