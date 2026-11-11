"""Validation for uploaded images and YOLO annotation files.

Uploaded content is treated as untrusted: format is checked by magic bytes, size and pixel
count are bounded (decompression-bomb guard), and the decoder is run inside try/except.
Nothing uploaded is ever executed or written outside the outputs folder.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image

MAGIC = {
    b"\xff\xd8\xff": "JPEG",
    b"\x89PNG\r\n\x1a\n": "PNG",
    b"II*\x00": "TIFF",
    b"MM\x00*": "TIFF",
    b"BM": "BMP",
}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


@dataclass
class ValidationResult:
    ok: bool
    filename: str
    format: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    size_bytes: int = 0
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def sniff_format(data: bytes) -> Optional[str]:
    for sig, name in MAGIC.items():
        if data.startswith(sig):
            return name
    return None


def validate_image_bytes(data: bytes, filename: str, max_mb: float = 25.0,
                         max_pixels: int = 40_000_000, min_side: int = 32) -> ValidationResult:
    res = ValidationResult(ok=False, filename=filename, size_bytes=len(data))
    ext = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        res.errors.append(f"Unsupported file extension '{ext or 'none'}'. Allowed: "
                          + ", ".join(sorted(ALLOWED_EXTENSIONS)))
    if len(data) == 0:
        res.errors.append("File is empty.")
        return res
    if len(data) > max_mb * 1024 * 1024:
        res.errors.append(f"File is {len(data) / 1048576:.1f} MB; limit is {max_mb:.0f} MB.")
        return res
    fmt = sniff_format(data)
    if fmt is None:
        res.errors.append("File content is not a recognised image (magic-byte check failed).")
        return res
    res.format = fmt
    try:
        with Image.open(io.BytesIO(data)) as im:
            w, h = im.size
            if w * h > max_pixels:
                res.errors.append(f"Image has {w * h:,} pixels; limit is {max_pixels:,}.")
                return res
            im.verify()  # structural check
        with Image.open(io.BytesIO(data)) as im:
            im.load()  # full decode catches truncated files
            res.width, res.height = im.size
    except Image.DecompressionBombError:
        res.errors.append("Image rejected: decompression-bomb protection triggered.")
        return res
    except Exception as exc:  # corrupted / truncated
        res.errors.append(f"Image could not be decoded ({type(exc).__name__}).")
        return res
    if min(res.width, res.height) < min_side:
        res.errors.append(f"Image is too small ({res.width}x{res.height}).")
    if res.width and res.height and (res.width, res.height) != (640, 512):
        res.warnings.append(f"Resolution {res.width}x{res.height} differs from the 640x512 thermal "
                            "frames the model was trained on; results may be less reliable.")
    res.ok = not res.errors
    return res


@dataclass
class LabelCheck:
    n_boxes: int = 0
    boxes: list = field(default_factory=list)  # (cls, xc, yc, w, h)
    errors: list = field(default_factory=list)


def parse_yolo_label(text: str, n_classes: int = 1) -> LabelCheck:
    """Parse and validate a YOLO detection label file (class xc yc w h, normalised)."""
    out = LabelCheck()
    for ln, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            out.errors.append(f"line {ln}: expected 5 values, got {len(parts)}"
                              + (" (polygon/segmentation?)" if len(parts) > 5 else ""))
            continue
        try:
            c = int(float(parts[0]))
            xc, yc, w, h = map(float, parts[1:])
        except ValueError:
            out.errors.append(f"line {ln}: non-numeric value")
            continue
        if not (0 <= c < n_classes):
            out.errors.append(f"line {ln}: class id {c} out of range")
        if not all(0.0 <= v <= 1.0 for v in (xc, yc, w, h)):
            out.errors.append(f"line {ln}: coordinates outside [0,1]")
        if w <= 0 or h <= 0:
            out.errors.append(f"line {ln}: zero/negative box size")
        if xc - w / 2 < -1e-3 or xc + w / 2 > 1 + 1e-3 or yc - h / 2 < -1e-3 or yc + h / 2 > 1 + 1e-3:
            out.errors.append(f"line {ln}: box extends outside image")
        out.boxes.append((c, xc, yc, w, h))
    out.n_boxes = len(out.boxes)
    return out
