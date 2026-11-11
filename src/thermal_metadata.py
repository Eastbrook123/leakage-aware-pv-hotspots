"""Thermal-image compatibility layer.

Classifies an image as one of:
  * ``radiometric_dji``  - DJI R-JPEG with an embedded 16-bit raw thermal block (APP3)
  * ``radiometric_flir`` - FLIR radiometric JPEG (APP1 "FLIR" segments)
  * ``colorized_thermal`` - thermal camera image without a recoverable raw block
  * ``photographic``     - ordinary visible-light photo
  * ``unknown``          - cannot be determined

Important honesty rule: the DJI raw block contains *uncalibrated sensor counts*.
Converting counts to degrees Celsius requires DJI's proprietary Thermal SDK and the
measurement parameters (emissivity, distance, humidity, reflected temperature). SunSight
therefore never reports absolute temperatures; it only reports *relative* raw-count
contrast, which is monotonic with radiance inside a single image.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

# Models of DJI thermal cameras whose R-JPEG raw block we know the layout of (640x512 uint16 LE)
DJI_THERMAL_MODELS = {"M3T", "M30T", "ZH20T", "H20T", "H20N", "XT S", "M2EA", "M3TD", "H30T", "M4T", "M4TD"}
THERMAL_PALETTE_NAMES = {
    "whitehot", "blackhot", "ironred", "hotiron", "rainbow", "rainbow2", "fulgurite", "hotmetal",
    "medical", "arctic", "tint", "ironbow", "lava", "gray", "grey",
}

VISIBLE_SOURCES = {"WideCamera", "ZoomCamera", "VisibleCamera", "RGBCamera"}

THERMAL_CLASS_LABELS = {
    "radiometric_dji": "Radiometric thermal (DJI R-JPEG)",
    "radiometric_flir": "Radiometric thermal (FLIR)",
    "colorized_thermal": "Colorized thermal (no radiometric data)",
    "photographic": "Standard photographic image",
    "unknown": "Unknown / ambiguous",
}


@dataclass
class ThermalInfo:
    image_class: str = "unknown"
    image_class_label: str = THERMAL_CLASS_LABELS["unknown"]
    has_raw_block: bool = False
    raw_shape: Optional[tuple] = None
    palette: Optional[str] = None
    image_source: Optional[str] = None
    raw_min: Optional[int] = None
    raw_max: Optional[int] = None
    raw_p50: Optional[float] = None
    absolute_temperature_available: bool = False
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["raw_shape"] = list(self.raw_shape) if self.raw_shape else None
        return d


def iter_jpeg_segments(data: bytes):
    """Yield (marker, payload) for JPEG header segments up to Start-Of-Scan.

    Bounded and defensive: stops on malformed input instead of raising.
    """
    if len(data) < 4 or data[0:2] != b"\xff\xd8":
        return
    i = 2
    n = len(data)
    while i + 4 <= n and data[i] == 0xFF:
        marker = data[i + 1]
        if marker == 0xD8 or 0xD0 <= marker <= 0xD7:  # standalone markers
            i += 2
            continue
        if marker == 0xDA:  # start of scan - compressed data follows
            return
        length = int.from_bytes(data[i + 2:i + 4], "big")
        if length < 2 or i + 2 + length > n:
            return
        yield marker, data[i + 4:i + 2 + length]
        i += 2 + length


def extract_dji_raw(data: bytes, width: int, height: int) -> Optional[np.ndarray]:
    """Return the DJI R-JPEG raw thermal block as a (height, width) uint16 array, or None.

    DJI stores the raw sensor frame split over consecutive APP3 (0xFFE3) segments.
    The frame size must match width*height*2 exactly, otherwise None is returned.
    """
    chunks = [payload for marker, payload in iter_jpeg_segments(data) if marker == 0xE3]
    if not chunks:
        return None
    raw = b"".join(chunks)
    expected = width * height * 2
    if len(raw) < expected:
        return None
    arr = np.frombuffer(raw[:expected], dtype="<u2").reshape(height, width)
    return arr


def _xmp_value(data: bytes, key: str) -> Optional[str]:
    tag = (key + '="').encode()
    i = data.find(tag)
    if i < 0:
        return None
    j = data.find(b'"', i + len(tag))
    if j < 0 or j - i > 200:
        return None
    return data[i + len(tag):j].decode("utf-8", "replace")


def analyze_thermal(data: bytes, width: int, height: int, make: str | None, model: str | None,
                    description: str | None = None, rgb: np.ndarray | None = None,
                    want_raw: bool = False):
    """Classify the image and (optionally) return the raw thermal frame.

    Returns ``(ThermalInfo, raw_or_None)``.
    """
    info = ThermalInfo()
    make_u = (make or "").strip().upper()
    model_u = (model or "").strip().upper()
    info.image_source = _xmp_value(data, "drone-dji:ImageSource") or _xmp_value(data, "dji:ImageSource")
    desc = (description or "").strip()
    if desc and desc.replace(" ", "").lower() in THERMAL_PALETTE_NAMES:
        info.palette = desc

    raw = None
    if info.image_source and info.image_source in VISIBLE_SOURCES:
        info.image_class = "photographic"
        info.image_class_label = THERMAL_CLASS_LABELS["photographic"]
        info.notes.append(f"DJI metadata reports ImageSource={info.image_source}: a visible-light frame, not "
                          "thermal. The SunSight model was trained on thermal imagery only.")
        return info, None
    if make_u == "DJI" or b"DJI" in data[:4096] or info.image_source:
        raw = extract_dji_raw(data, width, height)
    if raw is not None:
        info.image_class = "radiometric_dji"
        info.has_raw_block = True
        info.raw_shape = tuple(raw.shape)
        info.raw_min = int(raw.min())
        info.raw_max = int(raw.max())
        info.raw_p50 = float(np.median(raw))
        info.notes.append(
            "16-bit raw thermal frame found (DJI R-JPEG). Values are uncalibrated sensor counts; "
            "absolute temperature (deg C) requires the DJI Thermal SDK and is NOT computed by SunSight."
        )
    elif b"FLIR\x00" in data[:200000]:
        info.image_class = "radiometric_flir"
        info.notes.append("FLIR radiometric segments detected. A FLIR decoder is not bundled; "
                          "the image is analysed visually only.")
    elif info.image_source == "InfraredCamera" or info.palette or model_u.endswith("T") and make_u == "DJI":
        info.image_class = "colorized_thermal"
        info.notes.append("Thermal camera image without a recoverable raw block: visual analysis only, "
                          "no temperature measurement is possible.")
    elif rgb is not None and _looks_like_thermal_palette(rgb):
        info.image_class = "colorized_thermal"
        info.notes.append("Image colours resemble a thermal palette (heuristic). No metadata confirms it; "
                          "treat as colorized thermal, visual analysis only.")
    elif make_u or model_u:
        info.image_class = "photographic"
        info.notes.append("Camera metadata present but no thermal markers: probably a visible-light photo. "
                          "The SunSight model was trained on thermal imagery only.")
    else:
        info.image_class = "unknown"
        info.notes.append("No camera metadata. Image type cannot be confirmed.")
    info.image_class_label = THERMAL_CLASS_LABELS[info.image_class]
    return info, (raw if want_raw else None)


def _looks_like_thermal_palette(rgb: np.ndarray) -> bool:
    """Weak heuristic: thermal palettes (iron/rainbow) are dominated by warm hues with low green-blue
    variance, or are near-grayscale (white-hot). Used only as a hint, never as proof."""
    small = rgb[::8, ::8].astype(np.float32)
    r, g, b = small[..., 0], small[..., 1], small[..., 2]
    gray_like = np.mean(np.abs(r - g) + np.abs(g - b)) < 6
    warm = np.mean(r) > np.mean(b) + 25 and np.mean(r) > np.mean(g)
    return bool(gray_like or warm)


def relative_contrast(raw: np.ndarray, box_xyxy, ring: int = 6) -> Optional[dict]:
    """Relative raw-count contrast of a detection box versus a surrounding ring.

    Returns the box p95, ring median and their difference *in raw sensor counts*, plus the
    difference expressed in robust standard deviations of the whole frame (a unitless z-score).
    This is a relative indicator for ranking only - it is not a temperature difference.
    """
    h, w = raw.shape
    x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, max(x2, x1 + 1)), min(h, max(y2, y1 + 1))
    inner = raw[y1:y2, x1:x2]
    if inner.size == 0:
        return None
    ox1, oy1 = max(0, x1 - ring), max(0, y1 - ring)
    ox2, oy2 = min(w, x2 + ring), min(h, y2 + ring)
    outer = raw[oy1:oy2, ox1:ox2].astype(np.float64)
    mask = np.ones(outer.shape, dtype=bool)
    mask[y1 - oy1:y2 - oy1, x1 - ox1:x2 - ox1] = False
    ring_vals = outer[mask]
    if ring_vals.size == 0:
        return None
    box_p95 = float(np.percentile(inner, 95))
    ring_med = float(np.median(ring_vals))
    flat = raw.astype(np.float64).ravel()[::4]
    spread = float(np.median(np.abs(flat - np.median(flat)))) * 1.4826
    if spread <= 0:  # nearly uniform frame: fall back to the plain standard deviation
        spread = float(np.std(flat))
    z = (box_p95 - ring_med) / spread if spread > 0 else None
    return {"box_p95_counts": box_p95, "ring_median_counts": ring_med,
            "delta_counts": box_p95 - ring_med, "contrast_z": z}


def raw_to_display(raw: np.ndarray) -> np.ndarray:
    """Percentile-stretch the raw frame to uint8 for visualisation."""
    lo, hi = np.percentile(raw, [1, 99.5])
    if hi <= lo:
        hi = lo + 1
    return (np.clip((raw.astype(np.float32) - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
