"""Image decoding, working-copy preprocessing and result annotation.

The uploaded bytes are never modified; every operation works on a decoded copy.
"""
from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image, ImageOps

NAVY = (32, 44, 56)
AMBER = (233, 162, 59)
CREAM = (255, 249, 237)
# Boxes use cyan: it is absent from thermal palettes (IronRed, white-hot, rainbow), so tiny boxes stay visible.
BOX = (0, 229, 255)
PRIORITY_COLORS = {"High": (214, 69, 65), "Medium": (233, 162, 59), "Low": (82, 107, 128)}


def decode_rgb(data: bytes) -> tuple[np.ndarray, dict]:
    """Decode bytes to an 8-bit RGB working copy. Returns (rgb, notes)."""
    notes = {}
    with Image.open(io.BytesIO(data)) as im:
        im.seek(0)  # MPO / multi-frame: the first frame is the primary image
        notes["source_mode"] = im.mode
        im = ImageOps.exif_transpose(im)
        if im.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
            arr = np.asarray(im, dtype=np.float32)
            lo, hi = np.percentile(arr, [1, 99.5])
            arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1) * 255
            notes["converted"] = "16-bit/float image percentile-stretched to 8-bit grayscale for the model"
            rgb = np.stack([arr.astype(np.uint8)] * 3, axis=-1)
        else:
            rgb = np.array(im.convert("RGB"))
    return np.ascontiguousarray(rgb), notes


def to_png_bytes(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def to_jpeg_bytes(rgb: np.ndarray, quality: int = 90) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def draw_detections(rgb: np.ndarray, detections: list, color=BOX, show_labels: bool = True) -> np.ndarray:
    """Draw detection boxes. Hotspots are often < 15 px, so each box also gets an outer
    corner bracket to stay visible at presentation scale."""
    out = rgb.copy()
    h, w = out.shape[:2]
    scale = max(1.0, w / 640)
    th = max(1, int(round(1.5 * scale)))
    for i, d in enumerate(detections, 1):
        x1, y1, x2, y2 = int(d.x1), int(d.y1), int(round(d.x2)), int(round(d.y2))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, th, cv2.LINE_AA)
        pad = int(6 * scale)
        L = int(6 * scale)
        X1, Y1, X2, Y2 = x1 - pad, y1 - pad, x2 + pad, y2 + pad
        for (cx, cy, dx, dy) in ((X1, Y1, 1, 1), (X2, Y1, -1, 1), (X1, Y2, 1, -1), (X2, Y2, -1, -1)):
            cv2.line(out, (cx, cy), (cx + dx * L, cy), NAVY, th + 1, cv2.LINE_AA)
            cv2.line(out, (cx, cy), (cx, cy + dy * L), NAVY, th + 1, cv2.LINE_AA)
        if show_labels:
            label = f"#{i} {d.confidence:.2f}"
            fs = 0.42 * scale
            (tw, tht), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
            ty = Y1 - 4 if Y1 - tht - 6 > 0 else Y2 + tht + 6
            tx = min(max(0, X1), w - tw - 4)
            cv2.rectangle(out, (tx, ty - tht - 4), (tx + tw + 4, ty + 3), NAVY, -1)
            cv2.putText(out, label, (tx + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, CREAM, 1, cv2.LINE_AA)
    return out


def crop_zoom(rgb: np.ndarray, det, context: int = 28, out_size: int = 160) -> np.ndarray:
    """Zoomed crop around a detection (nearest-neighbour, so pixels are not invented)."""
    h, w = rgb.shape[:2]
    cx, cy = det.center
    half = max(context, int(max(det.width, det.height)))
    x1, y1 = int(max(0, cx - half)), int(max(0, cy - half))
    x2, y2 = int(min(w, cx + half)), int(min(h, cy + half))
    crop = rgb[y1:y2, x1:x2].copy()
    bx1, by1 = int(det.x1 - x1), int(det.y1 - y1)
    bx2, by2 = int(round(det.x2 - x1)), int(round(det.y2 - y1))
    crop = cv2.resize(crop, (out_size, int(out_size * crop.shape[0] / max(crop.shape[1], 1))),
                      interpolation=cv2.INTER_NEAREST)
    sx = crop.shape[1] / max(x2 - x1, 1)
    sy = crop.shape[0] / max(y2 - y1, 1)
    cv2.rectangle(crop, (int(bx1 * sx), int(by1 * sy)), (int(bx2 * sx), int(by2 * sy)), BOX, 2)
    return crop


def thermal_heatmap(raw_u8: np.ndarray) -> np.ndarray:
    """Colour-map an 8-bit stretched raw frame (inferno) for display; returns RGB."""
    return cv2.cvtColor(cv2.applyColorMap(raw_u8, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)


def thumbnail(rgb: np.ndarray, width: int = 160) -> np.ndarray:
    h, w = rgb.shape[:2]
    return cv2.resize(rgb, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
