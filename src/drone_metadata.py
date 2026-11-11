"""Camera / drone metadata extraction (EXIF + DJI XMP).

Only values physically present in the file are returned. Missing values stay ``None`` and are
listed in ``missing``; nothing is inferred or invented.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional

from PIL import Image
from PIL.ExifTags import TAGS

_XMP_RE = re.compile(rb'(?:drone-dji|dji):(\w+)="([^"]{0,120})"')


@dataclass
class DroneMetadata:
    make: Optional[str] = None
    model: Optional[str] = None
    drone_model: Optional[str] = None
    software: Optional[str] = None
    image_description: Optional[str] = None
    capture_time_local: Optional[str] = None
    capture_time_utc: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    gps_latitude: Optional[float] = None
    gps_longitude: Optional[float] = None
    gps_altitude_m: Optional[float] = None       # absolute altitude (EXIF / DJI AbsoluteAltitude)
    relative_altitude_m: Optional[float] = None  # height above take-off point (DJI)
    gimbal_pitch_deg: Optional[float] = None
    gimbal_yaw_deg: Optional[float] = None
    gimbal_roll_deg: Optional[float] = None
    flight_yaw_deg: Optional[float] = None
    image_source: Optional[str] = None
    gps_status: Optional[str] = None
    lrf_target_distance_m: Optional[float] = None
    camera_serial: Optional[str] = None
    drone_serial: Optional[str] = None
    missing: list = field(default_factory=list)

    @property
    def has_gps(self) -> bool:
        return self.gps_latitude is not None and self.gps_longitude is not None

    def to_dict(self, mask_serials: bool = True) -> dict:
        d = asdict(self)
        if mask_serials:
            for k in ("camera_serial", "drone_serial"):
                if d.get(k):
                    d[k] = mask_serial(d[k])
        d["has_gps"] = self.has_gps
        return d


def mask_serial(s: str) -> str:
    s = str(s)
    return s[:3] + "*" * max(0, len(s) - 6) + s[-3:] if len(s) > 6 else "***"


def _dms_to_deg(dms, ref) -> Optional[float]:
    try:
        d, m, s = [float(x) for x in dms]
        val = d + m / 60.0 + s / 3600.0
        if str(ref).strip().upper() in ("S", "W"):
            val = -val
        return round(val, 8)
    except Exception:
        return None


def _f(v) -> Optional[float]:
    try:
        return float(str(v).strip().replace("+", ""))
    except Exception:
        return None


def extract_metadata(data: bytes, img: Image.Image | None = None) -> DroneMetadata:
    """Extract EXIF and DJI XMP metadata from raw image bytes."""
    md = DroneMetadata()
    try:
        img = img or Image.open(io.BytesIO(data))
        md.width, md.height = img.size
        exif = img.getexif()
    except Exception:
        exif = {}
    named = {TAGS.get(k, k): v for k, v in dict(exif).items()} if exif else {}
    md.make = _clean(named.get("Make"))
    md.model = _clean(named.get("Model"))
    md.software = _clean(named.get("Software"))
    md.image_description = _clean(named.get("ImageDescription"))
    dt = _clean(named.get("DateTime"))
    if exif:
        try:
            sub = exif.get_ifd(0x8769)
            dt = _clean(sub.get(36867)) or dt  # DateTimeOriginal
        except Exception:
            pass
    if dt:
        try:
            md.capture_time_local = datetime.strptime(dt, "%Y:%m:%d %H:%M:%S").isoformat()
        except ValueError:
            md.capture_time_local = dt
    if exif:
        try:
            gps = exif.get_ifd(0x8825)
        except Exception:
            gps = {}
        if gps and 2 in gps and 4 in gps:
            md.gps_latitude = _dms_to_deg(gps[2], gps.get(1, "N"))
            md.gps_longitude = _dms_to_deg(gps[4], gps.get(3, "E"))
            if md.gps_latitude == 0 and md.gps_longitude == 0:  # null island = no fix
                md.gps_latitude = md.gps_longitude = None
            if 6 in gps:
                alt = _f(gps[6])
                if alt is not None and gps.get(5) in (1, b"\x01"):
                    alt = -alt
                md.gps_altitude_m = alt

    xmp = {k.decode(): v.decode("utf-8", "replace") for k, v in _XMP_RE.findall(data[:400000])}
    if xmp:
        if md.gps_latitude is None and _f(xmp.get("GpsLatitude")) not in (None, 0.0):
            md.gps_latitude = _f(xmp.get("GpsLatitude"))
            md.gps_longitude = _f(xmp.get("GpsLongitude") or xmp.get("GpsLongtitude"))
        md.gps_altitude_m = md.gps_altitude_m if md.gps_altitude_m is not None else _f(xmp.get("AbsoluteAltitude"))
        md.relative_altitude_m = _f(xmp.get("RelativeAltitude"))
        md.gimbal_pitch_deg = _f(xmp.get("GimbalPitchDegree"))
        md.gimbal_yaw_deg = _f(xmp.get("GimbalYawDegree"))
        md.gimbal_roll_deg = _f(xmp.get("GimbalRollDegree"))
        md.flight_yaw_deg = _f(xmp.get("FlightYawDegree"))
        md.drone_model = _clean(xmp.get("DroneModel"))
        md.image_source = _clean(xmp.get("ImageSource"))
        md.gps_status = _clean(xmp.get("GpsStatus"))
        lrf = _f(xmp.get("LRFTargetDistance"))
        md.lrf_target_distance_m = lrf if lrf else None
        md.camera_serial = _clean(xmp.get("CameraSerialNumber"))
        md.drone_serial = _clean(xmp.get("DroneSerialNumber"))
        utc = _clean(xmp.get("UTCAtExposure"))
        md.capture_time_utc = utc

    for key in ("make", "model", "capture_time_local", "gps_latitude", "gps_longitude",
                "relative_altitude_m", "gimbal_pitch_deg", "gimbal_yaw_deg"):
        if getattr(md, key) is None:
            md.missing.append(key)
    return md


def _clean(v) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, bytes):
        v = v.decode("utf-8", "replace")
    s = str(v).replace("\x00", "").strip()
    return s or None
