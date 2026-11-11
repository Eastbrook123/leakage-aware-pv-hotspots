"""Geospatial helpers for drone inspections.

Four different locations exist and SunSight keeps them separate:
  1. Image GPS location   - the GPS fix written into the image (for DJI: the aircraft position)
  2. Drone position       - same as (1) for DJI thermal frames
  3. Estimated view point - where the optical axis meets the ground, *estimated* from relative
                            altitude + gimbal pitch/yaw under a flat-terrain assumption
  4. Hotspot location     - NOT computed: needs camera intrinsics, terrain model and precise
                            georeferencing; SunSight does not claim exact hotspot geolocation.
"""
from __future__ import annotations

import math
from typing import Optional

EARTH_R = 6_378_137.0


def offset_latlon(lat: float, lon: float, north_m: float, east_m: float) -> tuple[float, float]:
    dlat = north_m / EARTH_R
    dlon = east_m / (EARTH_R * math.cos(math.radians(lat)))
    return lat + math.degrees(dlat), lon + math.degrees(dlon)


def to_local_enu(lat: float, lon: float, lat0: float, lon0: float) -> tuple[float, float]:
    """Equirectangular approximation (metres east, north) - accurate to <0.1% over a solar farm."""
    east = math.radians(lon - lon0) * EARTH_R * math.cos(math.radians(lat0))
    north = math.radians(lat - lat0) * EARTH_R
    return east, north


def estimate_view_point(lat: Optional[float], lon: Optional[float], rel_alt_m: Optional[float],
                        gimbal_pitch_deg: Optional[float], gimbal_yaw_deg: Optional[float],
                        max_range_m: float = 300.0) -> Optional[dict]:
    """Estimate the ground point at the image centre (flat terrain, take-off-level ground).

    Returns None if any input is missing or the camera looks above the horizon.
    """
    if None in (lat, lon, rel_alt_m, gimbal_pitch_deg, gimbal_yaw_deg):
        return None
    pitch = -gimbal_pitch_deg  # DJI: negative = looking down
    if pitch <= 1.0 or rel_alt_m <= 0:
        return None
    ground_dist = rel_alt_m / math.tan(math.radians(pitch))
    if ground_dist > max_range_m:
        return None
    yaw = math.radians(gimbal_yaw_deg)
    north, east = ground_dist * math.cos(yaw), ground_dist * math.sin(yaw)
    vlat, vlon = offset_latlon(lat, lon, north, east)
    return {"lat": round(vlat, 8), "lon": round(vlon, 8), "ground_distance_m": round(ground_dist, 1),
            "assumption": "flat terrain at take-off altitude; image-centre only; not a hotspot position"}


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R * math.asin(math.sqrt(a))
