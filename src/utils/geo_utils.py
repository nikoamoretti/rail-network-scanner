"""
Geographic utility functions.

Distance calculations, bounding boxes, coordinate transforms.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

Coord = Tuple[float, float]  # (lat, lon)


def haversine_distance(coord1: Coord, coord2: Coord) -> float:
    """
    Calculate the great-circle distance between two points in meters.

    Args:
        coord1: (lat, lon) in degrees.
        coord2: (lat, lon) in degrees.

    Returns:
        Distance in meters.
    """
    R = 6_371_000  # Earth radius in meters

    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))

    return R * c


def way_length(nodes: List[Coord]) -> float:
    """
    Calculate total length of a polyline in meters.

    Args:
        nodes: List of (lat, lon) coordinates forming the polyline.

    Returns:
        Total length in meters.
    """
    total = 0.0
    for i in range(len(nodes) - 1):
        total += haversine_distance(nodes[i], nodes[i + 1])
    return total


def midpoint(nodes: List[Coord]) -> Coord:
    """Return the geographic midpoint of a list of coordinates."""
    if not nodes:
        raise ValueError("Cannot compute midpoint of empty list")
    avg_lat = sum(c[0] for c in nodes) / len(nodes)
    avg_lon = sum(c[1] for c in nodes) / len(nodes)
    return (avg_lat, avg_lon)


def bbox_for_state(state_name: str) -> Optional[Tuple[float, float, float, float]]:
    """
    Return approximate (south, west, north, east) bounding box for a US state.

    Used as fallback if Overpass area query fails.
    """
    boxes = {
        "illinois": (36.97, -91.51, 42.51, -87.02),
        "indiana": (37.77, -88.10, 41.76, -84.78),
        "ohio": (38.40, -84.82, 42.32, -80.52),
        "texas": (25.84, -106.65, 36.50, -93.51),
    }
    return boxes.get(state_name.lower())
