"""
Phase 1: Extract railroad spurs and sidings from OpenStreetMap.

Uses the Overpass API to query for railway spurs and sidings
within a target region and stores the results as GeoJSON.

Note: US OSM data uses railway=rail + service=spur/siding rather than
railway=spur. We query both patterns for completeness.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import requests

from config import (
    OVERPASS_RATE_LIMIT,
    OVERPASS_TIMEOUT,
    OVERPASS_URLS,
    RAW_DIR,
    REGIONS,
)
from src.utils.cache import FileCache
from src.utils.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

overpass_cache = FileCache("overpass")
overpass_limiter = RateLimiter(OVERPASS_RATE_LIMIT, name="Overpass")


def build_overpass_query_area(region_key: str) -> str:
    """Build Overpass query using area selector."""
    region = REGIONS[region_key]
    area_selector = region["overpass_area"]

    return f"""[out:json][timeout:{OVERPASS_TIMEOUT}];
area{area_selector}->.state;
(
  way["railway"="spur"](area.state);
  way["railway"="siding"](area.state);
  way["railway"="rail"]["service"="spur"](area.state);
  way["railway"="rail"]["service"="siding"](area.state);
);
out body;
>;
out skel qt;"""


def build_overpass_query_bbox(region_key: str) -> str:
    """Build Overpass query using bounding box (fallback)."""
    region = REGIONS[region_key]
    bbox = region["bbox"]  # (south, west, north, east)

    return f"""[out:json][timeout:{OVERPASS_TIMEOUT}];
(
  way["railway"="spur"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
  way["railway"="siding"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
  way["railway"="rail"]["service"="spur"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
  way["railway"="rail"]["service"="siding"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
);
out body;
>;
out skel qt;"""


def build_overpass_query(region_key: str) -> str:
    """
    Build an Overpass QL query for railroad spurs/sidings in a region.

    Uses bbox approach which is more reliable than area selectors
    (area selectors depend on Overpass area DB being up to date).
    """
    if region_key not in REGIONS:
        raise KeyError(f"Unknown region: {region_key}")
    return build_overpass_query_bbox(region_key)


def build_mainline_nodes_query(region_key: str) -> str:
    """Build query to get mainline node IDs for junction detection."""
    region = REGIONS[region_key]
    bbox = region["bbox"]

    # Fetch mainline ways and expand to their nodes
    return f"""[out:json][timeout:{OVERPASS_TIMEOUT}];
way["railway"="rail"]["usage"="main"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
out ids;
>;
out ids;"""


def extract_mainline_node_ids(raw_data: dict) -> set:
    """Extract node IDs from a mainline-only Overpass query result."""
    node_ids = set()
    for element in raw_data.get("elements", []):
        if element["type"] == "node":
            node_ids.add(element["id"])
    return node_ids


def query_overpass(query: str) -> dict:
    """
    Execute an Overpass API query with caching, rate limiting, and server fallback.

    Tries multiple Overpass servers if the first one fails.
    """
    cache_params = {"query": query}

    cached = overpass_cache.get(cache_params)
    if cached is not None:
        logger.info("Using cached Overpass response")
        return cached

    last_error = None
    for url in OVERPASS_URLS:
        overpass_limiter.wait()

        try:
            logger.info(f"Querying Overpass API at {url}...")
            response = requests.post(
                url,
                data={"data": query},
                timeout=OVERPASS_TIMEOUT + 30,
            )
            response.raise_for_status()

            data = response.json()
            element_count = len(data.get("elements", []))
            logger.info(f"Overpass returned {element_count} elements")

            if element_count > 0:
                overpass_cache.set(cache_params, data)
                return data

            logger.warning(f"Got 0 elements from {url}, trying next server...")

        except (requests.RequestException, ValueError) as e:
            logger.warning(f"Overpass server {url} failed: {e}")
            last_error = e
            continue

    # If all servers returned 0 elements but didn't error, return last response
    if last_error is None:
        logger.warning("All Overpass servers returned 0 elements")
        overpass_cache.set(cache_params, data)
        return data

    raise last_error


def parse_osm_elements(raw_data: dict) -> Tuple[Dict[int, dict], Dict[int, dict]]:
    """
    Parse raw Overpass response into ways and nodes.

    Returns:
        Tuple of (ways_by_id, nodes_by_id).
    """
    ways: Dict[int, dict] = {}
    nodes: Dict[int, dict] = {}

    for element in raw_data.get("elements", []):
        if element["type"] == "way":
            ways[element["id"]] = {
                "id": element["id"],
                "tags": element.get("tags", {}),
                "node_ids": element.get("nodes", []),
            }
        elif element["type"] == "node":
            nodes[element["id"]] = {
                "id": element["id"],
                "lat": element["lat"],
                "lon": element["lon"],
            }

    logger.info(f"Parsed {len(ways)} ways and {len(nodes)} nodes")
    return ways, nodes


def ways_to_geojson(ways: Dict[int, dict], nodes: Dict[int, dict]) -> dict:
    """Convert parsed OSM ways to GeoJSON FeatureCollection."""
    features = []

    for way_id, way in ways.items():
        coords = []
        for node_id in way["node_ids"]:
            node = nodes.get(node_id)
            if node:
                coords.append([node["lon"], node["lat"]])  # GeoJSON is [lon, lat]

        if len(coords) < 2:
            logger.debug(f"Skipping way {way_id}: insufficient coordinates")
            continue

        feature = {
            "type": "Feature",
            "properties": {
                "osm_way_id": way_id,
                "railway": way["tags"].get("railway", ""),
                "name": way["tags"].get("name", ""),
                "operator": way["tags"].get("operator", ""),
                **{k: v for k, v in way["tags"].items() if k not in ("railway", "name", "operator")},
            },
            "geometry": {
                "type": "LineString",
                "coordinates": coords,
            },
        }
        features.append(feature)

    return {
        "type": "FeatureCollection",
        "features": features,
    }


def extract_rail_network(region_key: str) -> dict:
    """
    Full extraction pipeline: query OSM and return GeoJSON.

    Args:
        region_key: Key into config.REGIONS dict.

    Returns:
        GeoJSON FeatureCollection of spurs and sidings.
    """
    region = REGIONS[region_key]
    logger.info(f"Extracting rail network for {region['name']}")

    query = build_overpass_query(region_key)
    raw_data = query_overpass(query)

    ways, nodes = parse_osm_elements(raw_data)
    geojson = ways_to_geojson(ways, nodes)

    # Save to file
    output_path = RAW_DIR / f"{region_key}_rail_spurs.geojson"
    with open(output_path, "w") as f:
        json.dump(geojson, f)

    spur_count = sum(
        1 for feat in geojson["features"]
        if feat["properties"].get("railway") == "spur"
        or feat["properties"].get("service") == "spur"
    )
    siding_count = sum(
        1 for feat in geojson["features"]
        if feat["properties"].get("railway") == "siding"
        or feat["properties"].get("service") == "siding"
    )

    logger.info(
        f"Saved {len(geojson['features'])} features "
        f"({spur_count} spurs, {siding_count} sidings) to {output_path}"
    )
    return geojson
