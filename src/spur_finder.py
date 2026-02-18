"""
Phase 2: Extract facility-side spur/siding endpoint coordinates.

For each spur, determines which end connects to the facility (not the mainline)
by checking if endpoint nodes are shared with mainline railway ways.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

from config import SPURS_DIR
from src.utils.geo_utils import Coord, haversine_distance, midpoint, way_length

logger = logging.getLogger(__name__)


@dataclass
class SpurEndpoint:
    """A facility-side spur endpoint with metadata."""
    lat: float
    lon: float
    osm_way_id: int
    railway_type: str  # "spur" or "siding"
    spur_length_m: float
    connected_to_mainline: bool
    endpoint_method: str  # how we determined facility side: "mainline_exclusion", "midpoint"
    name: str = ""
    operator: str = ""
    tags: dict = field(default_factory=dict)

    @property
    def coord(self) -> Coord:
        return (self.lat, self.lon)


def identify_mainline_nodes(raw_data: dict) -> set[int]:
    """
    Identify all node IDs that belong to mainline railway ways.

    This queries the raw Overpass data to find nodes shared with
    railway=rail (mainline) ways. Since our query only fetches spurs/sidings,
    we use the node sharing between ways as a heuristic — nodes that appear
    as the first or last node in multiple spur ways are likely junction nodes.

    For more accurate results, this function also accepts pre-fetched
    mainline data if available.

    Args:
        raw_data: Raw Overpass JSON (should include mainline ways if fetched).

    Returns:
        Set of node IDs that belong to mainline tracks.
    """
    mainline_nodes: set[int] = set()

    for element in raw_data.get("elements", []):
        if element["type"] == "way":
            tags = element.get("tags", {})
            railway = tags.get("railway", "")
            service = tags.get("service", "")
            usage = tags.get("usage", "")
            # Mainline = railway=rail without service=spur/siding
            is_mainline = (
                railway in ("rail", "main", "mainline")
                and service not in ("spur", "siding", "yard")
                and usage in ("", "main", "branch")
            )
            if is_mainline:
                for node_id in element.get("nodes", []):
                    mainline_nodes.add(node_id)

    return mainline_nodes


def find_junction_nodes(ways: dict[int, dict]) -> set[int]:
    """
    Find nodes that are shared between multiple ways (likely junction points).

    A node that appears as the first or last node of multiple ways
    is likely a junction where a spur connects to another track.

    Args:
        ways: Ways dict keyed by way ID.

    Returns:
        Set of node IDs that appear at endpoints of multiple ways.
    """
    endpoint_counts: dict[int, int] = {}

    for way in ways.values():
        node_ids = way["node_ids"]
        if not node_ids:
            continue

        first_node = node_ids[0]
        last_node = node_ids[-1]

        endpoint_counts[first_node] = endpoint_counts.get(first_node, 0) + 1
        endpoint_counts[last_node] = endpoint_counts.get(last_node, 0) + 1

    return {node_id for node_id, count in endpoint_counts.items() if count >= 2}


def extract_spur_endpoints(
    ways: dict[int, dict],
    nodes: dict[int, dict],
    mainline_nodes: set[int] | None = None,
) -> list[SpurEndpoint]:
    """
    Extract the facility-side endpoint for each spur/siding way.

    Algorithm:
    1. For each spur way, get first and last node.
    2. Check if either node is shared with mainline (junction node).
    3. The non-junction node is the facility-side endpoint.
    4. If both or neither are shared, use the midpoint.

    Args:
        ways: Parsed ways from osm_extractor.parse_osm_elements.
        nodes: Parsed nodes from osm_extractor.parse_osm_elements.
        mainline_nodes: Optional set of known mainline node IDs.

    Returns:
        List of SpurEndpoint objects.
    """
    # Build junction set from shared endpoints between ways
    junction_nodes = find_junction_nodes(ways)

    # Merge with mainline nodes if available
    if mainline_nodes:
        junction_nodes |= mainline_nodes
        logger.info(f"Using {len(mainline_nodes)} mainline nodes + {len(junction_nodes)} junction nodes")
    else:
        logger.info(f"Using {len(junction_nodes)} junction nodes (no mainline data)")

    endpoints: list[SpurEndpoint] = []
    skipped = 0

    for way_id, way in ways.items():
        node_ids = way["node_ids"]
        tags = way.get("tags", {})

        if len(node_ids) < 2:
            skipped += 1
            continue

        # Resolve coordinates for all nodes in this way
        way_coords: list[Coord] = []
        for nid in node_ids:
            node = nodes.get(nid)
            if node:
                way_coords.append((node["lat"], node["lon"]))

        if len(way_coords) < 2:
            skipped += 1
            continue

        first_id = node_ids[0]
        last_id = node_ids[-1]

        first_is_junction = first_id in junction_nodes
        last_is_junction = last_id in junction_nodes

        # Determine facility-side endpoint
        if first_is_junction and not last_is_junction:
            # Last node is facility side
            facility_coord = way_coords[-1]
            method = "mainline_exclusion"
            connected = True
        elif last_is_junction and not first_is_junction:
            # First node is facility side
            facility_coord = way_coords[0]
            method = "mainline_exclusion"
            connected = True
        else:
            # Ambiguous — use midpoint
            facility_coord = midpoint(way_coords)
            method = "midpoint"
            connected = first_is_junction or last_is_junction

        spur_len = way_length(way_coords)

        endpoints.append(SpurEndpoint(
            lat=facility_coord[0],
            lon=facility_coord[1],
            osm_way_id=way_id,
            # US OSM uses railway=rail + service=spur/siding
            railway_type=tags.get("service", tags.get("railway", "spur")),
            spur_length_m=round(spur_len, 1),
            connected_to_mainline=connected,
            endpoint_method=method,
            name=tags.get("name", ""),
            operator=tags.get("operator", ""),
            tags={k: v for k, v in tags.items() if k not in ("railway", "name", "operator")},
        ))

    logger.info(
        f"Extracted {len(endpoints)} spur endpoints "
        f"({skipped} skipped, "
        f"{sum(1 for e in endpoints if e.endpoint_method == 'mainline_exclusion')} by mainline exclusion, "
        f"{sum(1 for e in endpoints if e.endpoint_method == 'midpoint')} by midpoint)"
    )

    return endpoints


def deduplicate_nearby_endpoints(
    endpoints: list[SpurEndpoint], min_distance_m: float = 50.0
) -> list[SpurEndpoint]:
    """
    Remove endpoints that are very close together (likely same facility).

    Keeps the endpoint from the longer spur.

    Args:
        endpoints: List of SpurEndpoint objects.
        min_distance_m: Minimum distance between kept endpoints.

    Returns:
        Deduplicated list.
    """
    if not endpoints:
        return []

    # Sort by spur length descending — prefer longer spurs
    sorted_eps = sorted(endpoints, key=lambda e: e.spur_length_m, reverse=True)
    kept: list[SpurEndpoint] = []

    for ep in sorted_eps:
        too_close = False
        for existing in kept:
            dist = haversine_distance(ep.coord, existing.coord)
            if dist < min_distance_m:
                too_close = True
                break
        if not too_close:
            kept.append(ep)

    removed = len(endpoints) - len(kept)
    if removed > 0:
        logger.info(f"Deduplicated {removed} nearby endpoints (within {min_distance_m}m)")

    return kept


def save_endpoints(endpoints: list[SpurEndpoint], region_key: str) -> Path:
    """
    Save spur endpoints to JSON file.

    Args:
        endpoints: List of SpurEndpoint objects.
        region_key: Region identifier for filename.

    Returns:
        Path to saved file.
    """
    output_path = SPURS_DIR / f"{region_key}_spur_endpoints.json"

    data = [
        {
            "lat": ep.lat,
            "lon": ep.lon,
            "osm_way_id": ep.osm_way_id,
            "railway_type": ep.railway_type,
            "spur_length_m": ep.spur_length_m,
            "connected_to_mainline": ep.connected_to_mainline,
            "endpoint_method": ep.endpoint_method,
            "name": ep.name,
            "operator": ep.operator,
            "tags": ep.tags,
        }
        for ep in endpoints
    ]

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved {len(data)} endpoints to {output_path}")
    return output_path


def load_endpoints(region_key: str) -> list[SpurEndpoint]:
    """Load previously saved spur endpoints from JSON."""
    path = SPURS_DIR / f"{region_key}_spur_endpoints.json"

    if not path.exists():
        raise FileNotFoundError(f"No endpoints file found: {path}")

    with open(path) as f:
        data = json.load(f)

    return [
        SpurEndpoint(
            lat=item["lat"],
            lon=item["lon"],
            osm_way_id=item["osm_way_id"],
            railway_type=item["railway_type"],
            spur_length_m=item["spur_length_m"],
            connected_to_mainline=item["connected_to_mainline"],
            endpoint_method=item["endpoint_method"],
            name=item.get("name", ""),
            operator=item.get("operator", ""),
            tags=item.get("tags", {}),
        )
        for item in data
    ]
