"""Tests for spur finder module."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.spur_finder import (
    SpurEndpoint,
    deduplicate_nearby_endpoints,
    extract_spur_endpoints,
    find_junction_nodes,
)


def make_ways_and_nodes():
    """Create test data with known junction structure."""
    # Way 100: spur from junction node 1 to facility node 3
    # Way 200: siding sharing node 1 (junction) to facility node 5
    ways = {
        100: {
            "id": 100,
            "tags": {"railway": "spur", "name": "Test Spur A"},
            "node_ids": [1, 2, 3],
        },
        200: {
            "id": 200,
            "tags": {"railway": "siding"},
            "node_ids": [1, 4, 5],
        },
    }
    nodes = {
        1: {"id": 1, "lat": 40.0, "lon": -89.0},      # Junction (shared)
        2: {"id": 2, "lat": 40.001, "lon": -89.001},
        3: {"id": 3, "lat": 40.002, "lon": -89.002},    # Facility end of way 100
        4: {"id": 4, "lat": 40.003, "lon": -89.003},
        5: {"id": 5, "lat": 40.004, "lon": -89.004},    # Facility end of way 200
    }
    return ways, nodes


def test_find_junction_nodes():
    ways, _ = make_ways_and_nodes()
    junctions = find_junction_nodes(ways)

    # Node 1 is shared as first node of both ways
    assert 1 in junctions
    # Nodes 3 and 5 are only endpoints of one way each
    assert 3 not in junctions
    assert 5 not in junctions


def test_extract_spur_endpoints():
    ways, nodes = make_ways_and_nodes()
    endpoints = extract_spur_endpoints(ways, nodes)

    assert len(endpoints) == 2

    # Way 100: junction at node 1 (first), facility at node 3 (last)
    ep100 = next(e for e in endpoints if e.osm_way_id == 100)
    assert ep100.lat == 40.002  # Node 3
    assert ep100.lon == -89.002
    assert ep100.endpoint_method == "mainline_exclusion"
    assert ep100.railway_type == "spur"

    # Way 200: junction at node 1 (first), facility at node 5 (last)
    ep200 = next(e for e in endpoints if e.osm_way_id == 200)
    assert ep200.lat == 40.004  # Node 5
    assert ep200.lon == -89.004


def test_extract_spur_endpoints_no_junction():
    """When no junction is detected, should use midpoint."""
    ways = {
        300: {
            "id": 300,
            "tags": {"railway": "spur"},
            "node_ids": [10, 11, 12],
        },
    }
    nodes = {
        10: {"id": 10, "lat": 40.0, "lon": -89.0},
        11: {"id": 11, "lat": 40.001, "lon": -89.001},
        12: {"id": 12, "lat": 40.002, "lon": -89.002},
    }

    endpoints = extract_spur_endpoints(ways, nodes)
    assert len(endpoints) == 1
    assert endpoints[0].endpoint_method == "midpoint"


def test_deduplicate_nearby_endpoints():
    ep1 = SpurEndpoint(lat=40.0, lon=-89.0, osm_way_id=1, railway_type="spur",
                       spur_length_m=500, connected_to_mainline=True, endpoint_method="mainline_exclusion")
    ep2 = SpurEndpoint(lat=40.0001, lon=-89.0001, osm_way_id=2, railway_type="spur",
                       spur_length_m=300, connected_to_mainline=True, endpoint_method="mainline_exclusion")
    ep3 = SpurEndpoint(lat=41.0, lon=-88.0, osm_way_id=3, railway_type="siding",
                       spur_length_m=200, connected_to_mainline=True, endpoint_method="mainline_exclusion")

    # ep1 and ep2 are ~15m apart, should be deduped (keep ep1 — longer spur)
    result = deduplicate_nearby_endpoints([ep1, ep2, ep3], min_distance_m=50.0)

    assert len(result) == 2
    assert any(e.osm_way_id == 1 for e in result)  # Kept longer spur
    assert any(e.osm_way_id == 3 for e in result)


def test_spur_endpoint_coord():
    ep = SpurEndpoint(lat=40.5, lon=-89.5, osm_way_id=1, railway_type="spur",
                      spur_length_m=100, connected_to_mainline=True, endpoint_method="midpoint")
    assert ep.coord == (40.5, -89.5)
