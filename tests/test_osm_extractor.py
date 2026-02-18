"""Tests for OSM extractor module."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.osm_extractor import (
    build_overpass_query,
    parse_osm_elements,
    ways_to_geojson,
)


def test_build_overpass_query_illinois():
    query = build_overpass_query("illinois")
    assert "railway" in query
    assert "spur" in query
    assert "siding" in query
    # Uses bbox approach: should contain Illinois bbox coordinates
    assert "36.97" in query  # south boundary
    assert "service" in query  # queries service=spur pattern too


def test_build_overpass_query_unknown_region():
    with pytest.raises(KeyError):
        build_overpass_query("narnia")


SAMPLE_OSM_DATA = {
    "elements": [
        {
            "type": "way",
            "id": 100,
            "tags": {"railway": "spur", "name": "Test Spur"},
            "nodes": [1, 2, 3],
        },
        {
            "type": "way",
            "id": 200,
            "tags": {"railway": "siding"},
            "nodes": [4, 5],
        },
        {"type": "node", "id": 1, "lat": 40.0, "lon": -89.0},
        {"type": "node", "id": 2, "lat": 40.001, "lon": -89.001},
        {"type": "node", "id": 3, "lat": 40.002, "lon": -89.002},
        {"type": "node", "id": 4, "lat": 41.0, "lon": -88.0},
        {"type": "node", "id": 5, "lat": 41.001, "lon": -88.001},
    ],
}


def test_parse_osm_elements():
    ways, nodes = parse_osm_elements(SAMPLE_OSM_DATA)

    assert len(ways) == 2
    assert len(nodes) == 5

    assert ways[100]["tags"]["railway"] == "spur"
    assert ways[200]["tags"]["railway"] == "siding"
    assert ways[100]["node_ids"] == [1, 2, 3]

    assert nodes[1]["lat"] == 40.0
    assert nodes[1]["lon"] == -89.0


def test_parse_empty_data():
    ways, nodes = parse_osm_elements({"elements": []})
    assert len(ways) == 0
    assert len(nodes) == 0


def test_ways_to_geojson():
    ways, nodes = parse_osm_elements(SAMPLE_OSM_DATA)
    geojson = ways_to_geojson(ways, nodes)

    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) == 2

    feature = geojson["features"][0]
    assert feature["type"] == "Feature"
    assert feature["geometry"]["type"] == "LineString"
    assert feature["properties"]["osm_way_id"] in (100, 200)


def test_geojson_coordinates_are_lon_lat():
    """GeoJSON standard is [longitude, latitude]."""
    ways, nodes = parse_osm_elements(SAMPLE_OSM_DATA)
    geojson = ways_to_geojson(ways, nodes)

    # Find the feature for way 100
    feat = next(f for f in geojson["features"] if f["properties"]["osm_way_id"] == 100)
    first_coord = feat["geometry"]["coordinates"][0]

    # lon, lat order — lon should be -89.0
    assert first_coord[0] == -89.0
    assert first_coord[1] == 40.0


def test_ways_with_missing_nodes():
    """Ways referencing nodes not in the data should be skipped."""
    data = {
        "elements": [
            {
                "type": "way",
                "id": 999,
                "tags": {"railway": "spur"},
                "nodes": [900, 901],  # nodes don't exist
            },
        ],
    }
    ways, nodes = parse_osm_elements(data)
    geojson = ways_to_geojson(ways, nodes)

    # Should produce 0 features since nodes are missing
    assert len(geojson["features"]) == 0
