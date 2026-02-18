"""Tests for company identifier module."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.company_identifier import (
    classify_confidence,
    guess_sector,
    parse_address_components,
)
from src.spur_finder import SpurEndpoint


def make_spur(**kwargs) -> SpurEndpoint:
    defaults = {
        "lat": 40.0, "lon": -89.0, "osm_way_id": 1,
        "railway_type": "spur", "spur_length_m": 100,
        "connected_to_mainline": True, "endpoint_method": "mainline_exclusion",
    }
    defaults.update(kwargs)
    return SpurEndpoint(**defaults)


class TestClassifyConfidence:
    def test_industrial_type_is_high(self):
        place = {"types": ["industrial", "establishment"], "name": "ABC Corp"}
        assert classify_confidence(place, make_spur()) == "HIGH"

    def test_industrial_name_is_high(self):
        place = {"types": ["establishment"], "name": "Springfield Grain Elevator"}
        assert classify_confidence(place, make_spur()) == "HIGH"

    def test_excluded_type_is_low(self):
        place = {"types": ["church", "place_of_worship"], "name": "First Baptist"}
        assert classify_confidence(place, make_spur()) == "LOW"

    def test_generic_establishment_is_medium(self):
        place = {"types": ["establishment", "point_of_interest"], "name": "Acme Inc"}
        assert classify_confidence(place, make_spur()) == "MEDIUM"

    def test_named_spur_match_is_high(self):
        place = {"types": ["establishment"], "name": "ABC Chemical"}
        spur = make_spur(name="ABC Chemical Spur")
        assert classify_confidence(place, spur) == "HIGH"


class TestGuessSector:
    def test_grain_is_agriculture(self):
        assert guess_sector({"name": "Cargill Grain Elevator"}) == "Agriculture"

    def test_steel_is_metals(self):
        assert guess_sector({"name": "US Steel Mill"}) == "Metals"

    def test_chemical_is_chemicals(self):
        assert guess_sector({"name": "Dow Chemical Plant"}) == "Chemicals"

    def test_unknown_returns_empty(self):
        assert guess_sector({"name": "XYZ Corporation"}) == ""

    def test_lumber_is_building_materials(self):
        assert guess_sector({"name": "84 Lumber Distribution"}) == "Building Materials"


class TestParseAddress:
    def test_standard_address(self):
        city, state, zip_code = parse_address_components(
            "123 Main St, Springfield, IL 62704, USA"
        )
        assert city == "Springfield"
        assert state == "IL"
        assert zip_code == "62704"

    def test_short_address(self):
        city, state, zip_code = parse_address_components("Springfield, IL 62704")
        # With only 2 parts, parsing is less reliable but shouldn't crash
        assert isinstance(city, str)
        assert isinstance(state, str)

    def test_empty_address(self):
        city, state, zip_code = parse_address_components("")
        assert city == ""
        assert state == ""
        assert zip_code == ""
