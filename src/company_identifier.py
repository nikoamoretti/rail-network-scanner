"""
Phase 3: Identify companies at spur endpoint locations.

Uses Google Places API nearbysearch to find businesses/facilities
at each spur endpoint coordinate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Set, Tuple

import requests

from config import (
    GOOGLE_MAPS_API_KEY,
    PLACES_NEARBY_URL,
    PLACES_RATE_LIMIT,
    PLACES_SEARCH_RADIUS_M,
)
from src.spur_finder import SpurEndpoint
from src.utils.cache import FileCache, make_location_key
from src.utils.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

places_cache = FileCache("places")
places_limiter = RateLimiter(PLACES_RATE_LIMIT, name="GooglePlaces")


@dataclass
class IdentifiedCompany:
    """A company identified at a spur endpoint."""
    company_name: str
    address: str
    city: str
    state: str
    zip_code: str
    lat: float
    lon: float
    facility_type: str
    google_place_id: str
    osm_way_id: int
    spur_length_m: float
    connected_to_mainline: bool
    confidence: str  # "HIGH", "MEDIUM", "LOW"
    discovery_date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    in_existing_list: bool = False
    sector_guess: str = ""
    place_types: list[str] = field(default_factory=list)

    # Extra metadata not in CSV
    rating: float = 0.0
    user_ratings_total: int = 0
    business_status: str = ""


def search_nearby_places(
    lat: float, lon: float, radius_m: int = PLACES_SEARCH_RADIUS_M
) -> list[dict]:
    """
    Search for businesses near a coordinate using Google Places API.

    Args:
        lat: Latitude.
        lon: Longitude.
        radius_m: Search radius in meters.

    Returns:
        List of place results from Google Places API.

    Raises:
        ValueError: If GOOGLE_MAPS_API_KEY is not set.
        requests.HTTPError: On API error.
    """
    if not GOOGLE_MAPS_API_KEY:
        raise ValueError(
            "GOOGLE_MAPS_API_KEY not set. Add it to .env file."
        )

    # Check cache (rounded coordinates)
    cache_key = {**make_location_key(lat, lon), "radius": radius_m}
    cached = places_cache.get(cache_key)
    if cached is not None:
        logger.debug(f"Cache hit for ({lat:.4f}, {lon:.4f})")
        return cached

    places_limiter.wait()

    params = {
        "location": f"{lat},{lon}",
        "radius": radius_m,
        "key": GOOGLE_MAPS_API_KEY,
    }

    response = requests.get(PLACES_NEARBY_URL, params=params, timeout=15)
    response.raise_for_status()

    data = response.json()
    status = data.get("status")

    if status == "ZERO_RESULTS":
        logger.debug(f"No places found near ({lat:.4f}, {lon:.4f})")
        places_cache.set(cache_key, [])
        return []
    elif status != "OK":
        logger.warning(f"Places API status: {status} for ({lat:.4f}, {lon:.4f})")
        if status == "REQUEST_DENIED":
            raise ValueError(f"Places API denied: {data.get('error_message', 'check API key')}")
        return []

    results = data.get("results", [])
    logger.debug(f"Found {len(results)} places near ({lat:.4f}, {lon:.4f})")

    places_cache.set(cache_key, results)
    return results


# Types that suggest industrial / rail-served facilities
INDUSTRIAL_TYPE_KEYWORDS = {
    "warehouse", "storage", "industrial", "manufacturing",
    "factory", "plant", "terminal", "depot", "yard",
    "elevator", "mill", "refinery", "chemical",
}

# Types to exclude (residential, non-industrial)
EXCLUDE_TYPES = {
    "church", "school", "hospital", "doctor", "dentist",
    "park", "cemetery", "library", "museum", "restaurant",
    "cafe", "bar", "lodging", "hotel", "real_estate_agency",
    "beauty_salon", "hair_care", "spa", "gym", "laundry",
    "atm", "bank", "insurance_agency", "lawyer", "pharmacy",
    "veterinary_care", "pet_store", "florist", "clothing_store",
}


def classify_confidence(place: dict, spur: SpurEndpoint) -> str:
    """
    Assign a confidence level to a place match.

    HIGH: Place is close to spur, has industrial type or name.
    MEDIUM: Place found but type is ambiguous.
    LOW: Place is generic / probably not rail-served.
    """
    types = set(place.get("types", []))
    name = place.get("name", "").lower()

    # Check for industrial type indicators
    has_industrial_type = bool(types & {"industrial", "storage", "warehouse"})
    has_industrial_name = any(kw in name for kw in INDUSTRIAL_TYPE_KEYWORDS)
    is_excluded_type = bool(types & EXCLUDE_TYPES)

    if is_excluded_type and not has_industrial_name:
        return "LOW"

    if has_industrial_type or has_industrial_name:
        return "HIGH"

    # If the spur is named and the name matches, high confidence
    if spur.name and spur.name.lower() in name:
        return "HIGH"

    # Business with no clear category
    if "establishment" in types or "point_of_interest" in types:
        return "MEDIUM"

    return "MEDIUM"


def guess_sector(place: dict) -> str:
    """Guess the industrial sector from place types and name."""
    name = place.get("name", "").lower()
    types = set(place.get("types", []))

    sector_keywords = {
        "Agriculture": ["grain", "elevator", "feed", "seed", "farm", "coop", "ethanol", "fertilizer", "agri"],
        "Chemicals": ["chemical", "chem", "polymer", "plastic", "resin"],
        "Energy": ["energy", "oil", "gas", "petroleum", "fuel", "propane", "pipeline"],
        "Metals": ["steel", "metal", "iron", "aluminum", "foundry", "forge", "scrap"],
        "Building Materials": ["lumber", "concrete", "cement", "aggregate", "sand", "gravel", "stone", "brick"],
        "Forest Products": ["paper", "pulp", "wood", "lumber", "timber"],
        "Food & Beverage": ["food", "beverage", "brewery", "sugar", "flour", "meat", "dairy"],
        "Mining": ["mine", "mining", "mineral", "coal", "quarry"],
        "Automotive": ["auto", "automotive", "vehicle", "car", "truck"],
        "Logistics": ["logistics", "warehouse", "distribution", "freight", "transport", "terminal", "intermodal"],
    }

    for sector, keywords in sector_keywords.items():
        if any(kw in name for kw in keywords):
            return sector

    return ""


def parse_address_components(address: str) -> tuple[str, str, str]:
    """
    Extract city, state, zip from a formatted address string.

    Google Places returns addresses like: "123 Main St, Springfield, IL 62704, USA"
    """
    parts = [p.strip() for p in address.split(",")]

    city = ""
    state = ""
    zip_code = ""

    if len(parts) >= 3:
        city = parts[-3] if len(parts) >= 3 else ""
        # State + zip are usually in the second-to-last part
        state_zip = parts[-2].strip() if len(parts) >= 2 else ""
        state_zip_parts = state_zip.split()
        if state_zip_parts:
            state = state_zip_parts[0]
            zip_code = state_zip_parts[1] if len(state_zip_parts) > 1 else ""

    return city, state, zip_code


def identify_companies_at_endpoints(
    endpoints: list[SpurEndpoint],
    radius_m: int = PLACES_SEARCH_RADIUS_M,
    max_results_per_endpoint: int = 3,
) -> list[IdentifiedCompany]:
    """
    Identify companies at each spur endpoint using Google Places API.

    Args:
        endpoints: List of SpurEndpoint objects.
        radius_m: Search radius in meters.
        max_results_per_endpoint: Max places to keep per endpoint.

    Returns:
        List of IdentifiedCompany objects.
    """
    companies: list[IdentifiedCompany] = []
    no_results_count = 0

    for i, spur in enumerate(endpoints, 1):
        if i % 50 == 0:
            logger.info(f"Processing endpoint {i}/{len(endpoints)}...")

        try:
            places = search_nearby_places(spur.lat, spur.lon, radius_m)
        except (ValueError, requests.RequestException) as e:
            logger.error(f"Error searching near ({spur.lat:.4f}, {spur.lon:.4f}): {e}")
            continue

        if not places:
            no_results_count += 1
            continue

        # Filter and rank places
        ranked = []
        for place in places:
            confidence = classify_confidence(place, spur)
            if confidence != "LOW" or len(places) == 1:
                ranked.append((place, confidence))

        # Sort by confidence (HIGH first)
        confidence_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        ranked.sort(key=lambda x: confidence_order.get(x[1], 3))

        for place, confidence in ranked[:max_results_per_endpoint]:
            address = place.get("vicinity", place.get("formatted_address", ""))
            city, state, zip_code = parse_address_components(address)

            companies.append(IdentifiedCompany(
                company_name=place.get("name", "Unknown"),
                address=address,
                city=city,
                state=state,
                zip_code=zip_code,
                lat=place["geometry"]["location"]["lat"],
                lon=place["geometry"]["location"]["lng"],
                facility_type=", ".join(place.get("types", [])[:3]),
                google_place_id=place.get("place_id", ""),
                osm_way_id=spur.osm_way_id,
                spur_length_m=spur.spur_length_m,
                connected_to_mainline=spur.connected_to_mainline,
                confidence=confidence,
                sector_guess=guess_sector(place),
                place_types=place.get("types", []),
                rating=place.get("rating", 0.0),
                user_ratings_total=place.get("user_ratings_total", 0),
                business_status=place.get("business_status", ""),
            ))

    logger.info(
        f"Identified {len(companies)} companies from {len(endpoints)} endpoints "
        f"({no_results_count} had no results)"
    )
    return companies
