"""
Phase 3 (Free): Identify companies at spur endpoints using OSM + Nominatim.

Two-pass approach:
1. Batch Overpass query for named industrial features near all endpoints
2. Nominatim reverse geocode for address data on remaining endpoints

No API key needed. Cost: $0.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import requests

from config import OVERPASS_URLS, OVERPASS_TIMEOUT
from src.spur_finder import SpurEndpoint
from src.utils.cache import FileCache, make_location_key
from src.utils.geo_utils import haversine_distance
from src.utils.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

nominatim_cache = FileCache("nominatim")
nominatim_limiter = RateLimiter(0.9, name="Nominatim")  # <1 req/sec per policy
osm_poi_cache = FileCache("osm_poi")

NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
USER_AGENT = "RailNetworkScanner/1.0 (github.com/nikoamoretti)"


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
    google_place_id: str  # empty for OSM-based identification
    osm_way_id: int
    spur_length_m: float
    connected_to_mainline: bool
    confidence: str  # "HIGH", "MEDIUM", "LOW"
    discovery_date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    in_existing_list: bool = False
    sector_guess: str = ""
    source: str = "osm"  # "osm", "nominatim", "google_places"
    place_types: list = field(default_factory=list)


# --- Overpass Batch POI Queries ---

def _chunk_endpoints(endpoints: List[SpurEndpoint], chunk_size: int = 200) -> List[List[SpurEndpoint]]:
    """Split endpoints into chunks for batch queries."""
    return [endpoints[i:i + chunk_size] for i in range(0, len(endpoints), chunk_size)]


def _build_poi_query(endpoints: List[SpurEndpoint], radius_m: int = 200) -> str:
    """
    Build Overpass query to find named features near multiple endpoints.

    Queries for industrial buildings, named facilities, and commercial POIs
    within radius_m of each endpoint coordinate.
    """
    around_clauses = []
    for ep in endpoints:
        around_clauses.append(f"{ep.lat},{ep.lon}")

    # Build a union of around queries for each endpoint
    # We use a single query with multiple (around:...) selectors
    # Overpass supports "around" with multiple center points
    coord_list = ",".join(around_clauses)

    return f"""[out:json][timeout:{OVERPASS_TIMEOUT}];
(
  // Named nodes near spur endpoints
  node(around:{radius_m},{coord_list})["name"]["name"!~"^(Street|Avenue|Road|Drive|Lane|Boulevard|Place|Court|Circle|Way|Trail|Highway|Route)"]["highway"!~"."];
  // Named ways (buildings, landuse) near spur endpoints
  way(around:{radius_m},{coord_list})["name"]["building"~"(industrial|commercial|warehouse|retail|office|yes)"];
  way(around:{radius_m},{coord_list})["name"]["landuse"="industrial"];
  way(around:{radius_m},{coord_list})["name"]["man_made"~"(works|silo|storage_tank|kiln)"];
  // Specific industrial/commercial nodes
  node(around:{radius_m},{coord_list})["shop"];
  node(around:{radius_m},{coord_list})["amenity"~"(fuel|marketplace)"];
  node(around:{radius_m},{coord_list})["industrial"];
  node(around:{radius_m},{coord_list})["office"];
);
out body;
>;
out skel qt;"""


def _build_simple_poi_query(endpoint: SpurEndpoint, radius_m: int = 200) -> str:
    """Build a single-endpoint POI query (used when batch is too large)."""
    return f"""[out:json][timeout:30];
(
  node(around:{radius_m},{endpoint.lat},{endpoint.lon})["name"]["highway"!~"."];
  way(around:{radius_m},{endpoint.lat},{endpoint.lon})["name"]["building"];
  way(around:{radius_m},{endpoint.lat},{endpoint.lon})["name"]["landuse"="industrial"];
  way(around:{radius_m},{endpoint.lat},{endpoint.lon})["name"]["man_made"];
);
out body;"""


def query_overpass_pois(query: str) -> list:
    """Execute an Overpass query and return elements."""
    cache_key = {"poi_query": query[:200]}  # truncated for cache key
    cached = osm_poi_cache.get(cache_key)
    if cached is not None:
        return cached

    for url in OVERPASS_URLS:
        try:
            resp = requests.post(url, data={"data": query}, timeout=OVERPASS_TIMEOUT + 30)
            resp.raise_for_status()
            data = resp.json()
            elements = data.get("elements", [])
            osm_poi_cache.set(cache_key, elements)
            return elements
        except Exception as e:
            logger.warning(f"Overpass POI query failed at {url}: {e}")
            continue

    return []


def batch_find_osm_facilities(
    endpoints: List[SpurEndpoint],
    radius_m: int = 200,
) -> Dict[int, List[dict]]:
    """
    Find OSM facilities near all spur endpoints using batch Overpass queries.

    Returns:
        Dict mapping osm_way_id to list of nearby named features.
    """
    logger.info(f"Batch querying OSM for facilities near {len(endpoints)} endpoints")

    # For each endpoint, we'll do individual queries since batch around
    # with many points can be unreliable
    results: Dict[int, List[dict]] = defaultdict(list)
    found_count = 0

    chunks = _chunk_endpoints(endpoints, chunk_size=50)
    for chunk_idx, chunk in enumerate(chunks):
        if chunk_idx % 5 == 0:
            logger.info(f"Processing chunk {chunk_idx + 1}/{len(chunks)} ({chunk_idx * 50}/{len(endpoints)} endpoints)")

        for ep in chunk:
            cache_key = make_location_key(ep.lat, ep.lon)
            cache_key["type"] = "osm_poi"
            cached = osm_poi_cache.get(cache_key)

            if cached is not None:
                if cached:
                    results[ep.osm_way_id] = cached
                    found_count += 1
                continue

            query = _build_simple_poi_query(ep, radius_m)
            time.sleep(1.0)  # Be nice to Overpass

            try:
                for url in OVERPASS_URLS:
                    try:
                        resp = requests.post(url, data={"data": query}, timeout=30)
                        resp.raise_for_status()
                        data = resp.json()
                        elements = data.get("elements", [])
                        break
                    except Exception:
                        continue
                else:
                    elements = []
            except Exception as e:
                logger.debug(f"POI query failed for ({ep.lat:.4f},{ep.lon:.4f}): {e}")
                elements = []

            # Filter to named features only
            named = [
                el for el in elements
                if el.get("tags", {}).get("name")
                and el.get("tags", {}).get("highway") is None  # exclude streets
            ]

            osm_poi_cache.set(cache_key, named)

            if named:
                results[ep.osm_way_id] = named
                found_count += 1

    logger.info(f"Found OSM facilities for {found_count}/{len(endpoints)} endpoints")
    return results


# --- Nominatim Reverse Geocoding ---

def reverse_geocode_nominatim(lat: float, lon: float) -> Optional[dict]:
    """
    Reverse geocode a coordinate using Nominatim.

    Returns:
        Nominatim response dict or None.
    """
    cache_key = {**make_location_key(lat, lon), "type": "nominatim"}
    cached = nominatim_cache.get(cache_key)
    if cached is not None:
        return cached

    nominatim_limiter.wait()

    try:
        resp = requests.get(
            NOMINATIM_URL,
            params={
                "lat": lat,
                "lon": lon,
                "format": "json",
                "zoom": 18,
                "addressdetails": 1,
                "namedetails": 1,
                "extratags": 1,
            },
            headers={"User-Agent": USER_AGENT},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        nominatim_cache.set(cache_key, data)
        return data
    except Exception as e:
        logger.debug(f"Nominatim failed for ({lat:.4f},{lon:.4f}): {e}")
        return None


# --- Classification ---

INDUSTRIAL_KEYWORDS = {
    "grain", "elevator", "terminal", "warehouse", "storage", "depot",
    "mill", "plant", "factory", "refinery", "chemical", "lumber",
    "concrete", "cement", "aggregate", "sand", "gravel", "stone",
    "coal", "mine", "quarry", "steel", "metal", "iron", "scrap",
    "oil", "gas", "petroleum", "fuel", "propane", "ethanol",
    "feed", "seed", "fertilizer", "coop", "co-op", "cooperative",
    "food", "beverage", "brewery", "sugar", "flour", "dairy",
    "paper", "pulp", "wood", "timber", "forest", "products",
    "auto", "truck", "vehicle", "railroad", "railway", "rail",
    "logistics", "distribution", "freight", "transport", "intermodal",
    "industrial", "manufacturing", "processing", "materials",
    "energy", "power", "electric", "utility",
}

SECTOR_MAP = {
    "Agriculture": ["grain", "elevator", "feed", "seed", "farm", "coop", "co-op", "cooperative", "ethanol", "fertilizer", "agri"],
    "Chemicals": ["chemical", "chem", "polymer", "plastic", "resin"],
    "Energy": ["energy", "oil", "gas", "petroleum", "fuel", "propane", "pipeline", "power", "electric"],
    "Metals": ["steel", "metal", "iron", "aluminum", "foundry", "forge", "scrap"],
    "Building Materials": ["lumber", "concrete", "cement", "aggregate", "sand", "gravel", "stone", "brick"],
    "Forest Products": ["paper", "pulp", "wood", "lumber", "timber"],
    "Food & Beverage": ["food", "beverage", "brewery", "sugar", "flour", "meat", "dairy"],
    "Mining": ["mine", "mining", "mineral", "coal", "quarry"],
    "Logistics": ["logistics", "warehouse", "distribution", "freight", "transport", "terminal", "intermodal", "depot", "storage"],
}


def _guess_sector(name: str) -> str:
    """Guess industry sector from facility name."""
    name_lower = name.lower()
    for sector, keywords in SECTOR_MAP.items():
        if any(kw in name_lower for kw in keywords):
            return sector
    return ""


def _classify_osm_feature(feature: dict, spur: SpurEndpoint) -> Tuple[str, str]:
    """
    Classify an OSM feature and return (confidence, facility_type).
    """
    tags = feature.get("tags", {})
    name = tags.get("name", "").lower()
    building = tags.get("building", "")
    landuse = tags.get("landuse", "")
    man_made = tags.get("man_made", "")

    has_industrial_keyword = any(kw in name for kw in INDUSTRIAL_KEYWORDS)
    is_industrial_building = building in ("industrial", "warehouse", "commercial")
    is_industrial_land = landuse == "industrial"
    is_infrastructure = man_made in ("works", "silo", "storage_tank", "kiln")

    if has_industrial_keyword or is_infrastructure:
        return "HIGH", man_made or building or landuse or "industrial"
    if is_industrial_building or is_industrial_land:
        return "HIGH", building or landuse
    if building:
        return "MEDIUM", building
    return "MEDIUM", "poi"


# --- Main Identification Pipeline ---

def identify_companies_at_endpoints(
    endpoints: List[SpurEndpoint],
    radius_m: int = 200,
    use_nominatim: bool = True,
    nominatim_limit: int = 0,
) -> List[IdentifiedCompany]:
    """
    Identify companies at spur endpoints using free OSM data.

    Two-pass approach:
    1. Overpass batch query for named industrial features
    2. Nominatim reverse geocode for remaining endpoints

    Args:
        endpoints: Spur endpoints to identify.
        radius_m: Search radius in meters.
        use_nominatim: Whether to use Nominatim for unidentified endpoints.
        nominatim_limit: Max Nominatim queries (0 = all remaining).

    Returns:
        List of IdentifiedCompany objects.
    """
    companies: List[IdentifiedCompany] = []
    identified_ways: Set[int] = set()

    # --- Pass 1: OSM POI batch query ---
    logger.info("Pass 1: Querying OSM for named facilities near endpoints")
    osm_results = batch_find_osm_facilities(endpoints, radius_m)

    ep_by_way = {ep.osm_way_id: ep for ep in endpoints}

    for way_id, features in osm_results.items():
        spur = ep_by_way.get(way_id)
        if not spur:
            continue

        # Pick best feature (prefer named industrial features)
        best_feature = None
        best_confidence = "LOW"
        for feat in features:
            conf, ftype = _classify_osm_feature(feat, spur)
            if conf == "HIGH" or (conf == "MEDIUM" and best_confidence != "HIGH"):
                best_feature = feat
                best_confidence = conf

        if best_feature is None and features:
            best_feature = features[0]
            best_confidence = "MEDIUM"

        if best_feature:
            tags = best_feature.get("tags", {})
            name = tags.get("name", "Unknown")
            conf, ftype = _classify_osm_feature(best_feature, spur)

            companies.append(IdentifiedCompany(
                company_name=name,
                address=tags.get("addr:street", ""),
                city=tags.get("addr:city", ""),
                state=tags.get("addr:state", ""),
                zip_code=tags.get("addr:postcode", ""),
                lat=spur.lat,
                lon=spur.lon,
                facility_type=ftype,
                google_place_id="",
                osm_way_id=way_id,
                spur_length_m=spur.spur_length_m,
                connected_to_mainline=spur.connected_to_mainline,
                confidence=conf,
                sector_guess=_guess_sector(name),
                source="osm",
            ))
            identified_ways.add(way_id)

    logger.info(f"Pass 1 result: {len(companies)} companies from OSM POI data")

    # --- Pass 2: Nominatim for remaining endpoints ---
    if use_nominatim:
        remaining = [ep for ep in endpoints if ep.osm_way_id not in identified_ways]
        if nominatim_limit > 0:
            remaining = remaining[:nominatim_limit]

        logger.info(f"Pass 2: Nominatim reverse geocoding for {len(remaining)} remaining endpoints")

        nom_found = 0
        for i, ep in enumerate(remaining):
            if i % 100 == 0 and i > 0:
                logger.info(f"Nominatim progress: {i}/{len(remaining)}")

            result = reverse_geocode_nominatim(ep.lat, ep.lon)
            if not result:
                continue

            name = result.get("name", "")
            category = result.get("category", "")
            rtype = result.get("type", "")
            addr = result.get("address", {})
            extratags = result.get("extratags", {})

            # Skip if it's just a road/address with no facility name
            if not name or category in ("highway", "boundary", "place"):
                # Still record the address for enrichment
                road = addr.get("road", "")
                city = addr.get("city", addr.get("town", addr.get("village", "")))
                if road:
                    companies.append(IdentifiedCompany(
                        company_name=f"Unknown facility near {road}",
                        address=road,
                        city=city,
                        state=addr.get("state", ""),
                        zip_code=addr.get("postcode", ""),
                        lat=ep.lat,
                        lon=ep.lon,
                        facility_type=rtype,
                        google_place_id="",
                        osm_way_id=ep.osm_way_id,
                        spur_length_m=ep.spur_length_m,
                        connected_to_mainline=ep.connected_to_mainline,
                        confidence="LOW",
                        sector_guess="",
                        source="nominatim_address",
                    ))
                continue

            # Named facility found
            has_industrial = any(kw in name.lower() for kw in INDUSTRIAL_KEYWORDS)
            confidence = "HIGH" if has_industrial else "MEDIUM"

            companies.append(IdentifiedCompany(
                company_name=name,
                address=addr.get("road", ""),
                city=addr.get("city", addr.get("town", addr.get("village", ""))),
                state=addr.get("state", ""),
                zip_code=addr.get("postcode", ""),
                lat=ep.lat,
                lon=ep.lon,
                facility_type=f"{category}/{rtype}",
                google_place_id="",
                osm_way_id=ep.osm_way_id,
                spur_length_m=ep.spur_length_m,
                connected_to_mainline=ep.connected_to_mainline,
                confidence=confidence,
                sector_guess=_guess_sector(name),
                source="nominatim",
            ))
            nom_found += 1

        logger.info(f"Pass 2 result: {nom_found} named facilities from Nominatim")

    logger.info(f"Total identified: {len(companies)} companies")
    return companies
