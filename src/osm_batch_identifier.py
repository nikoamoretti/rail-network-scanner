"""
Phase 3 (Free, Fast): Batch identify companies at spur endpoints.

Single Overpass query fetches ALL named industrial/commercial features
in the region bbox, then matches them to spur endpoints locally.

Much faster than per-endpoint queries. Cost: $0.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import requests

from config import OVERPASS_TIMEOUT, OVERPASS_URLS, REGIONS
from src.spur_finder import SpurEndpoint
from src.utils.cache import FileCache
from src.utils.geo_utils import haversine_distance
from src.utils.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

poi_cache = FileCache("batch_poi")
nominatim_cache = FileCache("nominatim")
nominatim_limiter = RateLimiter(0.9, name="Nominatim")

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
    google_place_id: str
    osm_way_id: int
    spur_length_m: float
    connected_to_mainline: bool
    confidence: str
    discovery_date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    in_existing_list: bool = False
    sector_guess: str = ""
    source: str = "osm_batch"
    place_types: list = field(default_factory=list)


# --- Sector classification ---

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
    "energy", "power", "electric", "utility", "silo", "tank",
}

SECTOR_MAP = {
    "Agriculture": ["grain", "elevator", "feed", "seed", "farm", "coop", "co-op", "cooperative", "ethanol", "fertilizer", "agri", "silo"],
    "Chemicals": ["chemical", "chem", "polymer", "plastic", "resin"],
    "Energy": ["energy", "oil", "gas", "petroleum", "fuel", "propane", "pipeline", "power", "electric", "utility"],
    "Metals": ["steel", "metal", "iron", "aluminum", "foundry", "forge", "scrap"],
    "Building Materials": ["lumber", "concrete", "cement", "aggregate", "sand", "gravel", "stone", "brick"],
    "Forest Products": ["paper", "pulp", "wood", "lumber", "timber"],
    "Food & Beverage": ["food", "beverage", "brewery", "sugar", "flour", "meat", "dairy"],
    "Mining": ["mine", "mining", "mineral", "coal", "quarry"],
    "Logistics": ["logistics", "warehouse", "distribution", "freight", "transport", "terminal", "intermodal", "depot", "storage"],
}


def _guess_sector(name: str) -> str:
    name_lower = name.lower()
    for sector, keywords in SECTOR_MAP.items():
        if any(kw in name_lower for kw in keywords):
            return sector
    return ""


# --- Batch POI Fetch ---

def _build_batch_poi_query(region_key: str) -> str:
    """Build a single Overpass query for ALL industrial POIs in a region."""
    bbox = REGIONS[region_key]["bbox"]
    b = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"

    return f"""[out:json][timeout:{OVERPASS_TIMEOUT}];
(
  way["building"~"(industrial|warehouse|commercial)"]["name"]({b});
  way["landuse"="industrial"]["name"]({b});
  node["man_made"~"(works|silo|storage_tank|kiln)"]["name"]({b});
  way["man_made"~"(works|silo|storage_tank|kiln)"]["name"]({b});
  node["industrial"]["name"]({b});
  way["building"]["name"]["operator"]({b});
  node["amenity"="fuel"]["name"]["brand"]({b});
  way["craft"]["name"]({b});
  node["office"~"(company|industrial)"]["name"]({b});
  way["amenity"~"(fuel|loading_dock|warehouse)"]["name"]({b});
  node["shop"~"(trade|wholesale|hardware)"]["name"]({b});
  way["shop"~"(trade|wholesale|hardware)"]["name"]({b});
  node["company"]["name"]({b});
  way["company"]["name"]({b});
  way["building"]["operator"]({b});
  node["man_made"~"(petroleum_well|wastewater_plant|water_works|pumping_station)"]["name"]({b});
  way["man_made"~"(petroleum_well|wastewater_plant|water_works|pumping_station)"]["name"]({b});
  way["power"~"(plant|substation|generator)"]["name"]({b});
  node["power"~"(plant|substation|generator)"]["name"]({b});
  way["landuse"~"(quarry|railway|commercial)"]["name"]({b});
  node["amenity"="recycling"]["name"]({b});
  way["amenity"="recycling"]["name"]({b});
);
out center;"""


def fetch_all_pois(region_key: str) -> List[dict]:
    """
    Fetch all named industrial/commercial POIs in a region via one Overpass query.

    Returns:
        List of OSM elements with lat/lon and tags.
    """
    cache_key = {"region": region_key, "type": "batch_poi_v3"}
    cached = poi_cache.get(cache_key)
    if cached is not None:
        logger.info(f"Using cached batch POI data ({len(cached)} features)")
        return cached

    query = _build_batch_poi_query(region_key)

    for url in OVERPASS_URLS:
        try:
            logger.info(f"Fetching all industrial POIs for {region_key} from {url}...")
            resp = requests.post(url, data={"data": query}, timeout=OVERPASS_TIMEOUT + 30)
            resp.raise_for_status()
            data = resp.json()
            elements = data.get("elements", [])

            if elements:
                # Normalize coordinates (ways have center, nodes have lat/lon)
                for el in elements:
                    if "center" in el:
                        el["lat"] = el["center"]["lat"]
                        el["lon"] = el["center"]["lon"]

                poi_cache.set(cache_key, elements)
                logger.info(f"Fetched {len(elements)} industrial POIs")
                return elements

        except Exception as e:
            logger.warning(f"Batch POI query failed at {url}: {e}")
            continue

    logger.error("All Overpass servers failed for batch POI query")
    return []


# --- Spatial Matching ---

def match_pois_to_endpoints(
    pois: List[dict],
    endpoints: List[SpurEndpoint],
    max_distance_m: float = 500.0,
) -> Dict[int, List[Tuple[dict, float]]]:
    """
    Match POIs to nearby spur endpoints using brute-force distance calculation.

    Returns:
        Dict mapping osm_way_id to list of (poi, distance_m) tuples.
    """
    logger.info(f"Matching {len(pois)} POIs to {len(endpoints)} spur endpoints (within {max_distance_m}m)...")
    start = time.time()

    matches: Dict[int, List[Tuple[dict, float]]] = defaultdict(list)

    for poi in pois:
        poi_lat = poi.get("lat", 0)
        poi_lon = poi.get("lon", 0)
        if not poi_lat or not poi_lon:
            continue

        poi_coord = (poi_lat, poi_lon)

        for ep in endpoints:
            dist = haversine_distance(poi_coord, ep.coord)
            if dist <= max_distance_m:
                matches[ep.osm_way_id].append((poi, dist))

    elapsed = time.time() - start
    matched_count = len(matches)
    logger.info(f"Matched POIs to {matched_count}/{len(endpoints)} endpoints in {elapsed:.1f}s")

    return matches


# --- Nominatim for unmatched ---

def reverse_geocode_nominatim(lat: float, lon: float) -> Optional[dict]:
    """Reverse geocode via Nominatim (cached, rate-limited)."""
    from src.utils.cache import make_location_key
    cache_key = {**make_location_key(lat, lon), "type": "nominatim"}
    cached = nominatim_cache.get(cache_key)
    if cached is not None:
        return cached

    nominatim_limiter.wait()

    try:
        resp = requests.get(
            NOMINATIM_URL,
            params={"lat": lat, "lon": lon, "format": "json", "zoom": 18, "addressdetails": 1},
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


# --- Main Pipeline ---

def identify_companies_at_endpoints(
    endpoints: List[SpurEndpoint],
    region_key: str = "illinois",
    max_distance_m: float = 500.0,
    use_nominatim: bool = False,
) -> List[IdentifiedCompany]:
    """
    Identify companies at spur endpoints using batch OSM data.

    1. Fetch ALL industrial POIs in region (one Overpass query)
    2. Match POIs to spur endpoints by distance
    3. Optionally use Nominatim for unmatched endpoints

    Args:
        endpoints: Spur endpoints to identify.
        region_key: Region for the batch query.
        max_distance_m: Max distance for POI-to-endpoint matching.
        use_nominatim: Whether to reverse geocode unmatched endpoints.

    Returns:
        List of IdentifiedCompany objects.
    """
    companies: List[IdentifiedCompany] = []

    # Step 1: Fetch all POIs in one shot
    pois = fetch_all_pois(region_key)
    if not pois:
        logger.warning("No POIs fetched, falling back to Nominatim only")

    # Step 2: Match to endpoints
    matches = match_pois_to_endpoints(pois, endpoints, max_distance_m)

    matched_ways: Set[int] = set()

    for ep in endpoints:
        poi_matches = matches.get(ep.osm_way_id, [])
        if not poi_matches:
            continue

        # Sort by distance, pick closest named feature
        poi_matches.sort(key=lambda x: x[1])

        best_poi, best_dist = poi_matches[0]
        tags = best_poi.get("tags", {})
        name = tags.get("name", "") or tags.get("operator", "") or tags.get("brand", "")
        if not name:
            continue

        # Classify confidence
        name_lower = name.lower()
        has_industrial = any(kw in name_lower for kw in INDUSTRIAL_KEYWORDS)
        building = tags.get("building", "")
        man_made = tags.get("man_made", "")
        landuse = tags.get("landuse", "")

        if has_industrial or man_made or building in ("industrial", "warehouse"):
            confidence = "HIGH"
        elif building or landuse:
            confidence = "MEDIUM"
        else:
            confidence = "MEDIUM"

        facility_type = man_made or building or landuse or "poi"

        companies.append(IdentifiedCompany(
            company_name=name,
            address=tags.get("addr:street", ""),
            city=tags.get("addr:city", ""),
            state=tags.get("addr:state", ""),
            zip_code=tags.get("addr:postcode", ""),
            lat=ep.lat,
            lon=ep.lon,
            facility_type=facility_type,
            google_place_id="",
            osm_way_id=ep.osm_way_id,
            spur_length_m=ep.spur_length_m,
            connected_to_mainline=ep.connected_to_mainline,
            confidence=confidence,
            sector_guess=_guess_sector(name),
            source="osm_batch",
        ))
        matched_ways.add(ep.osm_way_id)

    logger.info(f"Batch matching: {len(companies)} companies from {len(matched_ways)} endpoints")

    # Step 3: Nominatim for unmatched
    if use_nominatim:
        remaining = [ep for ep in endpoints if ep.osm_way_id not in matched_ways]
        logger.info(f"Nominatim pass for {len(remaining)} unmatched endpoints")

        nom_found = 0
        for i, ep in enumerate(remaining):
            if i % 200 == 0 and i > 0:
                logger.info(f"Nominatim: {i}/{len(remaining)} ({nom_found} found)")

            result = reverse_geocode_nominatim(ep.lat, ep.lon)
            if not result:
                continue

            name = result.get("name", "")
            addr = result.get("address", {})
            category = result.get("category", "")

            if not name or category in ("highway", "boundary", "place"):
                continue

            has_industrial = any(kw in name.lower() for kw in INDUSTRIAL_KEYWORDS)

            companies.append(IdentifiedCompany(
                company_name=name,
                address=addr.get("road", ""),
                city=addr.get("city", addr.get("town", addr.get("village", ""))),
                state=addr.get("state", ""),
                zip_code=addr.get("postcode", ""),
                lat=ep.lat,
                lon=ep.lon,
                facility_type=f"{category}/{result.get('type', '')}",
                google_place_id="",
                osm_way_id=ep.osm_way_id,
                spur_length_m=ep.spur_length_m,
                connected_to_mainline=ep.connected_to_mainline,
                confidence="HIGH" if has_industrial else "MEDIUM",
                sector_guess=_guess_sector(name),
                source="nominatim",
            ))
            nom_found += 1

        logger.info(f"Nominatim: found {nom_found} additional facilities")

    logger.info(f"Total identified: {len(companies)} companies")
    return companies
