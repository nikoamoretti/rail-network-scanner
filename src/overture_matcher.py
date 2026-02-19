"""
Phase 3b (Free, No-API): Match Overture Maps place POIs to unmatched spur endpoints.

Uses DuckDB with httpfs to query Overture Maps GeoParquet files directly from S3
without downloading the full dataset. Filters to industrial/commercial categories
likely to be rail-served, then matches by haversine distance to spur endpoints.

Cost: $0. Requires: duckdb>=0.9.0 with httpfs extension.
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from config import CACHE_DIR, REGIONS
from src.osm_batch_identifier import (
    INDUSTRIAL_KEYWORDS,
    SECTOR_MAP,
    IdentifiedCompany,
    _guess_sector,
)
from src.spur_finder import SpurEndpoint
from src.utils.geo_utils import haversine_distance

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Overture Maps constants
# ---------------------------------------------------------------------------

OVERTURE_RELEASE = "2026-01-21.0"
OVERTURE_S3_PATH = (
    f"s3://overturemaps-us-west-2/release/{OVERTURE_RELEASE}"
    "/theme=places/type=place/*"
)

# Overture category values that correspond to rail-served industries.
# These are matched against the `categories.main` field (string).
RAIL_CATEGORIES: Set[str] = {
    # Manufacturing / industrial
    "manufacturing_facility",
    "factory",
    "industrial_facility",
    "industrial_estate",
    "workshop",
    "fabrication",
    # Warehousing / logistics
    "warehouse",
    "distribution_center",
    "freight_terminal",
    "logistics_facility",
    "storage_facility",
    "loading_dock",
    "intermodal_terminal",
    "cargo_terminal",
    # Energy / mining
    "mine",
    "quarry",
    "oil_and_gas",
    "refinery",
    "power_plant",
    "coal_plant",
    "natural_gas_facility",
    "petroleum_facility",
    "fuel_depot",
    "propane_supplier",
    # Agriculture
    "grain_elevator",
    "feed_mill",
    "silo",
    "farm",
    "agricultural_facility",
    "fertilizer_plant",
    "ethanol_plant",
    "food_processing",
    "dairy_facility",
    "sugar_mill",
    "flour_mill",
    # Building materials
    "lumber_yard",
    "cement_plant",
    "concrete_plant",
    "stone_quarry",
    "gravel_pit",
    "aggregate_supplier",
    # Metals / recycling
    "steel_mill",
    "scrap_metal",
    "metal_fabrication",
    "foundry",
    "recycling_facility",
    # Chemicals
    "chemical_plant",
    "chemical_facility",
    # Wholesale / trade
    "wholesale",
    "wholesale_store",
    "building_supply",
    "hardware_store",
    "trade_supplier",
}

# Cache directory for Overture results
OVERTURE_CACHE_DIR = CACHE_DIR / "overture"
OVERTURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# DuckDB availability check
# ---------------------------------------------------------------------------

def _require_duckdb():
    """Import and return the duckdb module, raising a clear error if absent."""
    try:
        import duckdb  # noqa: PLC0415
        return duckdb
    except ImportError as exc:
        raise ImportError(
            "duckdb is required for Overture Maps queries. "
            "Install it with: pip install duckdb"
        ) from exc


def _get_duckdb_connection():
    """
    Create a DuckDB connection with httpfs and spatial extensions loaded.

    Sets S3 region and disables request signing so public Overture data
    can be accessed without AWS credentials.

    Returns:
        A configured duckdb.DuckDBPyConnection instance.

    Raises:
        ImportError: If duckdb is not installed.
        RuntimeError: If extensions fail to load.
    """
    duckdb = _require_duckdb()
    conn = duckdb.connect()

    try:
        conn.execute("INSTALL httpfs; LOAD httpfs;")
        conn.execute("INSTALL spatial; LOAD spatial;")
        conn.execute("SET s3_region='us-west-2';")
        conn.execute("SET s3_access_key_id=''; SET s3_secret_access_key='';")
        conn.execute("SET s3_url_style='path';")
    except Exception as exc:
        conn.close()
        raise RuntimeError(
            f"Failed to initialise DuckDB extensions (httpfs, spatial): {exc}"
        ) from exc

    return conn


# ---------------------------------------------------------------------------
# Category filtering helpers
# ---------------------------------------------------------------------------

def _is_rail_category(category: Optional[str]) -> bool:
    """Return True if the Overture category is likely rail-served."""
    if not category:
        return False
    return category.lower() in RAIL_CATEGORIES


def _name_has_industrial_keyword(name: Optional[str]) -> bool:
    """Return True if the place name contains any industrial keyword."""
    if not name:
        return False
    name_lower = name.lower()
    return any(kw in name_lower for kw in INDUSTRIAL_KEYWORDS)


def _should_include_place(name: Optional[str], category: Optional[str]) -> bool:
    """
    Decide whether a place is worth including in the match pool.

    Includes the place if:
    - Its Overture category is in RAIL_CATEGORIES, OR
    - Its name contains an industrial keyword.
    """
    return _is_rail_category(category) or _name_has_industrial_keyword(name)


# ---------------------------------------------------------------------------
# Main fetch function
# ---------------------------------------------------------------------------

def fetch_overture_places(region_key: str) -> List[dict]:
    """
    Query Overture Maps GeoParquet files for rail-relevant places in a region.

    Uses DuckDB httpfs to query Overture S3 parquet files directly — no full
    dataset download required. Results are cached to disk as JSON.

    The bounding box comes from ``config.REGIONS[region_key]["bbox"]`` which
    has the format ``(south, west, north, east)``.

    Args:
        region_key: Key into ``config.REGIONS`` (e.g. ``"illinois"``).

    Returns:
        List of dicts, each with keys:
        ``name``, ``category``, ``address``, ``city``, ``state``,
        ``postcode``, ``lat``, ``lon``.

    Raises:
        KeyError: If ``region_key`` is not found in ``REGIONS``.
        ImportError: If duckdb is not installed.
        RuntimeError: If the DuckDB query fails.
    """
    if region_key not in REGIONS:
        raise KeyError(
            f"Unknown region '{region_key}'. "
            f"Valid keys: {sorted(REGIONS.keys())}"
        )

    cache_path = OVERTURE_CACHE_DIR / f"{region_key}.json"

    # Return cached results if available
    if cache_path.exists():
        logger.info(
            f"Loading cached Overture places for '{region_key}' from {cache_path}"
        )
        with open(cache_path) as fh:
            cached = json.load(fh)
        logger.info(f"Loaded {len(cached)} cached Overture places")
        return cached

    # Unpack bbox: (south, west, north, east)
    south, west, north, east = REGIONS[region_key]["bbox"]

    logger.info(
        f"Querying Overture Maps for '{region_key}' "
        f"bbox=({south},{west},{north},{east}) ..."
    )

    conn = _get_duckdb_connection()
    start = time.time()

    try:
        rows = conn.execute(
            f"""
            SELECT
                names."primary"              AS name,
                categories."primary"         AS category,
                addresses[1].freeform        AS address,
                addresses[1].locality        AS city,
                addresses[1].region          AS state,
                addresses[1].postcode        AS postcode,
                ST_Y(geometry)               AS lat,
                ST_X(geometry)              AS lon
            FROM read_parquet('{OVERTURE_S3_PATH}', hive_partitioning=true)
            WHERE bbox.xmin BETWEEN {west} AND {east}
              AND bbox.ymin BETWEEN {south} AND {north}
            LIMIT 500000
            """,
        ).fetchall()
    except Exception as exc:
        conn.close()
        raise RuntimeError(
            f"Overture DuckDB query failed for region '{region_key}': {exc}"
        ) from exc
    finally:
        conn.close()

    elapsed = time.time() - start
    logger.info(f"Overture raw query returned {len(rows)} rows in {elapsed:.1f}s")

    # Build list of dicts and filter to rail-relevant places
    columns = ("name", "category", "address", "city", "state", "postcode", "lat", "lon")
    places: List[dict] = []

    for row in rows:
        place = dict(zip(columns, row))

        # Skip rows with no coordinates
        if not place["lat"] or not place["lon"]:
            continue

        # Filter to industrially relevant places
        if not _should_include_place(place.get("name"), place.get("category")):
            continue

        places.append(place)

    logger.info(
        f"Filtered to {len(places)} rail-relevant Overture places "
        f"(from {len(rows)} total)"
    )

    # Persist to cache
    with open(cache_path, "w") as fh:
        json.dump(places, fh)
    logger.info(f"Cached Overture places to {cache_path}")

    return places


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def _confidence_for_place(name: Optional[str], category: Optional[str]) -> str:
    """
    Derive a confidence level for an Overture match.

    HIGH  — industrial keyword in name OR category is an explicitly rail category.
    MEDIUM — everything else that passed the inclusion filter.
    """
    name_industrial = _name_has_industrial_keyword(name)
    cat_rail = _is_rail_category(category)

    if name_industrial or cat_rail:
        return "HIGH"
    return "MEDIUM"


def _facility_type_for_place(category: Optional[str]) -> str:
    """Return a short facility type string from the Overture category."""
    if not category:
        return "overture_poi"
    return category.replace("_", " ")


def match_overture_to_endpoints(
    places: List[dict],
    endpoints: List[SpurEndpoint],
    max_distance_m: float = 500.0,
    exclude_way_ids: Optional[Set[int]] = None,
) -> List[IdentifiedCompany]:
    """
    Spatially match Overture places to spur endpoints.

    For each endpoint (optionally filtering out already-matched ones via
    ``exclude_way_ids``), finds the nearest Overture place within
    ``max_distance_m`` metres and builds an ``IdentifiedCompany``.

    When multiple places match the same endpoint the closest one is used.

    Args:
        places: List of place dicts from :func:`fetch_overture_places`.
        endpoints: Spur endpoints to match against.
        max_distance_m: Radius in metres for matching.
        exclude_way_ids: OSM way IDs already matched by a prior source;
            those endpoints are skipped.

    Returns:
        List of :class:`~src.osm_batch_identifier.IdentifiedCompany` objects
        with ``source="overture"``.
    """
    exclude: Set[int] = exclude_way_ids or set()

    # Only consider endpoints not already matched
    target_endpoints = [ep for ep in endpoints if ep.osm_way_id not in exclude]

    if not target_endpoints:
        logger.info("No unmatched endpoints to process (all excluded)")
        return []

    logger.info(
        f"Matching {len(places)} Overture places to "
        f"{len(target_endpoints)} unmatched endpoints "
        f"(max_distance={max_distance_m}m) ..."
    )
    start = time.time()

    # Build per-endpoint best-match index: {way_id: (place, distance)}
    best_match: Dict[int, Tuple[dict, float]] = {}

    for place in places:
        place_lat = place.get("lat")
        place_lon = place.get("lon")
        if place_lat is None or place_lon is None:
            continue

        place_coord = (float(place_lat), float(place_lon))

        for ep in target_endpoints:
            dist = haversine_distance(place_coord, ep.coord)
            if dist > max_distance_m:
                continue

            existing = best_match.get(ep.osm_way_id)
            if existing is None or dist < existing[1]:
                best_match[ep.osm_way_id] = (place, dist)

    elapsed = time.time() - start
    logger.info(
        f"Spatial matching done in {elapsed:.1f}s — "
        f"{len(best_match)}/{len(target_endpoints)} endpoints matched"
    )

    # Build endpoint lookup for metadata
    ep_by_id: Dict[int, SpurEndpoint] = {ep.osm_way_id: ep for ep in target_endpoints}

    companies: List[IdentifiedCompany] = []

    for way_id, (place, _dist) in best_match.items():
        ep = ep_by_id[way_id]
        name = place.get("name") or ""
        category = place.get("category")

        if not name:
            # Skip nameless POIs — not useful as prospect data
            continue

        companies.append(
            IdentifiedCompany(
                company_name=name,
                address=place.get("address") or "",
                city=place.get("city") or "",
                state=place.get("state") or "",
                zip_code=place.get("postcode") or "",
                lat=ep.lat,
                lon=ep.lon,
                facility_type=_facility_type_for_place(category),
                google_place_id="",
                osm_way_id=ep.osm_way_id,
                spur_length_m=ep.spur_length_m,
                connected_to_mainline=ep.connected_to_mainline,
                confidence=_confidence_for_place(name, category),
                sector_guess=_guess_sector(name),
                source="overture",
            )
        )

    logger.info(f"Built {len(companies)} IdentifiedCompany objects from Overture data")
    return companies


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def identify_companies_overture(
    endpoints: List[SpurEndpoint],
    region_key: str,
    max_distance_m: float = 500.0,
    exclude_way_ids: Optional[Set[int]] = None,
) -> List[IdentifiedCompany]:
    """
    Identify companies at spur endpoints using Overture Maps place data.

    This is the top-level function. It:

    1. Fetches (or loads cached) Overture places for the region.
    2. Matches them to unmatched spur endpoints by haversine distance.
    3. Returns :class:`~src.osm_batch_identifier.IdentifiedCompany` objects.

    Typical usage::

        from src.spur_finder import load_endpoints
        from src.overture_matcher import identify_companies_overture

        endpoints = load_endpoints("illinois")
        already_matched = {c.osm_way_id for c in osm_companies}
        overture_companies = identify_companies_overture(
            endpoints,
            region_key="illinois",
            max_distance_m=400.0,
            exclude_way_ids=already_matched,
        )

    Args:
        endpoints: All spur endpoints for the region.
        region_key: Key into ``config.REGIONS`` (e.g. ``"illinois"``).
        max_distance_m: Search radius in metres around each endpoint.
        exclude_way_ids: OSM way IDs already identified by another source;
            those endpoints are skipped in matching.

    Returns:
        List of :class:`~src.osm_batch_identifier.IdentifiedCompany` with
        ``source="overture"``.

    Raises:
        ImportError: If duckdb is not installed (only on first call when
            cache is cold; subsequent calls load from JSON cache).
        KeyError: If ``region_key`` is unknown.
    """
    logger.info(
        f"Overture matcher: region='{region_key}', "
        f"endpoints={len(endpoints)}, "
        f"excluded={len(exclude_way_ids or set())}, "
        f"radius={max_distance_m}m"
    )

    try:
        places = fetch_overture_places(region_key)
    except ImportError as exc:
        logger.error(
            "DuckDB not installed — cannot fetch Overture data. "
            "Run: pip install duckdb\n"
            f"Original error: {exc}"
        )
        return []
    except RuntimeError as exc:
        logger.error(f"Overture fetch failed: {exc}")
        return []

    if not places:
        logger.warning(
            f"No Overture places found for region '{region_key}'. "
            "This may indicate a bbox issue or empty Overture coverage."
        )
        return []

    return match_overture_to_endpoints(
        places=places,
        endpoints=endpoints,
        max_distance_m=max_distance_m,
        exclude_way_ids=exclude_way_ids,
    )
