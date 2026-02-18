#!/usr/bin/env python3
"""
MVP Pipeline: End-to-end rail network scan for Illinois.

Runs all phases:
1. Extract rail spurs/sidings from OSM
2. Find facility-side endpoints
3. Identify companies (free OSM-based by default, --google-places for paid)
4. Enrich, deduplicate, and save results
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import LOG_DATE_FORMAT, LOG_FORMAT
from src.osm_extractor import (
    build_mainline_nodes_query,
    build_overpass_query,
    extract_mainline_node_ids,
    parse_osm_elements,
    query_overpass,
)
from src.spur_finder import (
    deduplicate_nearby_endpoints,
    extract_spur_endpoints,
    save_endpoints,
)
from src.enrichment import (
    deduplicate_companies,
    flag_known_accounts,
    load_known_accounts,
    print_summary,
    save_results,
)

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger("run_mvp")

REGION = "illinois"


def main() -> None:
    parser = argparse.ArgumentParser(description="Rail Network Scanner MVP — Illinois")
    parser.add_argument("--google-places", action="store_true",
                        help="Use Google Places API ($0.032/req) instead of free OSM")
    parser.add_argument("--no-nominatim", action="store_true",
                        help="Skip Nominatim pass (OSM POI only, fastest)")
    parser.add_argument("--min-length", type=float, default=50,
                        help="Minimum spur length in meters (default: 50)")
    parser.add_argument("--max-length", type=float, default=3000,
                        help="Maximum spur length in meters (default: 3000)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of endpoints to process (0=all)")
    args = parser.parse_args()

    start = time.time()
    logger.info(f"Starting MVP pipeline for {REGION}")

    # --- Phase 1: Extract rail network from OSM ---
    logger.info("=" * 60)
    logger.info("PHASE 1: Extracting rail spurs/sidings from OpenStreetMap")
    logger.info("=" * 60)

    query = build_overpass_query(REGION)
    raw_data = query_overpass(query)
    ways, nodes = parse_osm_elements(raw_data)

    spur_count = sum(
        1 for w in ways.values()
        if w["tags"].get("service") == "spur" or w["tags"].get("railway") == "spur"
    )
    siding_count = sum(
        1 for w in ways.values()
        if w["tags"].get("service") == "siding" or w["tags"].get("railway") == "siding"
    )
    logger.info(f"Found {spur_count} spurs and {siding_count} sidings ({len(ways)} total ways)")

    # Fetch mainline nodes for better junction detection
    try:
        mainline_query = build_mainline_nodes_query(REGION)
        mainline_data = query_overpass(mainline_query)
        mainline_nodes = extract_mainline_node_ids(mainline_data)
        logger.info(f"Fetched {len(mainline_nodes)} mainline node IDs")
    except Exception as e:
        logger.warning(f"Could not fetch mainline nodes: {e}")
        mainline_nodes = None

    # --- Phase 2: Extract facility-side endpoints ---
    logger.info("=" * 60)
    logger.info("PHASE 2: Extracting facility-side spur endpoints")
    logger.info("=" * 60)

    endpoints = extract_spur_endpoints(ways, nodes, mainline_nodes=mainline_nodes)
    endpoints = deduplicate_nearby_endpoints(endpoints, min_distance_m=50.0)

    # Filter by spur length
    before_filter = len(endpoints)
    endpoints = [
        ep for ep in endpoints
        if args.min_length <= ep.spur_length_m <= args.max_length
    ]
    logger.info(f"Filtered by length ({args.min_length}-{args.max_length}m): {before_filter} → {len(endpoints)}")

    if args.limit > 0:
        endpoints = endpoints[:args.limit]
        logger.info(f"Limited to {args.limit} endpoints")

    save_endpoints(endpoints, REGION)
    logger.info(f"Total endpoints to process: {len(endpoints)}")

    # --- Phase 3: Identify companies ---
    logger.info("=" * 60)
    if args.google_places:
        logger.info("PHASE 3: Identifying companies via Google Places API (PAID)")
        from src.company_identifier import identify_companies_at_endpoints
        companies = identify_companies_at_endpoints(endpoints)
    else:
        logger.info("PHASE 3: Identifying companies via OSM batch + Nominatim (FREE)")
        from src.osm_batch_identifier import identify_companies_at_endpoints
        companies = identify_companies_at_endpoints(
            endpoints,
            region_key=REGION,
            use_nominatim=not args.no_nominatim,
        )
    logger.info("=" * 60)
    logger.info(f"Identified {len(companies)} companies")

    # --- Phase 5: Enrich and deduplicate ---
    logger.info("=" * 60)
    logger.info("PHASE 5: Enrichment and deduplication")
    logger.info("=" * 60)

    known_accounts = load_known_accounts()
    companies = flag_known_accounts(companies, known_accounts)
    companies = deduplicate_companies(companies)

    # Save final results
    csv_path, json_path = save_results(companies, REGION)

    # Print summary
    print_summary(companies, REGION)

    elapsed = time.time() - start
    logger.info(f"Pipeline completed in {elapsed:.0f}s")
    logger.info(f"Results: {csv_path}")


if __name__ == "__main__":
    main()
