#!/usr/bin/env python3
"""
Run the rail network scanner pipeline for any configured region.

Usage:
    python scripts/run_region.py illinois
    python scripts/run_region.py --list
    python scripts/run_region.py texas --skip-places  # OSM extraction only
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import LOG_DATE_FORMAT, LOG_FORMAT, REGIONS
from src.osm_extractor import build_mainline_nodes_query, build_overpass_query, parse_osm_elements, query_overpass
from src.spur_finder import (
    deduplicate_nearby_endpoints,
    extract_spur_endpoints,
    save_endpoints,
)
from src.company_identifier import identify_companies_at_endpoints
from src.enrichment import (
    deduplicate_companies,
    flag_known_accounts,
    load_known_accounts,
    print_summary,
    save_results,
)

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger("run_region")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rail Network Scanner — run for a region")
    parser.add_argument("region", nargs="?", help="Region key (e.g., illinois)")
    parser.add_argument("--list", action="store_true", help="List available regions")
    parser.add_argument("--skip-places", action="store_true", help="Skip Google Places API (OSM only)")

    args = parser.parse_args()

    if args.list:
        print("Available regions:")
        for key, region in REGIONS.items():
            print(f"  {key:<12} {region['name']} — {region['description']}")
        return

    if not args.region:
        parser.print_help()
        sys.exit(1)

    region_key = args.region.lower()
    if region_key not in REGIONS:
        print(f"Unknown region: {region_key}")
        print(f"Available: {', '.join(REGIONS.keys())}")
        sys.exit(1)

    start = time.time()
    region = REGIONS[region_key]
    logger.info(f"Running pipeline for {region['name']}")

    # Phase 1: OSM extraction
    query = build_overpass_query(region_key)
    raw_data = query_overpass(query)
    ways, nodes = parse_osm_elements(raw_data)
    logger.info(f"Extracted {len(ways)} ways from OSM")

    # Fetch mainline nodes for better junction detection
    from src.osm_extractor import extract_mainline_node_ids
    try:
        mainline_query = build_mainline_nodes_query(region_key)
        mainline_data = query_overpass(mainline_query)
        mainline_nodes = extract_mainline_node_ids(mainline_data)
        logger.info(f"Fetched {len(mainline_nodes)} mainline node IDs")
    except Exception as e:
        logger.warning(f"Could not fetch mainline nodes: {e}")
        mainline_nodes = None

    # Phase 2: Spur endpoints
    endpoints = extract_spur_endpoints(ways, nodes, mainline_nodes=mainline_nodes)
    endpoints = deduplicate_nearby_endpoints(endpoints)
    save_endpoints(endpoints, region_key)
    logger.info(f"Found {len(endpoints)} unique spur endpoints")

    if args.skip_places:
        logger.info("Skipping Places API (--skip-places)")
        elapsed = time.time() - start
        logger.info(f"OSM extraction completed in {elapsed:.0f}s")
        return

    # Phase 3: Company identification
    companies = identify_companies_at_endpoints(endpoints)

    # Phase 5: Enrichment
    known_accounts = load_known_accounts()
    companies = flag_known_accounts(companies, known_accounts)
    companies = deduplicate_companies(companies)

    csv_path, json_path = save_results(companies, region_key)
    print_summary(companies, region_key)

    elapsed = time.time() - start
    logger.info(f"Pipeline completed in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
