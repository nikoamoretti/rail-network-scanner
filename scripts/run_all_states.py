#!/usr/bin/env python3
"""
Run the rail network scanner pipeline for all 48 contiguous US states.

Pipeline per state:
1. Extract rail spurs/sidings from OSM (Overpass API)
2. Find facility-side spur endpoints
3. Identify companies via free OSM batch identifier (no Google Places, no Nominatim)
4. Enrich, deduplicate, and save per-state results

All state results are merged at the end into a combined CSV/JSON.

Usage:
    python scripts/run_all_states.py
    python scripts/run_all_states.py --states illinois,iowa,ohio
    python scripts/run_all_states.py --skip texas,california
    python scripts/run_all_states.py --resume
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

# Add project root to path so src.* and config imports resolve correctly
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import LOG_DATE_FORMAT, LOG_FORMAT, OUTPUT_DIR, REGIONS
from src.enrichment import (
    companies_to_dataframe,
    deduplicate_companies,
    flag_known_accounts,
    load_known_accounts,
    print_summary,
    save_results,
)
from src.osm_batch_identifier import identify_companies_at_endpoints
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

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger("run_all_states")

# Rate-limit pause between states (seconds) to avoid Overpass API bans.
INTER_STATE_SLEEP_S = 10

# Spur length filters matching run_mvp.py defaults.
MIN_LENGTH_M = 50.0
MAX_LENGTH_M = 3000.0

# Output paths for the merged all-states files.
ALL_STATES_CSV = OUTPUT_DIR / "all_states_rail_companies.csv"
ALL_STATES_JSON = OUTPUT_DIR / "all_states_rail_companies.json"


@dataclass
class StateResult:
    """Holds per-state pipeline statistics."""
    state_key: str
    state_name: str
    endpoints_found: int = 0
    companies_found: int = 0
    elapsed_s: float = 0.0
    error: Optional[str] = None
    skipped: bool = False

    @property
    def success(self) -> bool:
        return self.error is None and not self.skipped


def _output_exists(state_key: str) -> bool:
    """Return True if the per-state output CSV already exists."""
    csv_path = OUTPUT_DIR / f"{state_key}_rail_companies.csv"
    return csv_path.exists()


def run_state_pipeline(
    state_key: str,
    known_accounts: set[str],
    min_length: float = MIN_LENGTH_M,
    max_length: float = MAX_LENGTH_M,
) -> StateResult:
    """
    Execute the full pipeline for a single state.

    Args:
        state_key: Key from REGIONS dict (e.g. "illinois").
        known_accounts: Pre-loaded set of known account names for flagging.
        min_length: Minimum spur length filter in metres.
        max_length: Maximum spur length filter in metres.

    Returns:
        StateResult with counts and timing for this state.
    """
    state_name = REGIONS[state_key]["name"]
    result = StateResult(state_key=state_key, state_name=state_name)
    start = time.time()

    try:
        # --- Phase 1: Extract rail network from OSM ---
        logger.info("  Phase 1: Querying Overpass for rail spurs/sidings...")
        query = build_overpass_query(state_key)
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
        logger.info(f"  Found {spur_count} spurs + {siding_count} sidings ({len(ways)} total ways)")

        # Fetch mainline nodes for better junction detection
        try:
            mainline_query = build_mainline_nodes_query(state_key)
            mainline_data = query_overpass(mainline_query)
            mainline_nodes = extract_mainline_node_ids(mainline_data)
            logger.info(f"  Fetched {len(mainline_nodes)} mainline node IDs")
        except Exception as exc:
            logger.warning(f"  Could not fetch mainline nodes: {exc}")
            mainline_nodes = None

        # --- Phase 2: Extract facility-side endpoints ---
        logger.info("  Phase 2: Extracting facility-side spur endpoints...")
        endpoints = extract_spur_endpoints(ways, nodes, mainline_nodes=mainline_nodes)
        endpoints = deduplicate_nearby_endpoints(endpoints, min_distance_m=50.0)

        before_filter = len(endpoints)
        endpoints = [
            ep for ep in endpoints
            if min_length <= ep.spur_length_m <= max_length
        ]
        logger.info(
            f"  Length filter ({min_length:.0f}-{max_length:.0f}m): "
            f"{before_filter} -> {len(endpoints)} endpoints"
        )

        save_endpoints(endpoints, state_key)
        result.endpoints_found = len(endpoints)

        # --- Phase 3: Identify companies (free OSM batch, no Nominatim) ---
        logger.info("  Phase 3: Batch OSM company identification (FREE, no Nominatim)...")
        companies = identify_companies_at_endpoints(
            endpoints,
            region_key=state_key,
            use_nominatim=False,
        )
        logger.info(f"  Identified {len(companies)} companies")

        # --- Phase 4: Enrich and deduplicate ---
        logger.info("  Phase 4: Enrichment and deduplication...")
        companies = flag_known_accounts(companies, known_accounts)
        companies = deduplicate_companies(companies)

        # Save per-state files
        save_results(companies, state_key)
        result.companies_found = len(companies)

        print_summary(companies, state_key)

    except Exception as exc:
        result.error = str(exc)
        logger.error(f"  FAILED: {exc}", exc_info=True)

    result.elapsed_s = time.time() - start
    return result


def merge_state_outputs(state_keys: list[str]) -> None:
    """
    Read all per-state CSVs and concatenate into a single combined file.

    Only includes states for which an output file exists.
    """
    frames: list[pd.DataFrame] = []
    missing: list[str] = []

    for key in state_keys:
        csv_path = OUTPUT_DIR / f"{key}_rail_companies.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            # Tag with state key so the combined file is self-describing
            if "state_key" not in df.columns:
                df.insert(0, "state_key", key)
            frames.append(df)
        else:
            missing.append(key)

    if missing:
        logger.warning(f"No output found for {len(missing)} state(s): {', '.join(missing)}")

    if not frames:
        logger.error("No state output files found — combined file not written.")
        return

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(ALL_STATES_CSV, index=False)
    combined.to_json(ALL_STATES_JSON, orient="records", indent=2)
    logger.info(
        f"Merged {len(combined)} total companies from {len(frames)} state(s) "
        f"-> {ALL_STATES_CSV}"
    )


def print_final_summary(results: list[StateResult]) -> None:
    """Print a formatted per-state results table to stdout."""
    header = f"{'#':<4} {'State':<20} {'Endpoints':>10} {'Companies':>10} {'Time (s)':>9}  Status"
    separator = "-" * 70
    print(f"\n{'=' * 70}")
    print("RAIL NETWORK SCANNER — ALL STATES SUMMARY")
    print(f"{'=' * 70}")
    print(header)
    print(separator)

    total_endpoints = 0
    total_companies = 0
    failed: list[str] = []
    skipped_list: list[str] = []

    for idx, r in enumerate(results, start=1):
        if r.skipped:
            status = "SKIPPED"
            skipped_list.append(r.state_name)
        elif r.error:
            status = f"FAILED: {r.error[:30]}"
            failed.append(r.state_name)
        else:
            status = "OK"
            total_endpoints += r.endpoints_found
            total_companies += r.companies_found

        print(
            f"{idx:<4} {r.state_name:<20} {r.endpoints_found:>10} "
            f"{r.companies_found:>10} {r.elapsed_s:>8.0f}s  {status}"
        )

    print(separator)
    print(f"{'TOTAL':<4} {'':<20} {total_endpoints:>10} {total_companies:>10}")
    print(f"{'=' * 70}")
    print(f"States processed: {sum(1 for r in results if r.success)}/{len(results)}")
    if failed:
        print(f"Failed ({len(failed)}): {', '.join(failed)}")
    if skipped_list:
        print(f"Skipped ({len(skipped_list)}): {', '.join(skipped_list)}")
    print(f"Combined output:  {ALL_STATES_CSV}")
    print(f"{'=' * 70}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the rail network scanner for all 48 contiguous US states.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--states",
        type=str,
        default="",
        metavar="STATE1,STATE2,...",
        help=(
            "Comma-separated subset of state keys to process "
            "(e.g. 'illinois,iowa,ohio'). Defaults to all states."
        ),
    )
    parser.add_argument(
        "--skip",
        type=str,
        default="",
        metavar="STATE1,STATE2,...",
        help="Comma-separated state keys to skip.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip states whose per-state output CSV already exists "
            "in data/output/. Useful for restarting an interrupted run."
        ),
    )
    parser.add_argument(
        "--min-length",
        type=float,
        default=MIN_LENGTH_M,
        help=f"Minimum spur length in metres (default: {MIN_LENGTH_M:.0f}).",
    )
    parser.add_argument(
        "--max-length",
        type=float,
        default=MAX_LENGTH_M,
        help=f"Maximum spur length in metres (default: {MAX_LENGTH_M:.0f}).",
    )
    parser.add_argument(
        "--sleep",
        type=int,
        default=INTER_STATE_SLEEP_S,
        metavar="SECONDS",
        help=(
            f"Sleep between states to respect Overpass rate limits "
            f"(default: {INTER_STATE_SLEEP_S}s)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_start = time.time()

    # Determine which states to process
    all_state_keys = list(REGIONS.keys())

    if args.states:
        requested = [s.strip().lower() for s in args.states.split(",") if s.strip()]
        invalid = [s for s in requested if s not in REGIONS]
        if invalid:
            logger.error(f"Unknown state key(s): {', '.join(invalid)}")
            logger.error(f"Valid keys: {', '.join(all_state_keys)}")
            sys.exit(1)
        state_keys = requested
    else:
        state_keys = all_state_keys

    if args.skip:
        skip_set = {s.strip().lower() for s in args.skip.split(",") if s.strip()}
        invalid_skip = skip_set - set(REGIONS.keys())
        if invalid_skip:
            logger.warning(f"Unknown --skip key(s) ignored: {', '.join(invalid_skip)}")
        state_keys = [k for k in state_keys if k not in skip_set]

    total = len(state_keys)
    logger.info(f"States to process: {total}")
    logger.info(f"Spur length filter: {args.min_length:.0f}m – {args.max_length:.0f}m")
    logger.info(f"Inter-state sleep: {args.sleep}s")
    if args.resume:
        logger.info("--resume active: states with existing output will be skipped")

    # Load known accounts once — shared across all states
    known_accounts = load_known_accounts()

    results: list[StateResult] = []

    for idx, state_key in enumerate(state_keys, start=1):
        state_name = REGIONS[state_key]["name"]
        logger.info("=" * 60)
        logger.info(f"Processing state {idx} of {total}: {state_name} ({state_key})")
        logger.info("=" * 60)

        # --resume: skip if output already present
        if args.resume and _output_exists(state_key):
            logger.info(f"  Skipping {state_name} — output already exists (--resume)")
            results.append(
                StateResult(
                    state_key=state_key,
                    state_name=state_name,
                    skipped=True,
                )
            )
            continue

        result = run_state_pipeline(
            state_key=state_key,
            known_accounts=known_accounts,
            min_length=args.min_length,
            max_length=args.max_length,
        )
        results.append(result)

        if result.success:
            logger.info(
                f"  Done: {result.companies_found} companies in {result.elapsed_s:.0f}s"
            )
        else:
            logger.warning(f"  State {state_name} encountered an error — continuing.")

        # Rate-limit pause between states (skip after the last one)
        if idx < total:
            logger.info(f"  Sleeping {args.sleep}s before next state...")
            time.sleep(args.sleep)

    # Merge all per-state outputs into combined files
    logger.info("=" * 60)
    logger.info("Merging all state outputs into combined files...")
    logger.info("=" * 60)
    merge_state_outputs(state_keys)

    # Final summary table
    print_final_summary(results)

    total_elapsed = time.time() - run_start
    logger.info(f"All-states pipeline completed in {total_elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
