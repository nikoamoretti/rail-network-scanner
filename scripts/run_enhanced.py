#!/usr/bin/env python3
"""
Run the enhanced rail network scanner pipeline for all 48 contiguous US states.

Uses three stacked discovery passes per state to maximise company coverage:

  Pass 1 — OSM Batch identifier (500m radius, free, fast)
            Covers ~15-20% of endpoints via Overpass POI matching.
  Pass 2 — Overture Maps (for endpoints unmatched after Pass 1)
            Queries the Overture Maps dataset for richer commercial data.
  Pass 3 — Nominatim reverse geocode (for endpoints still unmatched)
            Rate-limited to 1 req/s; most thorough but slowest.

Each pass receives only the endpoints that the previous pass could not resolve,
so no duplicate work is done and sources are additive.

Enhanced results are written to data/output/{state}_enhanced_rail_companies.csv
(separate from the base pipeline outputs) and merged at the end into
data/output/all_states_enhanced_rail_companies.csv.

Usage:
    python scripts/run_enhanced.py
    python scripts/run_enhanced.py --states illinois,iowa,ohio
    python scripts/run_enhanced.py --skip texas,california
    python scripts/run_enhanced.py --resume
    python scripts/run_enhanced.py --skip-overture
    python scripts/run_enhanced.py --skip-nominatim
    python scripts/run_enhanced.py --skip-overture --skip-nominatim
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

# Add project root to path so src.* and config imports resolve correctly
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import LOG_DATE_FORMAT, LOG_FORMAT, OUTPUT_DIR, REGIONS
from src.enrichment import (
    deduplicate_companies,
    flag_known_accounts,
    load_known_accounts,
    print_summary,
    save_results,
    companies_to_dataframe,
)
from src.osm_batch_identifier import (
    identify_companies_at_endpoints,
    IdentifiedCompany,
    reverse_geocode_nominatim,
    INDUSTRIAL_KEYWORDS,
)
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

# Optional Overture matcher — import gracefully so --skip-overture doesn't
# require the module to be installed.
try:
    from src.overture_matcher import identify_companies_overture
    _OVERTURE_AVAILABLE = True
except ImportError:
    identify_companies_overture = None  # type: ignore[assignment]
    _OVERTURE_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger("run_enhanced")

# Rate-limit pause between states (seconds) to avoid Overpass API bans.
INTER_STATE_SLEEP_S = 10

# Spur length filters matching run_mvp.py defaults.
MIN_LENGTH_M = 50.0
MAX_LENGTH_M = 3000.0

# Output paths for the merged all-states enhanced files.
ALL_STATES_ENHANCED_CSV = OUTPUT_DIR / "all_states_enhanced_rail_companies.csv"
ALL_STATES_ENHANCED_JSON = OUTPUT_DIR / "all_states_enhanced_rail_companies.json"

# Suffix used for enhanced per-state output files.
_ENHANCED_SUFFIX = "enhanced_rail_companies"


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class EnhancedStateResult:
    """Holds per-state pipeline statistics for the enhanced run."""

    state_key: str
    state_name: str
    endpoints_found: int = 0
    osm_batch_found: int = 0
    overture_found: int = 0
    nominatim_found: int = 0
    total_companies: int = 0
    elapsed_s: float = 0.0
    error: Optional[str] = None
    skipped: bool = False

    @property
    def success(self) -> bool:
        return self.error is None and not self.skipped


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _enhanced_output_path(state_key: str) -> Path:
    return OUTPUT_DIR / f"{state_key}_{_ENHANCED_SUFFIX}.csv"


def _enhanced_output_exists(state_key: str) -> bool:
    return _enhanced_output_path(state_key).exists()


def _save_enhanced_results(companies: List[IdentifiedCompany], state_key: str) -> Path:
    """Save enhanced results to a separate CSV (does not overwrite base results)."""
    df = companies_to_dataframe(companies)
    csv_path = _enhanced_output_path(state_key)
    json_path = OUTPUT_DIR / f"{state_key}_{_ENHANCED_SUFFIX}.json"
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)
    logger.info(f"Saved {len(df)} enhanced companies to {csv_path}")
    return csv_path


def _unmatched_endpoints(
    endpoints: list,
    matched_way_ids: Set[int],
) -> list:
    """Return endpoints whose osm_way_id is not in matched_way_ids."""
    return [ep for ep in endpoints if ep.osm_way_id not in matched_way_ids]


def _matched_way_ids(companies: List[IdentifiedCompany]) -> Set[int]:
    """Collect the set of osm_way_ids from a list of IdentifiedCompany."""
    return {c.osm_way_id for c in companies}


# ---------------------------------------------------------------------------
# Nominatim pass (standalone, for unmatched endpoints)
# ---------------------------------------------------------------------------

def _run_nominatim_pass(
    endpoints: list,
) -> List[IdentifiedCompany]:
    """
    Reverse geocode each endpoint via Nominatim and create IdentifiedCompany
    objects using the same logic as osm_batch_identifier.py.

    Rate-limited to ~1 req/sec by the shared nominatim_limiter inside
    reverse_geocode_nominatim().

    Args:
        endpoints: Spur endpoints that have not been matched by prior passes.

    Returns:
        List of IdentifiedCompany objects discovered via Nominatim.
    """
    companies: List[IdentifiedCompany] = []
    total = len(endpoints)
    logger.info(f"    Nominatim: querying {total} unmatched endpoints (1 req/s)...")

    nom_found = 0
    for i, ep in enumerate(endpoints):
        if i % 50 == 0 and i > 0:
            logger.info(f"    Nominatim: {i}/{total} processed ({nom_found} found so far)")

        result = reverse_geocode_nominatim(ep.lat, ep.lon)
        if not result:
            continue

        name = result.get("name", "")
        addr = result.get("address", {})
        category = result.get("category", "")

        # Skip roads, boundaries, and unnamed results
        if not name or category in ("highway", "boundary", "place"):
            continue

        has_industrial = any(kw in name.lower() for kw in INDUSTRIAL_KEYWORDS)

        from src.osm_batch_identifier import _guess_sector  # avoid circular at module level
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

    logger.info(f"    Nominatim: found {nom_found} facilities from {total} endpoints")
    return companies


# ---------------------------------------------------------------------------
# Per-state enhanced pipeline
# ---------------------------------------------------------------------------

def run_enhanced_state_pipeline(
    state_key: str,
    known_accounts: Set[str],
    min_length: float = MIN_LENGTH_M,
    max_length: float = MAX_LENGTH_M,
    skip_overture: bool = False,
    skip_nominatim: bool = False,
) -> EnhancedStateResult:
    """
    Execute the full three-pass enhanced pipeline for a single state.

    Args:
        state_key:       Key from REGIONS dict (e.g. "illinois").
        known_accounts:  Pre-loaded set of known account names for flagging.
        min_length:      Minimum spur length filter in metres.
        max_length:      Maximum spur length filter in metres.
        skip_overture:   If True, skip the Overture Maps pass.
        skip_nominatim:  If True, skip the Nominatim reverse geocode pass.

    Returns:
        EnhancedStateResult with per-pass counts and timing.
    """
    state_name = REGIONS[state_key]["name"]
    result = EnhancedStateResult(state_key=state_key, state_name=state_name)
    start = time.time()

    try:
        # ------------------------------------------------------------------ #
        # Phase 1: Extract rail network from OSM                              #
        # ------------------------------------------------------------------ #
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
        logger.info(
            f"  Found {spur_count} spurs + {siding_count} sidings ({len(ways)} total ways)"
        )

        # Fetch mainline nodes for better junction detection
        try:
            mainline_query = build_mainline_nodes_query(state_key)
            mainline_data = query_overpass(mainline_query)
            mainline_nodes = extract_mainline_node_ids(mainline_data)
            logger.info(f"  Fetched {len(mainline_nodes)} mainline node IDs")
        except Exception as exc:
            logger.warning(f"  Could not fetch mainline nodes: {exc}")
            mainline_nodes = None

        # ------------------------------------------------------------------ #
        # Phase 2: Extract facility-side endpoints                            #
        # ------------------------------------------------------------------ #
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

        all_companies: List[IdentifiedCompany] = []
        matched: Set[int] = set()

        # ------------------------------------------------------------------ #
        # Pass 1: OSM Batch (widened to 500m)                                #
        # ------------------------------------------------------------------ #
        logger.info("  Pass 1: OSM Batch identifier (500m radius, free)...")
        osm_companies = identify_companies_at_endpoints(
            endpoints,
            region_key=state_key,
            max_distance_m=500.0,
            use_nominatim=False,
        )
        result.osm_batch_found = len(osm_companies)
        all_companies.extend(osm_companies)
        matched.update(_matched_way_ids(osm_companies))
        logger.info(
            f"  Pass 1 complete: {result.osm_batch_found} companies "
            f"({len(endpoints) - len(matched)} endpoints remaining)"
        )

        # ------------------------------------------------------------------ #
        # Pass 2: Overture Maps                                               #
        # ------------------------------------------------------------------ #
        remaining_after_osm = _unmatched_endpoints(endpoints, matched)

        if skip_overture:
            logger.info("  Pass 2: Overture Maps — SKIPPED (--skip-overture)")
        elif not _OVERTURE_AVAILABLE:
            logger.warning(
                "  Pass 2: Overture Maps — SKIPPED (src.overture_matcher not found)"
            )
        elif not remaining_after_osm:
            logger.info("  Pass 2: Overture Maps — no remaining endpoints, skipping")
        else:
            logger.info(
                f"  Pass 2: Overture Maps for {len(remaining_after_osm)} unmatched endpoints..."
            )
            try:
                overture_companies = identify_companies_overture(
                    remaining_after_osm,
                    region_key=state_key,
                )
                result.overture_found = len(overture_companies)
                all_companies.extend(overture_companies)
                matched.update(_matched_way_ids(overture_companies))
                logger.info(
                    f"  Pass 2 complete: {result.overture_found} companies "
                    f"({len(endpoints) - len(matched)} endpoints remaining)"
                )
            except Exception as exc:
                logger.warning(f"  Pass 2 (Overture) failed: {exc}")

        # ------------------------------------------------------------------ #
        # Pass 3: Nominatim reverse geocode                                   #
        # ------------------------------------------------------------------ #
        remaining_after_overture = _unmatched_endpoints(endpoints, matched)

        if skip_nominatim:
            logger.info("  Pass 3: Nominatim — SKIPPED (--skip-nominatim)")
        elif not remaining_after_overture:
            logger.info("  Pass 3: Nominatim — no remaining endpoints, skipping")
        else:
            logger.info(
                f"  Pass 3: Nominatim reverse geocode for "
                f"{len(remaining_after_overture)} unmatched endpoints..."
            )
            try:
                nominatim_companies = _run_nominatim_pass(remaining_after_overture)
                result.nominatim_found = len(nominatim_companies)
                all_companies.extend(nominatim_companies)
                matched.update(_matched_way_ids(nominatim_companies))
                logger.info(
                    f"  Pass 3 complete: {result.nominatim_found} companies"
                )
            except Exception as exc:
                logger.warning(f"  Pass 3 (Nominatim) failed: {exc}")

        # ------------------------------------------------------------------ #
        # Merge, deduplicate, enrich, save                                    #
        # ------------------------------------------------------------------ #
        logger.info("  Merging all passes, deduplicating, enriching...")
        all_companies = flag_known_accounts(all_companies, known_accounts)
        all_companies = deduplicate_companies(all_companies)
        result.total_companies = len(all_companies)

        _save_enhanced_results(all_companies, state_key)
        print_summary(all_companies, state_key)

        logger.info(
            f"  State {state_name} — "
            f"Pass 1: OSM Batch ({result.osm_batch_found}) — "
            f"Pass 2: Overture ({result.overture_found}) — "
            f"Pass 3: Nominatim ({result.nominatim_found}) — "
            f"Total after dedup: {result.total_companies}"
        )

    except Exception as exc:
        result.error = str(exc)
        logger.error(f"  FAILED: {exc}", exc_info=True)

    result.elapsed_s = time.time() - start
    return result


# ---------------------------------------------------------------------------
# Merge helper
# ---------------------------------------------------------------------------

def merge_enhanced_state_outputs(state_keys: List[str]) -> None:
    """
    Read all per-state enhanced CSVs and concatenate into a single file.

    Only includes states for which an enhanced output file exists.
    """
    frames: List[pd.DataFrame] = []
    missing: List[str] = []

    for key in state_keys:
        csv_path = _enhanced_output_path(key)
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if "state_key" not in df.columns:
                df.insert(0, "state_key", key)
            frames.append(df)
        else:
            missing.append(key)

    if missing:
        logger.warning(
            f"No enhanced output found for {len(missing)} state(s): {', '.join(missing)}"
        )

    if not frames:
        logger.error("No enhanced state output files found — combined file not written.")
        return

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(ALL_STATES_ENHANCED_CSV, index=False)
    combined.to_json(ALL_STATES_ENHANCED_JSON, orient="records", indent=2)
    logger.info(
        f"Merged {len(combined)} total companies from {len(frames)} state(s) "
        f"-> {ALL_STATES_ENHANCED_CSV}"
    )


# ---------------------------------------------------------------------------
# Final summary table
# ---------------------------------------------------------------------------

def print_final_summary(results: List[EnhancedStateResult]) -> None:
    """Print a formatted per-state enhanced results table to stdout."""
    col_w = 18
    separator = "-" * 90
    header = (
        f"{'#':<4} {'State':<{col_w}} {'Endpoints':>10} "
        f"{'OSM':>8} {'Overture':>9} {'Nominatim':>10} "
        f"{'Total':>7} {'Time (s)':>9}  Status"
    )

    print(f"\n{'=' * 90}")
    print("RAIL NETWORK SCANNER — ENHANCED ALL-STATES SUMMARY")
    print(f"{'=' * 90}")
    print(header)
    print(separator)

    total_endpoints = 0
    total_osm = 0
    total_overture = 0
    total_nominatim = 0
    total_companies = 0
    failed: List[str] = []
    skipped_list: List[str] = []

    for idx, r in enumerate(results, start=1):
        if r.skipped:
            status = "SKIPPED"
            skipped_list.append(r.state_name)
        elif r.error:
            status = f"FAILED: {r.error[:25]}"
            failed.append(r.state_name)
        else:
            status = "OK"
            total_endpoints += r.endpoints_found
            total_osm += r.osm_batch_found
            total_overture += r.overture_found
            total_nominatim += r.nominatim_found
            total_companies += r.total_companies

        print(
            f"{idx:<4} {r.state_name:<{col_w}} {r.endpoints_found:>10} "
            f"{r.osm_batch_found:>8} {r.overture_found:>9} {r.nominatim_found:>10} "
            f"{r.total_companies:>7} {r.elapsed_s:>8.0f}s  {status}"
        )

    print(separator)
    print(
        f"{'TOTAL':<4} {'':<{col_w}} {total_endpoints:>10} "
        f"{total_osm:>8} {total_overture:>9} {total_nominatim:>10} "
        f"{total_companies:>7}"
    )
    print(f"{'=' * 90}")
    print(f"States processed: {sum(1 for r in results if r.success)}/{len(results)}")
    if failed:
        print(f"Failed  ({len(failed)}): {', '.join(failed)}")
    if skipped_list:
        print(f"Skipped ({len(skipped_list)}): {', '.join(skipped_list)}")
    print(f"Combined output:  {ALL_STATES_ENHANCED_CSV}")
    print(f"{'=' * 90}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the enhanced rail network scanner for all 48 contiguous US states.",
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
            "Skip states whose enhanced per-state output CSV already exists "
            "in data/output/. Useful for restarting an interrupted run."
        ),
    )
    parser.add_argument(
        "--skip-overture",
        action="store_true",
        help="Skip Pass 2 (Overture Maps) for all states.",
    )
    parser.add_argument(
        "--skip-nominatim",
        action="store_true",
        help="Skip Pass 3 (Nominatim reverse geocode) for all states.",
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

    # ------------------------------------------------------------------ #
    # Determine which states to process                                    #
    # ------------------------------------------------------------------ #
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
        skip_set: Set[str] = {s.strip().lower() for s in args.skip.split(",") if s.strip()}
        invalid_skip = skip_set - set(REGIONS.keys())
        if invalid_skip:
            logger.warning(f"Unknown --skip key(s) ignored: {', '.join(invalid_skip)}")
        state_keys = [k for k in state_keys if k not in skip_set]

    total = len(state_keys)

    logger.info(f"Enhanced pipeline — states to process: {total}")
    logger.info(f"Spur length filter: {args.min_length:.0f}m – {args.max_length:.0f}m")
    logger.info(f"Inter-state sleep: {args.sleep}s")
    logger.info(f"Pass 2 (Overture):   {'DISABLED' if args.skip_overture else 'ENABLED'}")
    logger.info(f"Pass 3 (Nominatim):  {'DISABLED' if args.skip_nominatim else 'ENABLED'}")
    if not _OVERTURE_AVAILABLE and not args.skip_overture:
        logger.warning(
            "src.overture_matcher not importable — Pass 2 will be skipped automatically. "
            "Install the module or use --skip-overture to suppress this warning."
        )
    if args.resume:
        logger.info("--resume active: states with existing enhanced output will be skipped")

    # Load known accounts once — shared across all states
    known_accounts = load_known_accounts()

    results: List[EnhancedStateResult] = []

    for idx, state_key in enumerate(state_keys, start=1):
        state_name = REGIONS[state_key]["name"]
        logger.info("=" * 60)
        logger.info(f"State {idx} of {total}: {state_name} ({state_key})")
        logger.info("=" * 60)

        # --resume: skip if enhanced output already present
        if args.resume and _enhanced_output_exists(state_key):
            logger.info(
                f"  Skipping {state_name} — enhanced output already exists (--resume)"
            )
            results.append(
                EnhancedStateResult(
                    state_key=state_key,
                    state_name=state_name,
                    skipped=True,
                )
            )
            continue

        result = run_enhanced_state_pipeline(
            state_key=state_key,
            known_accounts=known_accounts,
            min_length=args.min_length,
            max_length=args.max_length,
            skip_overture=args.skip_overture,
            skip_nominatim=args.skip_nominatim,
        )
        results.append(result)

        if result.success:
            logger.info(
                f"  Done: {result.total_companies} companies in {result.elapsed_s:.0f}s "
                f"(OSM={result.osm_batch_found}, "
                f"Overture={result.overture_found}, "
                f"Nominatim={result.nominatim_found})"
            )
        else:
            logger.warning(f"  State {state_name} encountered an error — continuing.")

        # Rate-limit pause between states (skip after the last one)
        if idx < total:
            logger.info(f"  Sleeping {args.sleep}s before next state...")
            time.sleep(args.sleep)

    # ------------------------------------------------------------------ #
    # Merge all per-state enhanced outputs into combined files             #
    # ------------------------------------------------------------------ #
    logger.info("=" * 60)
    logger.info("Merging all enhanced state outputs into combined files...")
    logger.info("=" * 60)
    merge_enhanced_state_outputs(state_keys)

    # Final summary table
    print_final_summary(results)

    total_elapsed = time.time() - run_start
    logger.info(f"Enhanced all-states pipeline completed in {total_elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
