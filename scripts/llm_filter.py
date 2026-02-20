#!/usr/bin/env python3
"""
llm_filter.py
=============
Uses Claude Haiku to evaluate every row in all_states_final.csv and determine:
  1. Is this a real company that could plausibly ship/receive goods by rail?
  2. What industry sector does it belong to?

Sends batches of 100 companies to the LLM, caches results by batch hash,
then overwrites all_states_final.csv with only the kept rows.

Python 3.9 compatible.

Usage:
    python scripts/llm_filter.py
    python scripts/llm_filter.py --api-key sk-ant-...
    python scripts/llm_filter.py --batch-size 50
    python scripts/llm_filter.py --dry-run          # skip API calls, keep all
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Bootstrap path so config is importable when run from any directory.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from config import LOG_DATE_FORMAT, LOG_FORMAT

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger("llm_filter")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent
INPUT_CSV = BASE_DIR / "data/output/all_states_final.csv"
OUTPUT_CSV = BASE_DIR / "data/output/all_states_final.csv"  # overwrite in-place
CACHE_DIR = BASE_DIR / ".cache/llm_filter"

# ---------------------------------------------------------------------------
# Valid US states — full names (lowercase) and 2-letter abbreviations
# ---------------------------------------------------------------------------
_US_STATE_NAMES: frozenset = frozenset({
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
    "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana",
    "maine", "maryland", "massachusetts", "michigan", "minnesota",
    "mississippi", "missouri", "montana", "nebraska", "nevada",
    "new hampshire", "new jersey", "new mexico", "new york",
    "north carolina", "north dakota", "ohio", "oklahoma", "oregon",
    "pennsylvania", "rhode island", "south carolina", "south dakota",
    "tennessee", "texas", "utah", "vermont", "virginia", "washington",
    "west virginia", "wisconsin", "wyoming",
})

_US_STATE_ABBREVS: frozenset = frozenset({
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
})

# Also accept title-case full names (e.g. "Texas", "New York")
_US_STATE_NAMES_TITLE: frozenset = frozenset(n.title() for n in _US_STATE_NAMES)


def is_us_state(state_key: str, state_abbrev: str) -> bool:
    """
    Return True if the row belongs to a US state.

    Checks state_key (which stores lowercase state names like 'new york')
    against all 50 full state names.  Also checks the state column for
    2-letter abbreviations as a fallback.
    """
    sk = state_key.strip().lower()
    if sk in _US_STATE_NAMES:
        return True
    # Some rows may have state_key as title-case or abbreviation
    if sk.upper() in _US_STATE_ABBREVS:
        return True
    st = state_abbrev.strip()
    if st.upper() in _US_STATE_ABBREVS:
        return True
    if st.lower() in _US_STATE_NAMES:
        return True
    if st in _US_STATE_NAMES_TITLE:
        return True
    return False


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """\
You are evaluating whether businesses are likely rail-served industrial facilities that ship or receive goods by railroad.

For each entry, determine:
1. "keep": true if this is a real company that could plausibly use rail freight (manufacturers, distributors, warehouses, terminals, chemical plants, refineries, lumber yards, grain elevators, mines, power plants, steel mills, food processors, paper mills, etc.). false if it's a retail store, restaurant, gas station, post office, auto repair shop, residential building, street name, highway name, geographic feature, government office, church, school, park, trail, or any other non-rail-shipping entity.
2. "sector": One of: Agriculture, Automotive Manufacturing, Building Materials, Chemicals, Energy, Food & Beverage, Forest Products, Government/Military, Logistics, Manufacturing, Metals, Mining, Rail Services, Utilities, Waste/Recycling, or Other Industrial. Only fill if keep=true.

Return ONLY a JSON array with objects {"idx": <index>, "keep": true/false, "sector": "<sector or null>"}.

Entries to evaluate:
"""


def _build_batch_prompt(entries: List[Tuple[int, dict]]) -> str:
    """Build the user-turn text for a batch of entries."""
    lines: List[str] = []
    for idx, row in entries:
        company = row.get("company_name", "").strip()
        city = row.get("city", "").strip() or "(unknown)"
        state = (row.get("state", "") or row.get("state_key", "")).strip() or "(unknown)"
        ftype = row.get("facility_type", "").strip() or "(unknown)"
        lines.append(f"[{idx}] {company} | {city}, {state} | facility_type: {ftype}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _batch_cache_key(batch_text: str) -> str:
    """Stable SHA-256 hex digest of the batch prompt text."""
    return hashlib.sha256(batch_text.encode("utf-8")).hexdigest()


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def _load_cache(key: str) -> Optional[List[dict]]:
    path = _cache_path(key)
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.warning("Corrupted cache file %s — will re-fetch", path)
    return None


def _save_cache(key: str, data: List[dict]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(_cache_path(key), "w", encoding="utf-8") as f:
        json.dump(data, f)


# ---------------------------------------------------------------------------
# Anthropic API call
# ---------------------------------------------------------------------------

def _call_llm(client, batch_text: str, batch_size: int) -> List[dict]:
    """
    Call Claude Haiku with the batch prompt.
    Returns a list of dicts: [{"idx": int, "keep": bool, "sector": str|None}]

    Retries once on JSON parse failure; on second failure returns safe default
    (keep=True for all).
    """
    user_message = batch_text

    for attempt in range(2):
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=4096,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        raw = response.content[0].text.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        if raw.endswith("```"):
            raw = raw[: raw.rfind("```")].strip()

        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
            logger.warning("LLM returned non-list JSON on attempt %d; retrying", attempt + 1)
        except json.JSONDecodeError as exc:
            logger.warning(
                "JSON parse error on attempt %d: %s — raw: %.200s",
                attempt + 1, exc, raw,
            )

    logger.error("Both LLM parse attempts failed; marking entire batch as keep=True")
    return []  # Empty means caller will use safe default


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM filter for rail company dataset")
    parser.add_argument("--api-key", default=None, help="Anthropic API key (overrides .env)")
    parser.add_argument("--batch-size", type=int, default=100, help="Companies per LLM call")
    parser.add_argument("--dry-run", action="store_true", help="Skip API calls; keep all rows")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    logger.info("=" * 70)
    logger.info("Rail Network Scanner — LLM Filter (Claude Haiku)")
    logger.info("=" * 70)

    # ------------------------------------------------------------------ #
    # Load CSV                                                            #
    # ------------------------------------------------------------------ #
    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_rows: List[dict] = list(reader)
        fieldnames: List[str] = list(reader.fieldnames or [])

    total_input = len(all_rows)
    logger.info("Loaded %d rows from %s", total_input, INPUT_CSV)

    # ------------------------------------------------------------------ #
    # US state filter (fast pre-pass, no LLM needed)                     #
    # ------------------------------------------------------------------ #
    us_rows: List[dict] = []
    non_us_removed: List[dict] = []

    for row in all_rows:
        if is_us_state(row.get("state_key", ""), row.get("state", "")):
            us_rows.append(row)
        else:
            non_us_removed.append(row)

    logger.info(
        "US state filter: kept %d / removed %d non-US rows",
        len(us_rows), len(non_us_removed),
    )

    # ------------------------------------------------------------------ #
    # Prepare LLM verdicts storage                                        #
    # ------------------------------------------------------------------ #
    # Map from row index (in us_rows) → {"keep": bool, "sector": str|None}
    verdicts: Dict[int, dict] = {}

    if args.dry_run:
        logger.info("DRY RUN — skipping LLM calls; all rows marked keep=True")
        for i in range(len(us_rows)):
            verdicts[i] = {"keep": True, "sector": None}
    else:
        # Set up Anthropic client
        import os
        import anthropic

        api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            logger.error("No ANTHROPIC_API_KEY found. Pass --api-key or set in .env")
            sys.exit(1)

        client = anthropic.Anthropic(api_key=api_key)

        batch_size = args.batch_size
        batches: List[List[Tuple[int, dict]]] = []
        for start in range(0, len(us_rows), batch_size):
            batch = [(i, us_rows[i]) for i in range(start, min(start + batch_size, len(us_rows)))]
            batches.append(batch)

        total_batches = len(batches)
        logger.info(
            "Processing %d rows in %d batches of %d",
            len(us_rows), total_batches, batch_size,
        )

        cache_hits = 0
        api_calls = 0

        for batch_num, batch in enumerate(batches, start=1):
            batch_text = _build_batch_prompt(batch)
            cache_key = _batch_cache_key(batch_text)

            cached = _load_cache(cache_key)
            if cached is not None:
                results = cached
                cache_hits += 1
            else:
                results = _call_llm(client, batch_text, len(batch))
                _save_cache(cache_key, results)
                api_calls += 1
                # Small delay between API calls to be polite
                time.sleep(0.1)

            # Build a lookup: local idx within batch → verdict
            result_by_idx: Dict[int, dict] = {}
            for item in results:
                if isinstance(item, dict) and "idx" in item:
                    result_by_idx[item["idx"]] = item

            # Apply verdicts; safe default = keep=True
            for row_idx, row in batch:
                if row_idx in result_by_idx:
                    item = result_by_idx[row_idx]
                    verdicts[row_idx] = {
                        "keep": bool(item.get("keep", True)),
                        "sector": item.get("sector") or None,
                    }
                else:
                    # Not returned by LLM (parse failure or missing) — safe keep
                    verdicts[row_idx] = {"keep": True, "sector": None}

            if batch_num % 10 == 0 or batch_num == total_batches:
                kept_so_far = sum(1 for v in verdicts.values() if v["keep"])
                logger.info(
                    "Progress: batch %d/%d | cache hits: %d | api calls: %d | kept so far: %d",
                    batch_num, total_batches, cache_hits, api_calls, kept_so_far,
                )

        logger.info(
            "Done fetching verdicts. Cache hits: %d | API calls: %d",
            cache_hits, api_calls,
        )

    # ------------------------------------------------------------------ #
    # Apply verdicts                                                       #
    # ------------------------------------------------------------------ #
    # Ensure "keep" column exists in fieldnames
    if "keep" not in fieldnames:
        fieldnames = fieldnames + ["keep"]

    kept_rows: List[dict] = []
    llm_removed: List[dict] = []

    for i, row in enumerate(us_rows):
        verdict = verdicts.get(i, {"keep": True, "sector": None})
        row["keep"] = "true" if verdict["keep"] else "false"

        # Update sector_guess if LLM provided one and current is blank/unknown
        if verdict["sector"]:
            existing = row.get("sector_guess", "").strip()
            if not existing or existing.lower() in ("unknown", "none", "nan", ""):
                row["sector_guess"] = verdict["sector"]

        if verdict["keep"]:
            kept_rows.append(row)
        else:
            llm_removed.append(row)

    # ------------------------------------------------------------------ #
    # Write output (overwrite in place)                                   #
    # ------------------------------------------------------------------ #
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(kept_rows)

    logger.info("Wrote %d rows to %s", len(kept_rows), OUTPUT_CSV)

    # ------------------------------------------------------------------ #
    # Final report                                                        #
    # ------------------------------------------------------------------ #
    print()
    print("=" * 70)
    print("FINAL REPORT")
    print("=" * 70)
    print(f"  Input rows:                      {total_input:>8,}")
    print(f"  Non-US rows removed:             {len(non_us_removed):>8,}")
    print(f"  LLM filtered (keep=false):       {len(llm_removed):>8,}")
    print(f"  Final rows kept:                 {len(kept_rows):>8,}")
    print()

    # Sector breakdown
    sector_counts: Counter = Counter()
    for row in kept_rows:
        sector = row.get("sector_guess", "").strip() or "(none)"
        sector_counts[sector] += 1

    print("--- Sector Breakdown (kept rows) ---")
    for sector, count in sector_counts.most_common():
        pct = 100 * count / len(kept_rows) if kept_rows else 0
        print(f"  {sector:<35s} {count:>7,}  ({pct:.1f}%)")

    print()
    print("--- Top 30 Company Names ---")
    name_counts: Counter = Counter(r["company_name"].strip() for r in kept_rows)
    for rank, (name, count) in enumerate(name_counts.most_common(30), start=1):
        print(f"  {rank:>2}. {name:<50s} {count:>4}x")

    print()
    print("--- Sample of 20 Removed Entries ---")
    sample_removed = llm_removed[:20]
    if not sample_removed:
        sample_removed = non_us_removed[:20]
    for row in sample_removed:
        name = row.get("company_name", "").strip()
        city = row.get("city", "").strip() or "(no city)"
        state = (row.get("state", "") or row.get("state_key", "")).strip()
        ftype = row.get("facility_type", "").strip()
        print(f"  {name:<45s} | {city:<20s} {state:<5s} | {ftype}")

    print()
    print("=" * 70)
    print("Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
