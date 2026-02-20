#!/usr/bin/env python3
"""
Merge Overture-source entries from the old classified CSV into the new
enhanced rail companies CSV produced by the latest Nominatim run.

Deduplication key: (company_name, lat rounded to 5dp, lon rounded to 5dp).

Input 1:  data/output/all_states_enhanced_rail_companies.csv  (72 066 rows, osm_batch + nominatim)
Input 2:  data/output/all_states_classified.csv               (55 855 rows, osm_batch + overture)
Output:   data/output/all_states_merged.csv
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

# Make project root importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import LOG_DATE_FORMAT, LOG_FORMAT, OUTPUT_DIR

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger("merge_overture")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
NEW_FILE = OUTPUT_DIR / "all_states_enhanced_rail_companies.csv"
OLD_CLASSIFIED = OUTPUT_DIR / "all_states_classified.csv"
OUTPUT_FILE = OUTPUT_DIR / "all_states_merged.csv"

# Precision for lat/lon dedup key (5 decimal places ≈ 1.1 m resolution)
COORD_PRECISION = 5


def make_dedup_key(df: pd.DataFrame) -> pd.Series:
    """Create a deduplication key string: 'name|lat_5dp|lon_5dp'."""
    name = df["company_name"].fillna("").str.strip().str.lower()
    lat = df["lat"].round(COORD_PRECISION).astype(str)
    lon = df["lon"].round(COORD_PRECISION).astype(str)
    return name + "|" + lat + "|" + lon


def main() -> None:
    # ------------------------------------------------------------------
    # Load inputs
    # ------------------------------------------------------------------
    logger.info("Loading new enhanced file: %s", NEW_FILE)
    df_new = pd.read_csv(NEW_FILE, low_memory=False)
    logger.info("  %d rows loaded (sources: %s)", len(df_new), df_new["source"].value_counts().to_dict())

    logger.info("Loading old classified file: %s", OLD_CLASSIFIED)
    df_old = pd.read_csv(OLD_CLASSIFIED, low_memory=False)
    logger.info("  %d rows loaded (sources: %s)", len(df_old), df_old["source"].value_counts().to_dict())

    # ------------------------------------------------------------------
    # Extract Overture-only rows from old classified file
    # ------------------------------------------------------------------
    df_overture = df_old[df_old["source"] == "overture"].copy()
    logger.info("Overture-only rows extracted: %d", len(df_overture))

    # ------------------------------------------------------------------
    # Build dedup key sets
    # ------------------------------------------------------------------
    new_keys = set(make_dedup_key(df_new))
    logger.info("Unique keys in new file: %d", len(new_keys))

    overture_keys = make_dedup_key(df_overture)
    df_overture["_dedup_key"] = overture_keys

    # Keep only overture rows NOT already present in the new file
    df_overture_new = df_overture[~df_overture["_dedup_key"].isin(new_keys)].copy()
    df_overture_new = df_overture_new.drop(columns=["_dedup_key"])
    logger.info("Overture rows not yet in new file: %d", len(df_overture_new))
    logger.info("Overture rows skipped (already present): %d", len(df_overture) - len(df_overture_new))

    # ------------------------------------------------------------------
    # Concatenate and write output
    # ------------------------------------------------------------------
    df_merged = pd.concat([df_new, df_overture_new], ignore_index=True)
    logger.info("Merged total rows: %d", len(df_merged))
    logger.info("Source breakdown: %s", df_merged["source"].value_counts().to_dict())

    df_merged.to_csv(OUTPUT_FILE, index=False)
    logger.info("Saved merged file to: %s", OUTPUT_FILE)

    print(f"\nMerge complete:")
    print(f"  New enhanced rows  : {len(df_new):,}")
    print(f"  Overture rows added: {len(df_overture_new):,}")
    print(f"  Overture dupes skipped: {len(df_overture) - len(df_overture_new):,}")
    print(f"  Total merged rows  : {len(df_merged):,}")


if __name__ == "__main__":
    main()
