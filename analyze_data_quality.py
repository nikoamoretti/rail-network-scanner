"""
Data quality analysis for all_states_classified.csv
Focuses on company_name field integrity.
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import pandas as pd

CSV_PATH = Path("/Users/nicoamoretti/nico_repo/rail-network-scanner/data/output/all_states_classified.csv")

# ── Patterns that indicate a "not a real company name" value ────────────────

# Road/street/geographic keywords
GEO_KEYWORDS = re.compile(
    r"\b(road|rd|street|st|avenue|ave|ave\.|blvd|boulevard|highway|hwy|"
    r"lane|ln|drive|dr|court|ct|place|pl|way|route|rte|pike|pkwy|parkway|"
    r"trail|path|crossing|junction|jct|bridge|creek|river|lake|pond|bay|"
    r"island|mountain|mt|hill|valley|hollow|holler|run)\b",
    re.IGNORECASE,
)

# Looks like pure coordinates or numeric noise
COORD_OR_NUMERIC = re.compile(r"^-?\d+(\.\d+)?(,\s*-?\d+(\.\d+)?)?$")

# Very short (1-2 chars) or all-digits
TRIVIAL = re.compile(r"^\d+$|^.{0,2}$")

# Looks like a raw OSM tag / machine-generated ID
OSM_TAG_LIKE = re.compile(r"^(node|way|relation)\s*\d+", re.IGNORECASE)


def classify_name(name: str) -> str:
    """Return 'empty', 'geo_keyword', 'numeric', 'trivial', 'osm_tag', or 'real'."""
    if pd.isna(name) or str(name).strip() == "":
        return "empty"
    s = str(name).strip()
    if COORD_OR_NUMERIC.match(s):
        return "numeric"
    if TRIVIAL.match(s):
        return "trivial"
    if OSM_TAG_LIKE.match(s):
        return "osm_tag"
    if GEO_KEYWORDS.search(s):
        return "geo_keyword"
    return "real"


def separator(title: str = "", width: int = 70) -> None:
    if title:
        print(f"\n{'─' * 4}  {title}  {'─' * (width - len(title) - 8)}")
    else:
        print("─" * width)


def main() -> None:
    df = pd.read_csv(CSV_PATH, dtype=str, low_memory=False)
    total = len(df)
    print(f"Total rows: {total:,}")

    # ── 1. Empty / blank / NaN company_name ────────────────────────────────
    separator("1. EMPTY / BLANK / NaN company_name")
    empty_mask = df["company_name"].isna() | (df["company_name"].str.strip() == "")
    n_empty = empty_mask.sum()
    print(f"  Empty company_name: {n_empty:,}  ({n_empty / total * 100:.1f}%)")

    # ── 2. Weird / address-like names ──────────────────────────────────────
    separator("2. WEIRD NAMES (address-like, numeric, trivial, OSM tag)")
    df["_name_class"] = df["company_name"].apply(classify_name)

    bad_classes = {"empty", "geo_keyword", "numeric", "trivial", "osm_tag"}
    bad_mask = df["_name_class"].isin(bad_classes)
    n_bad = bad_mask.sum()
    print(f"  Total unusable names: {n_bad:,}  ({n_bad / total * 100:.1f}%)\n")
    print("  Breakdown by class:")
    for cls, count in df["_name_class"].value_counts().items():
        flag = " <-- BAD" if cls in bad_classes else ""
        print(f"    {cls:<14} {count:>6,}  ({count / total * 100:5.1f}%){flag}")

    # ── 3. Breakdown by source ──────────────────────────────────────────────
    separator("3. UNUSABLE NAMES BY SOURCE")
    source_total = df.groupby("source").size().rename("total")
    source_bad   = df[bad_mask].groupby("source").size().rename("bad")
    source_stats = pd.concat([source_total, source_bad], axis=1).fillna(0).astype(int)
    source_stats["bad_%"] = (source_stats["bad"] / source_stats["total"] * 100).round(1)
    source_stats = source_stats.sort_values("bad", ascending=False)
    print(source_stats.to_string())

    # ── 4. 20 sample rows with empty/weird company_name ────────────────────
    separator("4. SAMPLE ROWS — EMPTY / WEIRD company_name (20)")
    sample_bad = (
        df[bad_mask][["state", "company_name", "_name_class", "address", "facility_type", "source"]]
        .sample(n=min(20, n_bad), random_state=42)
    )
    pd.set_option("display.max_colwidth", 40)
    pd.set_option("display.width", 140)
    print(sample_bad.to_string(index=False))

    # ── 5. 20 sample rows with REAL company names ──────────────────────────
    separator("5. SAMPLE ROWS — REAL company_name (20)")
    real_mask = df["_name_class"] == "real"
    n_real = real_mask.sum()
    sample_real = (
        df[real_mask][["state", "company_name", "facility_type", "sector_guess", "source"]]
        .sample(n=min(20, n_real), random_state=42)
    )
    print(sample_real.to_string(index=False))

    # ── 6. Summary percentage ───────────────────────────────────────────────
    separator("6. USABILITY SUMMARY")
    n_usable = n_real
    n_unusable = n_bad
    print(f"  Usable   (real name):  {n_usable:>6,}  ({n_usable / total * 100:.1f}%)")
    print(f"  Unusable (bad/empty):  {n_unusable:>6,}  ({n_unusable / total * 100:.1f}%)")
    print(f"  Total rows:            {total:>6,}")
    print()

    # Extra: most common "geo_keyword" names (sanity check)
    separator("BONUS — Top 15 most frequent geo_keyword names")
    geo_names = df[df["_name_class"] == "geo_keyword"]["company_name"].value_counts().head(15)
    print(geo_names.to_string())

    # Extra: unique sources
    separator("BONUS — All unique sources in dataset")
    for src in sorted(df["source"].dropna().unique()):
        print(f"  {src}")


if __name__ == "__main__":
    main()
