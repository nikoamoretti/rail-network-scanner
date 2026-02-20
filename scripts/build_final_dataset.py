"""
build_final_dataset.py
======================
Reads all_states_cleaned.csv, applies aggressive filtering and deduplication,
and writes all_states_final.csv with a comprehensive report.

Python 3.9 compatible.
"""

from __future__ import annotations

import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from rapidfuzz import fuzz

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path("/Users/nicoamoretti/nico_repo/rail-network-scanner")
INPUT_CSV = BASE_DIR / "data/output/all_states_cleaned.csv"
OUTPUT_CSV = BASE_DIR / "data/output/all_states_final.csv"

# ---------------------------------------------------------------------------
# Quality ordering (lower index = better)
# ---------------------------------------------------------------------------
QUALITY_RANK: Dict[str, int] = {"complete": 0, "no_location": 1, "low_quality_name": 2}

# ---------------------------------------------------------------------------
# Junk-detection patterns
# ---------------------------------------------------------------------------

# US Highway / route patterns
_HIGHWAY_PATTERNS: List[re.Pattern] = [
    # Basic: "US 60", "US-52", "U.S. 101"
    re.compile(r"^(US|U\.S\.)\s*[-#]?\s*\d+[\w\s\-\.;]*$", re.I),
    # Interstate: "I-80", "I 80", "I80"
    re.compile(r"^I\s*[-]?\s*\d+[\w\s\-\.;]*$", re.I),
    re.compile(r"^(Interstate)\s+\d+[\w\s\-\.;]*$", re.I),
    # State routes: "State Route 9", "SR 45", "SH 30", "State Highway 1"
    re.compile(r"^(State\s+Route|SR|SH|State\s+Highway|State\s+Road)\s*[-#]?\s*\d+[\w\s\-\.;]*$", re.I),
    # County roads: "County Road 12", "CR 100", "CR N2185", "CR 7.4"
    re.compile(r"^(County\s+Road|CR|County\s+Route)\s*[-#]?\s*[A-Z0-9][\w\.\-;]*$", re.I),
    # FM/RM roads: "FM 1960", "RM 620"
    re.compile(r"^(FM|RM|Ranch\s+(?:to\s+Market|Road)|Farm\s+(?:to\s+Market|Road))\s*[-#]?\s*\d+[\w\s\-\.;]*$", re.I),
    # Highway: "Highway 101", "Hwy 30"
    re.compile(r"^(Highway|Hwy|HWY)\s+\d+[\w\s\-\.;]*$", re.I),
    # Route: "Route 66", "RT 9", "RTE 1"
    re.compile(r"^(Route|RT|RTE)\s+\d+[\w\s\-\.;]*$", re.I),
    # State abbreviation + number: "NY 481", "GA 30", "M 35"
    re.compile(r"^(NY|CA|TX|OH|IL|PA|GA|FL|WI|MN|IA|KS|MO|AR|LA|MS|AL|TN|KY|IN|MI|NC|SC|VA|WV|MD|DE|NJ|CT|MA|RI|NH|VT|ME|ND|SD|NE|CO|UT|AZ|NM|NV|ID|MT|WY|OR|WA|AK|HI|OK|M)\s*[-]?\s*\d+[\w\s\-\.;]*$", re.I),
    # Directional prefix + route type + number
    re.compile(r"^(North|South|East|West|NE|NW|SE|SW)\s+(Interstate|I[-\s]|US\s|Highway|Hwy|Route|SR)\s*\d+", re.I),
    # Business/Alt/Loop variants: "Business US 20", "US 20 Business", "US 64 ALT", "US 95 ALT"
    re.compile(r"^(Business|Loop|Spur|Alt|Alternate|Business\s+Loop)\s+(US|I|Highway|Hwy|Route)\s*[-#]?\s*\d+", re.I),
    re.compile(r"^(US|I|SR|SH|FM|RM|Highway|Hwy|Route)\s*[-#]?\s*\d+[\w\s]*\s+(Business|Alt|Alternate|Bus|Express|Spur|Hist|Historic|Bypass|BYP|ExpressLanes|Metro\s+ExpressLanes)[\w\s;]*$", re.I),
    # Compound routes joined by semicolons: "US 82;AL 14", "US 6;US 34;NE 44"
    re.compile(r"^(US|I-|SR|FM|CR|RM|Highway|Hwy)\s*\d[\w\s\-]*(?:;\s*(US|I-|SR|SH|FM|CR|RM|[A-Z]{2})\s*[-]?\s*\d[\w\s\-]*)+$", re.I),
    # Truck routes, access/service roads (bare designations)
    re.compile(r"^(Truck\s+Route|Access\s+Road\s*[A-Z0-9]*|Service\s+Road)\s*\d*$", re.I),
]

# Street / road name patterns
_STREET_SUFFIXES = (
    "street", "st", "avenue", "ave", "boulevard", "blvd", "road", "rd",
    "drive", "dr", "lane", "ln", "way", "place", "pl", "court", "ct",
    "circle", "cir", "trail", "trl", "parkway", "pkwy", "terrace", "ter",
    "highway", "hwy", "freeway", "expressway", "pike", "turnpike",
    "frontage road", "access road", "industrial drive", "railroad avenue",
    "industrial parkway", "industrial blvd", "railroad blvd",
)

_STREET_PATTERN = re.compile(
    r"^(\w[\w\s\-'\.]*)\s+(" + "|".join(re.escape(s) for s in _STREET_SUFFIXES) + r")$",
    re.I,
)

# More explicit street-name indicators (starts with a number or direction + word + suffix)
_NUMBERED_STREET = re.compile(
    r"^\d+(st|nd|rd|th)\s+(street|ave|avenue|road|drive|blvd|boulevard|place|lane|way|court)$",
    re.I,
)
_DIRECTIONAL_STREET = re.compile(
    r"^(north|south|east|west|northeast|northwest|southeast|southwest|n\.|s\.|e\.|w\.)\s+\w[\w\s]*\s+(street|ave|avenue|road|drive|blvd|boulevard|place|lane|way|court|circle|trail|parkway|terrace)$",
    re.I,
)

# Generic geographic / administrative features
_COUNTY_TOWNSHIP_PATTERN = re.compile(
    r"^[\w\s\-'\.]+\s+(county|parish|township|twp|borough|boro|municipality|precinct)$",
    re.I,
)

# Pure number or very short (1-2 chars, not a known abbreviation)
_JUST_NUMBER = re.compile(r"^\d+(\.\d+)?$")
_TOO_SHORT = re.compile(r"^.{1,2}$")

# Only directional / completely generic words
_GENERIC_ONLY = re.compile(
    r"^(north|south|east|west|northeast|northwest|southeast|southwest|the|a|an|unknown|unnamed|unnamed facility|fixme|n\/a|null|none|tbd)$",
    re.I,
)

# Known 2-letter brands/abbreviations to NOT remove even though short
_KNOWN_SHORT_BRANDS: Set[str] = {"BP", "GE", "3M", "UPS", "CSX", "CN", "CP", "NS", "UP"}


def is_highway_or_route(name: str) -> bool:
    """Return True if the name looks like a highway/route designation."""
    name = name.strip()
    for pattern in _HIGHWAY_PATTERNS:
        if pattern.match(name):
            return True
    return False


def is_street_name(name: str) -> bool:
    """Return True if name looks like a raw street/road name with no company."""
    name = name.strip()
    if _STREET_PATTERN.match(name):
        return True
    if _NUMBERED_STREET.match(name):
        return True
    if _DIRECTIONAL_STREET.match(name):
        return True
    return False


def is_county_or_township(name: str) -> bool:
    """Return True if name is just a county, parish, or township."""
    return bool(_COUNTY_TOWNSHIP_PATTERN.match(name.strip()))


def is_junk_name(name: str) -> bool:
    """Aggregate junk check."""
    n = name.strip()
    if not n:
        return True
    if _JUST_NUMBER.match(n):
        return True
    if _TOO_SHORT.match(n) and n.upper() not in _KNOWN_SHORT_BRANDS:
        return True
    if _GENERIC_ONLY.match(n):
        return True
    if is_highway_or_route(n):
        return True
    if is_street_name(n):
        return True
    if is_county_or_township(n):
        return True
    return False


# ---------------------------------------------------------------------------
# Haversine distance
# ---------------------------------------------------------------------------

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in metres between two lat/lon points."""
    R = 6_371_000.0  # Earth radius in metres
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_float(val: str) -> Optional[float]:
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def quality_rank(q: str) -> int:
    return QUALITY_RANK.get(q, 99)


def has_coords(row: dict) -> bool:
    return parse_float(row.get("lat", "")) is not None and parse_float(row.get("lon", "")) is not None


# ---------------------------------------------------------------------------
# Deduplication helpers
# ---------------------------------------------------------------------------

def exact_dedup_key(row: dict) -> Optional[Tuple[str, float, float]]:
    """Key for exact-match dedup: (lower name, 4dp lat, 4dp lon)."""
    lat = parse_float(row.get("lat", ""))
    lon = parse_float(row.get("lon", ""))
    if lat is None or lon is None:
        return None
    return (row["company_name"].strip().lower(), round(lat, 4), round(lon, 4))


def exact_dedup(rows: List[dict]) -> Tuple[List[dict], int]:
    """Remove exact duplicates; keep the one with best data_quality."""
    seen: Dict[Tuple, dict] = {}
    no_coords: List[dict] = []
    removed = 0

    for row in rows:
        key = exact_dedup_key(row)
        if key is None:
            no_coords.append(row)
            continue
        if key not in seen:
            seen[key] = row
        else:
            existing = seen[key]
            if quality_rank(row["data_quality"]) < quality_rank(existing["data_quality"]):
                seen[key] = row
            removed += 1

    return list(seen.values()) + no_coords, removed


def proximity_dedup_same_name(
    rows: List[dict], distance_m: float = 500
) -> Tuple[List[dict], int]:
    """
    Remove near-duplicates: same company_name (case-insensitive) within
    `distance_m` metres. Keep best quality; if tied, keep first encountered.
    Only applies to rows with coordinates.
    """
    # Separate rows with and without coordinates
    with_coords = [r for r in rows if has_coords(r)]
    without_coords = [r for r in rows if not has_coords(r)]

    # Group by normalised name
    by_name: Dict[str, List[dict]] = defaultdict(list)
    for row in with_coords:
        key = row["company_name"].strip().lower()
        by_name[key].append(row)

    removed_ids: Set[int] = set()

    for name, group in by_name.items():
        if len(group) < 2:
            continue
        # Sort best quality first so we always keep the best
        group_sorted = sorted(group, key=lambda r: quality_rank(r["data_quality"]))
        for i, a in enumerate(group_sorted):
            if id(a) in removed_ids:
                continue
            lat_a = parse_float(a["lat"])
            lon_a = parse_float(a["lon"])
            for b in group_sorted[i + 1 :]:
                if id(b) in removed_ids:
                    continue
                lat_b = parse_float(b["lat"])
                lon_b = parse_float(b["lon"])
                dist = haversine_m(lat_a, lon_a, lat_b, lon_b)
                if dist <= distance_m:
                    removed_ids.add(id(b))

    kept = [r for r in with_coords if id(r) not in removed_ids]
    removed = len(removed_ids)
    return kept + without_coords, removed


def fuzzy_proximity_dedup(
    rows: List[dict],
    distance_m: float = 200,
    similarity_threshold: int = 88,
) -> Tuple[List[dict], int]:
    """
    Remove near-duplicates within `distance_m` metres with fuzzy name
    similarity >= `similarity_threshold` (token_set_ratio). Keep best quality.
    Only applies to rows with coordinates.
    """
    with_coords = [r for r in rows if has_coords(r)]
    without_coords = [r for r in rows if not has_coords(r)]

    # Build a spatial grid to avoid O(n^2) comparisons across entire dataset.
    # Grid cell size ~0.002 degrees ≈ ~200m at mid-latitudes.
    CELL_DEG = 0.002

    def cell(lat: float, lon: float) -> Tuple[int, int]:
        return (int(lat / CELL_DEG), int(lon / CELL_DEG))

    grid: Dict[Tuple[int, int], List[dict]] = defaultdict(list)
    for row in with_coords:
        c = cell(parse_float(row["lat"]), parse_float(row["lon"]))
        grid[c].append(row)

    # Sort best quality first globally
    with_coords_sorted = sorted(with_coords, key=lambda r: quality_rank(r["data_quality"]))
    removed_ids: Set[int] = set()

    for i, a in enumerate(with_coords_sorted):
        if id(a) in removed_ids:
            continue
        lat_a = parse_float(a["lat"])
        lon_a = parse_float(a["lon"])
        name_a = a["company_name"].strip().lower()
        ca = cell(lat_a, lon_a)

        # Check neighbouring cells
        candidates: List[dict] = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                candidates.extend(grid[(ca[0] + dr, ca[1] + dc)])

        for b in candidates:
            if id(b) is id(a) or id(b) in removed_ids:
                continue
            # Must be worse or equal quality to be removed
            if quality_rank(b["data_quality"]) < quality_rank(a["data_quality"]):
                continue
            lat_b = parse_float(b["lat"])
            lon_b = parse_float(b["lon"])
            dist = haversine_m(lat_a, lon_a, lat_b, lon_b)
            if dist > distance_m:
                continue
            name_b = b["company_name"].strip().lower()
            if name_a == name_b:
                continue  # already handled by proximity_dedup_same_name
            score = fuzz.token_set_ratio(name_a, name_b)
            if score >= similarity_threshold:
                removed_ids.add(id(b))

    kept = [r for r in with_coords if id(r) not in removed_ids]
    removed = len(removed_ids)
    return kept + without_coords, removed


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("Rail Network Scanner — Final Dataset Builder")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_rows: List[dict] = list(reader)
        fieldnames = reader.fieldnames or []

    start_count = len(all_rows)
    print(f"\nStarting rows: {start_count:,}")

    # ------------------------------------------------------------------
    # 2. Remove low_quality_name rows
    # ------------------------------------------------------------------
    before = len(all_rows)
    all_rows = [r for r in all_rows if r["data_quality"] != "low_quality_name"]
    removed_low_quality = before - len(all_rows)
    print(f"\n[1] Removed low_quality_name rows:          {removed_low_quality:,}")
    print(f"    Remaining:                               {len(all_rows):,}")

    # ------------------------------------------------------------------
    # 3. Remove junk names (highways, streets, counties, generics)
    # ------------------------------------------------------------------
    highway_removed: List[str] = []
    street_removed: List[str] = []
    county_removed: List[str] = []
    generic_removed: List[str] = []
    kept_after_junk: List[dict] = []

    for row in all_rows:
        name = row["company_name"].strip()
        if is_highway_or_route(name):
            highway_removed.append(name)
        elif is_street_name(name):
            street_removed.append(name)
        elif is_county_or_township(name):
            county_removed.append(name)
        elif is_junk_name(name):
            # Catches numbers-only, too-short, generic-only (not highway/street/county)
            generic_removed.append(name)
        else:
            kept_after_junk.append(row)

    print(f"\n[2] Junk name removal:")
    print(f"    Highway/route names removed:             {len(highway_removed):,}")
    print(f"    Street names removed:                    {len(street_removed):,}")
    print(f"    County/township names removed:           {len(county_removed):,}")
    print(f"    Generic/short/number names removed:      {len(generic_removed):,}")
    total_junk = len(highway_removed) + len(street_removed) + len(county_removed) + len(generic_removed)
    print(f"    Total junk removed:                      {total_junk:,}")
    all_rows = kept_after_junk
    print(f"    Remaining:                               {len(all_rows):,}")

    # ------------------------------------------------------------------
    # 4. Exact deduplication (same name + rounded lat/lon to 4dp)
    # ------------------------------------------------------------------
    before = len(all_rows)
    all_rows, removed_exact = exact_dedup(all_rows)
    print(f"\n[3] Exact dedup (name + 4dp coords):        {removed_exact:,} removed")
    print(f"    Remaining:                               {len(all_rows):,}")

    # ------------------------------------------------------------------
    # 5. Proximity dedup — same company name within 500m
    # ------------------------------------------------------------------
    before = len(all_rows)
    all_rows, removed_proximity = proximity_dedup_same_name(all_rows, distance_m=500)
    print(f"\n[4] Proximity dedup same-name (<500m):       {removed_proximity:,} removed")
    print(f"    Remaining:                               {len(all_rows):,}")

    # ------------------------------------------------------------------
    # 6. Fuzzy proximity dedup — similar names within 200m
    # ------------------------------------------------------------------
    before = len(all_rows)
    all_rows, removed_fuzzy = fuzzy_proximity_dedup(
        all_rows, distance_m=200, similarity_threshold=88
    )
    print(f"\n[5] Fuzzy dedup similar-name (<200m, ≥88%): {removed_fuzzy:,} removed")
    print(f"    Remaining:                               {len(all_rows):,}")

    # ------------------------------------------------------------------
    # 7. Write output
    # ------------------------------------------------------------------
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nOutput written to: {OUTPUT_CSV}")

    # ------------------------------------------------------------------
    # 8. Comprehensive report
    # ------------------------------------------------------------------
    final_count = len(all_rows)
    unique_names = len({r["company_name"].strip().lower() for r in all_rows})

    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)
    print(f"  Starting rows:                             {start_count:,}")
    print(f"  - low_quality_name removed:                {removed_low_quality:,}")
    print(f"  - Highway/route junk:                      {len(highway_removed):,}")
    print(f"  - Street name junk:                        {len(street_removed):,}")
    print(f"  - County/township junk:                    {len(county_removed):,}")
    print(f"  - Generic/number/short junk:               {len(generic_removed):,}")
    print(f"  - Exact duplicates removed:                {removed_exact:,}")
    print(f"  - Same-name proximity dupes removed:       {removed_proximity:,}")
    print(f"  - Fuzzy proximity dupes removed:           {removed_fuzzy:,}")
    total_removed = (
        removed_low_quality + total_junk + removed_exact + removed_proximity + removed_fuzzy
    )
    print(f"  Total removed:                             {total_removed:,}")
    print(f"  Final row count:                           {final_count:,}")
    print(f"  Unique company names (normalised):         {unique_names:,}")

    print("\n--- Top 30 Company Names ---")
    name_counts = Counter(r["company_name"].strip() for r in all_rows)
    for i, (name, cnt) in enumerate(name_counts.most_common(30), 1):
        print(f"  {i:>2}. {name:<50} {cnt:>4}x")

    print("\n--- Data Quality Breakdown ---")
    dq_counts = Counter(r["data_quality"] for r in all_rows)
    for k, v in sorted(dq_counts.items()):
        pct = 100 * v / final_count if final_count else 0
        print(f"  {k:<20} {v:>6,}  ({pct:.1f}%)")

    print("\n--- Industry Sector Breakdown ---")
    sector_counts = Counter(r.get("sector_guess", "").strip() or "(none)" for r in all_rows)
    for k, v in sector_counts.most_common():
        pct = 100 * v / final_count if final_count else 0
        print(f"  {k:<30} {v:>6,}  ({pct:.1f}%)")

    print("\n--- States with Data ---")
    state_vals = [r.get("state", "").strip() or r.get("state_key", "").strip() for r in all_rows]
    # Normalise: use state_key as fallback for blank state
    def best_state(row: dict) -> str:
        s = row.get("state", "").strip()
        if s:
            return s
        return row.get("state_key", "").strip() or "(unknown)"

    state_counts = Counter(best_state(r) for r in all_rows)
    print(f"  Distinct state values: {len(state_counts)}")
    for k, v in sorted(state_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {k:<30} {v:>5,}")

    print("\n--- Source Breakdown ---")
    src_counts = Counter(r.get("source", "").strip() for r in all_rows)
    for k, v in src_counts.most_common():
        pct = 100 * v / final_count if final_count else 0
        print(f"  {k:<30} {v:>6,}  ({pct:.1f}%)")

    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
