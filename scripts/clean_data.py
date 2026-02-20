#!/usr/bin/env python3
"""
Clean and filter the merged all-states rail company dataset.

Steps:
1. Remove exact duplicates (same company_name + lat + lon).
2. Filter out non-rail businesses (gas stations, retail, restaurants, etc.)
   using exact brand matches and keyword regex, with an industrial-keyword
   override that keeps anything legitimately rail-served.
3. Tag each row with a `data_quality` label (complete / no_location /
   low_quality_name).
4. Normalise generic OSM facility_type tags ("yes", "roof", "poi").

Input:  data/output/all_states_classified.csv
Output: data/output/all_states_cleaned.csv

Usage:
    python scripts/clean_data.py
    python scripts/clean_data.py --input path/to/other.csv
    python scripts/clean_data.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Bootstrap — make sure ``config`` and ``src.*`` are importable when the
# script is executed from any working directory.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import LOG_DATE_FORMAT, LOG_FORMAT, OUTPUT_DIR

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger("clean_data")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
INPUT_CSV = OUTPUT_DIR / "all_states_classified.csv"
OUTPUT_CSV = OUTPUT_DIR / "all_states_cleaned.csv"

# ---------------------------------------------------------------------------
# Non-rail brand exclusion list (exact, case-insensitive match on
# company_name after stripping whitespace).
# ---------------------------------------------------------------------------
NON_RAIL_BRANDS: frozenset[str] = frozenset({
    # --- Fuel / convenience ---
    "shell", "bp", "exxon", "chevron", "sunoco", "marathon", "citgo",
    "phillips 66", "mobil", "cenex", "casey's general store", "casey's",
    "kwik trip", "speedway", "quiktrip", "pilot", "pilot travel center",
    "love's", "love's travel stop", "flying j", "ractrac",
    "wawa", "sheetz", "circle k", "7-eleven", "seven-eleven",
    "valero", "murphy usa", "murphy express",
    "bj's gas", "costco gas", "sam's club fuel",
    # Fuel brand variants that are retail stations (not refineries/terminals)
    "sinclair", "sinclair gas station",
    "conoco", "texaco", "gulf",
    "marathon gas", "sunoco gas station",
    "maverik", "kum & go", "pacific pride",
    "holiday", "irving",
    # --- Fast food ---
    "mcdonald's", "subway", "burger king", "wendy's", "taco bell",
    "dairy queen", "kfc", "popeyes", "chick-fil-a", "chick fil a",
    "sonic drive-in", "sonic", "arby's", "jack in the box", "hardee's",
    "carl's jr", "five guys", "in-n-out", "whataburger", "culver's",
    "jimmy john's", "jersey mike's", "potbelly", "firehouse subs",
    "panera bread", "chipotle", "qdoba", "moe's southwest grill",
    "starbucks", "dunkin'", "dunkin", "tim hortons",
    # --- Dollar / discount stores ---
    "dollar general", "dollar tree", "family dollar",
    # --- Auto parts ---
    "autozone", "o'reilly auto parts", "o'reilly", "advance auto parts",
    "napa auto parts", "napa",
    "carquest auto parts", "carquest",
    "harbor freight tools",
    # --- Big-box retail ---
    "walmart", "walmart supercenter", "target", "lowe's", "home depot",
    "the home depot", "walgreens", "cvs", "cvs pharmacy", "rite aid",
    "menards", "ace hardware", "true value",
    # --- Parcel / postal retail storefronts ---
    "usps", "united states postal service", "united states post office",
    "the ups store", "ups store",
    "fedex office", "fedex kinkos",
    # --- Propane exchange / consumer ---
    "blue rhino", "blue rhino propane exchange",
    "amerigas", "amerigas propane",
    # --- Self-storage (consumer) ---
    "public storage", "extra space storage",
    "storage rentals of america", "u-haul",
    "life storage", "cubesmart", "simply self storage",
    # --- Transit / bus (not rail freight) ---
    "valley metro",
    # --- ATM / financial kiosks ---
    "bitcoin depot", "bitcoin depot - bitcoin atm",
    # --- Consumer vehicle rental ---
    "penske truck rental",
    # --- Other consumer services ---
    "jiffy lube", "midas", "pep boys", "firestone", "goodyear",
    "monro muffler", "take 5 oil change",
})

# ---------------------------------------------------------------------------
# Industrial override keywords — if ANY of these appear in the company name,
# the row is kept even if the exclusion regex would otherwise remove it.
# ---------------------------------------------------------------------------
_INDUSTRIAL_TERMS = (
    r"warehouse|plant|terminal|yard|siding|elevator|mill|refinery|factory|"
    r"foundry|forge|smelter|mine|quarry|lumber|steel|chemical|petroleum|"
    r"power|generating|substation|grain|ethanol|fertilizer|cement|concrete|"
    r"aggregate|asphalt|pipeline|storage|distribution|logistics|freight|"
    r"intermodal|transload|bulk|scrap|recycling|paper|pulp|timber|plywood|"
    r"coal|oil|natural\s+gas|manufacturing|processing|industries|industrial|"
    r"works|metals|minerals|materials|commodities|energy|fuel\s+terminal|"
    r"fuel\s+storage|propane\s+storage|ammonia|nitrogen|phosphate|potash|"
    r"sulfur|chlorine|polymer|resin|plastic|glass|ceramic|rubber|auto\s+plant|"
    r"assembly\s+plant|stamping|casting|rail\s+car|railcar|tank\s+car|boxcar|"
    r"hopper|gondola|flatcar|switching|transfer|loading|unloading|shipping|"
    # Large-scale beverage producers that are genuine rail customers
    r"anheuser|budweiser|molson|coors\s+brewing|miller\s+brewing|"
    r"pabst|yuengling|boston\s+beer|dogfish"
)

INDUSTRIAL_RE: re.Pattern[str] = re.compile(
    rf"\b(?:{_INDUSTRIAL_TERMS})\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Non-rail keyword regex — matches names that are clearly non-industrial
# community / consumer venues.
# ---------------------------------------------------------------------------
_NON_RAIL_KEYWORD_TERMS = (
    # Places of worship / civic
    r"church|cathedral|chapel|synagogue|mosque|temple|parish|congregation|"
    r"fellowship|ministry|tabernacle|assembly\s+of\s+god|"
    r"cemetery|memorial\s+garden|"
    r"library|"
    r"museum|"
    # Schools / education
    r"elementary\s+school|elementary|middle\s+school|high\s+school|"
    r"kindergarten|daycare|day\s+care|child\s+care|preschool|"
    r"university|college|community\s+college|"
    # Parks / outdoor — only standalone "park" and explicit outdoor types
    r"(?<!\w)park(?!\s*(?:way|ing|land|side|hurst|ville|ridge|view|wood|"
    r"dale|gate|haven|shire|place|center|industrial|business|commerce|"
    r"corporate|technology|tech|science|research|trade|enterprise|"
    r"distribution|logistics|manufacturing|railroad|rail))\b|"
    r"playground|trailhead|"
    # "trail" but NOT "Rail Trail" or "Railroad Trail" (those are rail-adjacent)
    r"(?<!rail\s)(?<!railroad\s)(?<!\w)trail(?!\s*(?:rail|railroad|freight|junction|spur|siding))\b|"
    # Emergency / government services
    r"fire\s+(?:department|station|district|house)|"
    r"police\s+(?:department|station|precinct)|"
    r"sheriff|"
    # Medical / health
    r"hospital|clinic|urgent\s+care|medical\s+center|"
    r"dentist|dental|orthodontist|optometrist|"
    r"veterinary|veterinarian|animal\s+clinic|pet\s+clinic|"
    # Financial retail
    r"bank|credit\s+union|"
    # Personal care
    r"hair\s+salon|barber|barbershop|nail\s+salon|nail\s+spa|"
    r"beauty\s+salon|day\s+spa|massage|"
    # Housing
    r"apartment|apartments|condos?|townhomes?|townhouse|"
    r"mobile\s+home\s+park|trailer\s+park|"
    r"bed\s+and\s+breakfast|inn(?:\s|$)|motel|hotel|"
    # Food & beverage (consumer)
    r"restaurant|bistro|diner|eatery|cafe|coffee\s+shop|"
    r"pizza|pizzeria|burger|chicken\s+shack|chicken\s+wings|"
    r"taco|chinese\s+(?:restaurant|kitchen|garden|buffet)|"
    r"mexican\s+(?:restaurant|grill|kitchen)|"
    r"thai\s+(?:restaurant|kitchen|cuisine)|"
    r"japanese\s+(?:restaurant|steakhouse|kitchen)|"
    r"sushi|bbq\s+(?:restaurant|shack|joint)|"
    r"grill(?!\s*(?:room|manufacturing|industries))|"
    r"pub(?!\s*(?:lishing|lic\s+(?:storage|warehouse)))|"
    r"tavern|saloon|"
    r"brewery(?!\s*(?:supply|equipment|industrial))|"
    r"brewing(?!\s*(?:supply|equipment|industrial))|"
    # Fuel / car services (consumer)
    r"gas\s+station|filling\s+station|fuel\s+station|"
    r"car\s+wash|laundromat|"
    # Other retail / services
    r"dollar\s+store|thrift\s+store|"
    r"grocery(?!\s*(?:distribution|warehouse))|supermarket|"
    r"pharmacy(?!\s*(?:distribution|manufacturer))|"
    r"golf\s+course|bowling\s+alley|movie\s+theater|cinema"
)

NON_RAIL_KEYWORD_RE: re.Pattern[str] = re.compile(
    rf"(?:{_NON_RAIL_KEYWORD_TERMS})",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Low-quality name detection (mirrors analyze_data_quality.py logic).
# ---------------------------------------------------------------------------
_GEO_KEYWORDS_PAT = (
    r"\b(road|rd|street|st|avenue|ave|blvd|boulevard|highway|hwy|"
    r"lane|ln|drive|dr|court|ct|place|pl|way|route|rte|pike|pkwy|parkway|"
    r"trail|path|crossing|junction|jct|bridge|creek|river|lake|pond|bay|"
    r"island|mountain|mt|hill|valley|hollow|holler|run)\b"
)
GEO_KEYWORDS_RE: re.Pattern[str] = re.compile(_GEO_KEYWORDS_PAT, re.IGNORECASE)
COORD_OR_NUMERIC_RE: re.Pattern[str] = re.compile(r"^-?\d+(\.\d+)?(,\s*-?\d+(\.\d+)?)?$")
TRIVIAL_RE: re.Pattern[str] = re.compile(r"^\d+$|^.{0,2}$")
OSM_TAG_RE: re.Pattern[str] = re.compile(r"^(node|way|relation)\s*\d+", re.IGNORECASE)

# Facility type normalisation map
FACILITY_TYPE_NORMALISE: dict[str, str] = {
    "yes": "building",
    "roof": "building",
    "poi": "point_of_interest",
}


# ---------------------------------------------------------------------------
# Core filter functions
# ---------------------------------------------------------------------------

def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop rows where both company_name and coordinates are identical.

    Returns the deduplicated DataFrame and the count of rows removed.
    """
    before = len(df)
    # Normalise lat/lon to a fixed precision so floating-point noise does not
    # create spurious unique values.
    df = df.copy()
    df["_lat_r"] = pd.to_numeric(df["lat"], errors="coerce").round(6)
    df["_lon_r"] = pd.to_numeric(df["lon"], errors="coerce").round(6)

    df = df.drop_duplicates(subset=["company_name", "_lat_r", "_lon_r"])
    df = df.drop(columns=["_lat_r", "_lon_r"])

    removed = before - len(df)
    logger.info("Duplicates removed: %d", removed)
    return df, removed


def _is_exact_brand_match(name: str) -> bool:
    """Return True if name matches a known non-rail brand exactly."""
    return name.strip().lower() in NON_RAIL_BRANDS


def _is_non_rail_keyword(name: str) -> bool:
    """Return True if name contains non-rail keywords but NOT industrial ones."""
    if not NON_RAIL_KEYWORD_RE.search(name):
        return False
    # Industrial override: keep the row if it also contains industrial terms.
    if INDUSTRIAL_RE.search(name):
        return False
    return True


def classify_non_rail(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Remove rows that are clearly non-rail businesses.

    Returns the filtered DataFrame and a breakdown dict with counts per
    removal category.
    """
    # NaN names cannot be matched; treat them as non-excluded
    names = df["company_name"].fillna("").astype(str)

    exact_mask = names.apply(_is_exact_brand_match)
    keyword_mask = names.apply(_is_non_rail_keyword) & ~exact_mask

    breakdown: dict[str, int] = {
        "exact_brand_match": int(exact_mask.sum()),
        "keyword_match": int(keyword_mask.sum()),
    }
    total_removed = int((exact_mask | keyword_mask).sum())
    breakdown["total"] = total_removed

    keep_mask = ~(exact_mask | keyword_mask)
    df_filtered = df[keep_mask].copy()

    logger.info(
        "Non-rail removed: %d total  (exact=%d, keyword=%d)",
        total_removed,
        breakdown["exact_brand_match"],
        breakdown["keyword_match"],
    )
    return df_filtered, breakdown


def _classify_name_quality(name: object) -> str:
    """Return a quality label for a single company_name value."""
    if pd.isna(name) or str(name).strip() == "":
        return "empty_name"
    s = str(name).strip()
    if COORD_OR_NUMERIC_RE.match(s):
        return "low_quality_name"
    if TRIVIAL_RE.match(s):
        return "low_quality_name"
    if OSM_TAG_RE.match(s):
        return "low_quality_name"
    if GEO_KEYWORDS_RE.search(s):
        return "low_quality_name"
    return "real"


def add_data_quality_column(df: pd.DataFrame) -> pd.DataFrame:
    """Append a ``data_quality`` column to *df* (in-place copy).

    Values:
        complete         — has company_name + city + state
        no_location      — city or state is missing / blank
        low_quality_name — name looks like a road, coordinate, or OSM artefact
    """
    df = df.copy()

    name_quality = df["company_name"].apply(_classify_name_quality)

    has_city = df["city"].notna() & (df["city"].astype(str).str.strip() != "")
    has_state = df["state"].notna() & (df["state"].astype(str).str.strip() != "")

    conditions = [
        name_quality == "low_quality_name",
        ~(has_city & has_state),
    ]
    choices = ["low_quality_name", "no_location"]
    # numpy.select: first matching condition wins
    import numpy as np
    df["data_quality"] = np.select(conditions, choices, default="complete")

    return df


def normalise_facility_type(df: pd.DataFrame) -> pd.DataFrame:
    """Replace generic OSM facility_type tags with more descriptive values."""
    df = df.copy()
    df["facility_type"] = df["facility_type"].replace(FACILITY_TYPE_NORMALISE)
    return df


# ---------------------------------------------------------------------------
# Summary reporting
# ---------------------------------------------------------------------------

def print_summary(
    original_count: int,
    after_dedup: int,
    after_nonrail: int,
    nonrail_breakdown: dict[str, int],
    df_final: pd.DataFrame,
) -> None:
    """Print a formatted summary report to stdout."""
    sep = "=" * 70
    thin = "-" * 70

    print(f"\n{sep}")
    print("RAIL NETWORK SCANNER — DATA CLEANING SUMMARY")
    print(sep)

    print(f"\n{'Metric':<45} {'Count':>10}")
    print(thin)
    print(f"{'Original rows':<45} {original_count:>10,}")
    print(f"{'  Duplicates removed':<45} {original_count - after_dedup:>10,}")
    print(f"{'  After deduplication':<45} {after_dedup:>10,}")
    print(f"{'  Non-rail removed (total)':<45} {nonrail_breakdown['total']:>10,}")
    print(f"{'    — exact brand match':<45} {nonrail_breakdown['exact_brand_match']:>10,}")
    print(f"{'    — keyword match':<45} {nonrail_breakdown['keyword_match']:>10,}")
    print(f"{'Final cleaned rows':<45} {len(df_final):>10,}")
    print(thin)

    # Data quality breakdown
    print(f"\n{'Data Quality':<45} {'Count':>10}   {'%':>6}")
    print(thin)
    quality_counts = df_final["data_quality"].value_counts()
    total = len(df_final)
    for label in ["complete", "no_location", "low_quality_name"]:
        count = quality_counts.get(label, 0)
        pct = count / total * 100 if total else 0.0
        print(f"  {label:<43} {count:>10,}   {pct:>5.1f}%")
    print(thin)

    # Top 20 company names
    print("\nTop 20 company names in cleaned data:")
    print(thin)
    top20 = df_final["company_name"].value_counts().head(20)
    for rank, (name, count) in enumerate(top20.items(), start=1):
        print(f"  {rank:>2}. {name:<50} {count:>5,}")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def clean_dataset(
    input_path: Path = INPUT_CSV,
    output_path: Path = OUTPUT_CSV,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Run the full cleaning pipeline.

    Args:
        input_path: Path to the raw classified CSV.
        output_path: Destination for the cleaned CSV.
        dry_run:     When True, skip writing the output file.

    Returns:
        The cleaned DataFrame.
    """
    logger.info("Reading input: %s", input_path)
    df = pd.read_csv(input_path, dtype=str, low_memory=False)
    original_count = len(df)
    logger.info("Loaded %d rows, %d columns", original_count, len(df.columns))

    # Step 1: Remove duplicates
    df, _dupes_removed = remove_duplicates(df)
    after_dedup = len(df)

    # Step 2: Filter non-rail businesses
    df, nonrail_breakdown = classify_non_rail(df)
    after_nonrail = len(df)

    # Step 3: Tag data quality (no geocoding — just structural assessment)
    logger.info("Computing data quality labels...")
    df = add_data_quality_column(df)

    # Step 4: Normalise facility_type
    logger.info("Normalising facility_type tags...")
    df = normalise_facility_type(df)

    # Report
    print_summary(original_count, after_dedup, after_nonrail, nonrail_breakdown, df)

    # Write output
    if dry_run:
        logger.info("--dry-run active: output not written to disk.")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info("Cleaned CSV written: %s  (%d rows)", output_path, len(df))

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean and filter the all-states rail company dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_CSV,
        metavar="CSV",
        help=f"Path to input CSV (default: {INPUT_CSV})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_CSV,
        metavar="CSV",
        help=f"Path for cleaned output CSV (default: {OUTPUT_CSV})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run all cleaning steps but do not write the output file.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.input.exists():
        logger.error("Input file not found: %s", args.input)
        sys.exit(1)

    clean_dataset(
        input_path=args.input,
        output_path=args.output,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
