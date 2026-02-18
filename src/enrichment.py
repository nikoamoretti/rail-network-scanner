"""
Phase 5: Enrich, deduplicate, and score discovered companies.

Cross-references against known accounts, deduplicates by name and location,
and assigns final confidence scores.
"""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from typing import Dict, List, Set, Tuple
from pathlib import Path

import pandas as pd

from config import COMPANIES_DIR, KNOWN_ACCOUNTS_PATH, OUTPUT_DIR
from src.utils.geo_utils import haversine_distance

# IdentifiedCompany can come from either module (Google Places or free OSM)
# Both define the same fields. We use duck typing here.
try:
    from src.company_identifier import IdentifiedCompany
except ImportError:
    from src.osm_company_identifier import IdentifiedCompany

logger = logging.getLogger(__name__)


def load_known_accounts(path: Path = KNOWN_ACCOUNTS_PATH) -> set[str]:
    """
    Load known rail-served company names from the validation CSV.

    Returns:
        Set of lowercase company names.
    """
    if not path.exists():
        logger.warning(f"Known accounts file not found: {path}")
        return set()

    names = set()
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("Company", "").strip().lower()
            if name:
                names.add(name)

    logger.info(f"Loaded {len(names)} known accounts")
    return names


def normalize_company_name(name: str) -> str:
    """Normalize a company name for deduplication."""
    name = name.lower().strip()
    # Remove common suffixes
    for suffix in [" inc", " inc.", " llc", " llc.", " corp", " corp.",
                   " co", " co.", " ltd", " ltd.", " company", " corporation",
                   " group", " industries", " international"]:
        if name.endswith(suffix):
            name = name[:-len(suffix)].strip()
    return name


def flag_known_accounts(
    companies: list[IdentifiedCompany],
    known_names: set[str],
) -> list[IdentifiedCompany]:
    """
    Flag companies that exist in the known accounts list.

    Uses fuzzy matching: checks if the known name is contained in
    the discovered name or vice versa.
    """
    flagged = 0
    for company in companies:
        normalized = normalize_company_name(company.company_name)
        # Exact match
        if normalized in known_names:
            company.in_existing_list = True
            flagged += 1
            continue
        # Substring match
        for known in known_names:
            if known in normalized or normalized in known:
                company.in_existing_list = True
                flagged += 1
                break

    logger.info(f"Flagged {flagged}/{len(companies)} as matching known accounts")
    return companies


def deduplicate_companies(
    companies: list[IdentifiedCompany],
    name_distance_m: float = 200.0,
) -> list[IdentifiedCompany]:
    """
    Deduplicate companies by name and location.

    Groups companies with the same normalized name that are within
    name_distance_m of each other, keeping the highest-confidence one.

    Args:
        companies: List of IdentifiedCompany objects.
        name_distance_m: Max distance in meters for same-name dedup.

    Returns:
        Deduplicated list.
    """
    confidence_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}

    # Group by normalized name
    name_groups: dict[str, list[IdentifiedCompany]] = defaultdict(list)
    for c in companies:
        name_groups[normalize_company_name(c.company_name)].append(c)

    deduped: list[IdentifiedCompany] = []

    for name, group in name_groups.items():
        if len(group) == 1:
            deduped.append(group[0])
            continue

        # Sort by confidence then spur length
        group.sort(
            key=lambda c: (confidence_rank.get(c.confidence, 3), -c.spur_length_m)
        )

        kept: list[IdentifiedCompany] = []
        for c in group:
            is_duplicate = False
            for existing in kept:
                dist = haversine_distance(
                    (c.lat, c.lon), (existing.lat, existing.lon)
                )
                if dist < name_distance_m:
                    is_duplicate = True
                    break
            if not is_duplicate:
                kept.append(c)

        deduped.extend(kept)

    removed = len(companies) - len(deduped)
    logger.info(f"Deduplicated: {len(companies)} → {len(deduped)} ({removed} removed)")
    return deduped


def companies_to_dataframe(companies: list[IdentifiedCompany]) -> pd.DataFrame:
    """Convert list of IdentifiedCompany to a pandas DataFrame."""
    records = []
    for c in companies:
        record = {
            "company_name": c.company_name,
            "address": c.address,
            "city": c.city,
            "state": c.state,
            "zip": c.zip_code,
            "lat": c.lat,
            "lon": c.lon,
            "facility_type": c.facility_type,
            "google_place_id": c.google_place_id,
            "osm_way_id": c.osm_way_id,
            "spur_length_m": c.spur_length_m,
            "connected_to_mainline": c.connected_to_mainline,
            "confidence": c.confidence,
            "discovery_date": c.discovery_date,
            "in_existing_list": c.in_existing_list,
            "sector_guess": c.sector_guess,
        }
        # Include source if available (osm, nominatim, google_places)
        if hasattr(c, "source"):
            record["source"] = c.source
        records.append(record)
    return pd.DataFrame(records)


def save_results(
    companies: list[IdentifiedCompany],
    region_key: str,
) -> tuple[Path, Path]:
    """
    Save enriched results to CSV and JSON.

    Returns:
        Tuple of (csv_path, json_path).
    """
    df = companies_to_dataframe(companies)

    csv_path = OUTPUT_DIR / f"{region_key}_rail_companies.csv"
    json_path = OUTPUT_DIR / f"{region_key}_rail_companies.json"

    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)

    logger.info(f"Saved {len(df)} companies to {csv_path}")
    return csv_path, json_path


def print_summary(companies: list[IdentifiedCompany], region_key: str) -> None:
    """Print a summary of discovery results."""
    total = len(companies)
    if total == 0:
        logger.info("No companies discovered.")
        return

    high = sum(1 for c in companies if c.confidence == "HIGH")
    medium = sum(1 for c in companies if c.confidence == "MEDIUM")
    low = sum(1 for c in companies if c.confidence == "LOW")
    known = sum(1 for c in companies if c.in_existing_list)
    new = total - known

    sectors = defaultdict(int)
    for c in companies:
        sector = c.sector_guess or "Unknown"
        sectors[sector] += 1

    print(f"\n{'=' * 60}")
    print(f"RAIL NETWORK SCANNER — {region_key.upper()} RESULTS")
    print(f"{'=' * 60}")
    print(f"Total companies discovered:  {total}")
    print(f"  HIGH confidence:           {high}")
    print(f"  MEDIUM confidence:         {medium}")
    print(f"  LOW confidence:            {low}")
    print(f"\nKnown accounts matched:      {known}")
    print(f"NEW discoveries:             {new}")
    print(f"\nSector breakdown:")
    for sector, count in sorted(sectors.items(), key=lambda x: -x[1]):
        print(f"  {sector:<25} {count}")
    print(f"{'=' * 60}\n")
