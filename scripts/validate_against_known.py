#!/usr/bin/env python3
"""
Validate scanner results against known rail-served accounts.

Compares discovered companies with the 347 known accounts in
high_rail_accounts.csv to measure discovery accuracy.
"""

import csv
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import KNOWN_ACCOUNTS_PATH, LOG_DATE_FORMAT, LOG_FORMAT, OUTPUT_DIR
from src.enrichment import normalize_company_name

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger("validate")


def load_discovered(region_key: str) -> list[dict]:
    """Load discovered companies from output CSV."""
    path = OUTPUT_DIR / f"{region_key}_rail_companies.csv"
    if not path.exists():
        logger.error(f"Results file not found: {path}")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_known() -> list[dict]:
    """Load known rail accounts."""
    if not KNOWN_ACCOUNTS_PATH.exists():
        logger.error(f"Known accounts not found: {KNOWN_ACCOUNTS_PATH}")
        sys.exit(1)

    with open(KNOWN_ACCOUNTS_PATH, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fuzzy_match(name1: str, name2: str) -> bool:
    """Check if two company names match (normalized substring check)."""
    n1 = normalize_company_name(name1)
    n2 = normalize_company_name(name2)
    return n1 == n2 or n1 in n2 or n2 in n1


def main() -> None:
    region_key = sys.argv[1] if len(sys.argv) > 1 else "illinois"

    discovered = load_discovered(region_key)
    known = load_known()

    logger.info(f"Discovered companies: {len(discovered)}")
    logger.info(f"Known rail accounts:  {len(known)}")

    # Find matches
    discovered_names = {normalize_company_name(d["company_name"]) for d in discovered}
    known_names = {normalize_company_name(k["Company"]) for k in known}

    # Exact matches
    exact_matches = discovered_names & known_names

    # Fuzzy matches (substring)
    fuzzy_matches = set()
    for d_name in discovered_names:
        for k_name in known_names:
            if d_name != k_name and (d_name in k_name or k_name in d_name):
                fuzzy_matches.add((d_name, k_name))

    total_matched = len(exact_matches) + len(fuzzy_matches)

    # New discoveries (not in known list)
    matched_discovered = set()
    for d_name in discovered_names:
        for k_name in known_names:
            if fuzzy_match(d_name, k_name):
                matched_discovered.add(d_name)

    new_discoveries = discovered_names - matched_discovered

    # Known accounts NOT found by scanner
    matched_known = set()
    for k_name in known_names:
        for d_name in discovered_names:
            if fuzzy_match(d_name, k_name):
                matched_known.add(k_name)

    missed_known = known_names - matched_known

    # Confidence breakdown for discovered
    confidence_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for d in discovered:
        conf = d.get("confidence", "LOW")
        confidence_counts[conf] = confidence_counts.get(conf, 0) + 1

    # Print report
    print(f"\n{'=' * 60}")
    print(f"VALIDATION REPORT — {region_key.upper()}")
    print(f"{'=' * 60}")
    print(f"\nDiscovered companies:     {len(discovered)}")
    print(f"  Unique names:           {len(discovered_names)}")
    print(f"  HIGH confidence:        {confidence_counts['HIGH']}")
    print(f"  MEDIUM confidence:      {confidence_counts['MEDIUM']}")
    print(f"  LOW confidence:         {confidence_counts['LOW']}")
    print(f"\nKnown rail accounts:      {len(known)}")
    print(f"  Exact matches:          {len(exact_matches)}")
    print(f"  Fuzzy matches:          {len(fuzzy_matches)}")
    print(f"  Total matched:          {len(matched_discovered)}")
    print(f"  Known accounts missed:  {len(missed_known)}")
    print(f"\nNEW DISCOVERIES:          {len(new_discoveries)}")
    print(f"  (Not in known 347 list)")

    if new_discoveries:
        print(f"\nTop new discoveries:")
        # Show new discoveries sorted by confidence
        new_high = [d for d in discovered
                    if normalize_company_name(d["company_name"]) in new_discoveries
                    and d.get("confidence") == "HIGH"]
        for d in new_high[:20]:
            print(f"  {d['company_name']:<40} {d.get('city', '')}, {d.get('state', '')}  [{d.get('sector_guess', '')}]")

    if exact_matches:
        print(f"\nConfirmed matches (sample):")
        for name in sorted(exact_matches)[:10]:
            print(f"  {name}")

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
