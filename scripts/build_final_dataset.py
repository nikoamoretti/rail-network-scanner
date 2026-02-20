"""
build_final_dataset.py
======================
Surgical filtering and smart deduplication of all_states_classified_v2.csv.

Strategy (in order):
  0. Non-rail business filter: remove gas stations, fast food, retail, auto
     repair shops, bare substations, and other businesses that are definitively
     NOT rail shippers — before any dedup step.
  1. Remove TRUE junk names: highways/routes, bare street addresses,
     county/township administrative names, pure numbers, 1-2 char names
     (with brand exceptions).  Industrial-keyword safeguard: any name
     containing a known industrial word is NEVER removed by this step.
  2. Smart city-level dedup: group by (normalized_name, city) and keep
     the single best row per group.
  3. Near-duplicate proximity dedup: same normalized name within 100 m
     only — i.e., physically the same building.

Python 3.9 compatible.
"""

from __future__ import annotations

import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path("/Users/nicoamoretti/nico_repo/rail-network-scanner")
INPUT_CSV = BASE_DIR / "data/output/all_states_cleaned.csv"
OUTPUT_CSV = BASE_DIR / "data/output/all_states_final.csv"

# ---------------------------------------------------------------------------
# Industrial keyword safeguard — names containing ANY of these are NEVER
# removed by the junk-name step.
#
# All keywords are matched as whole words (using \b boundaries) to avoid
# false positives like:
#   ' co'      matching inside 'county'
#   'mill'     matching inside 'Miller' or 'Mill Road'  (still OK — mill
#              road IS a street, but the highway/street checks run first
#              inside _is_junk_name, so the safeguard never fires for them)
#   'inc'      matching inside 'ince' etc.
# ---------------------------------------------------------------------------
_INDUSTRIAL_KEYWORD_RE: re.Pattern = re.compile(
    r"\b("
    r"chemicals?|plants?|refinery|refineries|terminal|warehouse|mills?|factory|"
    r"foundry|smelter|forge|elevator|ethanol|fertilizer|steel|lumber|paper|pulp|"
    r"cement|concrete|aggregate|petroleum|pipeline|storage|logistics|freight|"
    r"recycling|scrap|grain|coal|power|generating|substation|products?|"
    r"materials?|supply|industries|industry|manufacturing|processing|services?|"
    r"solutions?|systems?|technologies|technology|enterprises?|holdings?|group|"
    r"international|america|corporation|inc|llc|corp|co"
    r")\b",
    re.I,
)


def _has_industrial_keyword(name: str) -> bool:
    """Return True if name contains at least one industrial keyword (whole word)."""
    return bool(_INDUSTRIAL_KEYWORD_RE.search(name))


# ---------------------------------------------------------------------------
# Step 0 — Non-rail business filter
# ---------------------------------------------------------------------------

# Industrial/context override words: if the bare brand matches but the name
# also contains one of these, KEEP the record (it's a legitimate industrial
# facility, not a retail outlet).
_INDUSTRIAL_OVERRIDE_RE: re.Pattern = re.compile(
    r"\b("
    r"distribution\s*center|distribution\s*centre|dc\b|distrib|"
    r"terminal|plant|refinery|refineries|chemical|chemicals|"
    r"manufacturing|warehouse|logistics|freight|pipeline|"
    r"renewable|renewable\s*fuels|biodiesel|ethanol|"
    r"petrochemical|petrochemicals|plastics?\b|"
    r"processing|facility|facilities|complex|"
    r"bulk|depot|storage|transload|transloading|"
    r"geismar|olefins|butadiene"
    r")\b",
    re.I,
)

def _has_industrial_override(name: str) -> bool:
    return bool(_INDUSTRIAL_OVERRIDE_RE.search(name))


# ---------------------------------------------------------------------------
# Bare-brand exact-match sets
# Each entry is a normalized lowercase brand name.
# Match logic: strip name, lowercase, then check equality after removing
# punctuation variants (e.g. "7-eleven" -> "7-eleven").
# ---------------------------------------------------------------------------

# Gas stations / convenience — bare only
_GAS_STATION_BRANDS: Set[str] = {
    "shell", "bp", "exxon", "chevron", "sunoco", "mobil", "cenex",
    "speedway", "valero", "circle k", "7-eleven", "7 eleven", "wawa",
    "sheetz", "racetrac", "quiktrip", "kwik trip", "kwik star",
    "pilot", "murphy usa", "casey's general store", "caseys general store",
    "casey's", "kum & go", "kum and go", "love's travel stops",
    "loves travel stops", "love's", "loves", "flying j", "conoco",
    "marathon", "sunmart",
}

# Propane kiosk brands — bare only (Ferrellgas, Suburban Propane are kept)
_PROPANE_KIOSK_BRANDS: Set[str] = {
    "blue rhino", "amerigas",
}

# Retail brands — bare only
_RETAIL_BRANDS: Set[str] = {
    "dollar general", "dollar tree", "family dollar",
    "walgreens", "cvs", "rite aid",
    "walmart", "target",
    "home depot", "lowe's", "lowes",
    "costco", "sam's club", "sams club",
    "kroger", "aldi", "publix", "food lion", "safeway",
    "trader joe's", "trader joes", "whole foods",
    "best buy", "staples", "tractor supply",
}

# Fast food / restaurants — bare only
_FAST_FOOD_BRANDS: Set[str] = {
    "mcdonald's", "mcdonalds", "subway", "burger king", "wendy's", "wendys",
    "taco bell", "kfc", "chick-fil-a", "chick fil a", "arby's", "arbys",
    "sonic drive-in", "sonic", "popeyes", "popeye's",
    "jack in the box", "whataburger", "culver's", "culvers",
    "dairy queen", "dq", "dunkin", "dunkin donuts", "dunkin'",
    "starbucks", "pizza hut", "domino's", "dominos",
    "papa john's", "papa johns", "little caesars", "little caesar's",
    "denny's", "dennys", "ihop", "waffle house", "cracker barrel",
    "applebee's", "applebees", "chili's", "chilis",
    "olive garden", "panera bread", "panera", "chipotle",
    "five guys",
}

# Services — bare brand only (bank branches, postal retail)
_SERVICE_BRANDS: Set[str] = {
    "wells fargo", "chase", "bank of america", "pnc",
    "usps", "united states postal service",
    "fedex office", "ups store",
}

# Patterns for keyword-based removal (not brand-exact — applied to full name)
# Each is a compiled regex; match is case-insensitive.
_NON_RAIL_PATTERN_RES: List[re.Pattern] = [
    # Auto repair / body shop / tire
    re.compile(r"\bauto\s+repair\b", re.I),
    re.compile(r"\bbody\s+shop\b", re.I),
    re.compile(r"\bauto\s+body\b", re.I),
    re.compile(r"\btire\s+(?:and\s+)?(?:wheel|shop|center|service|store|barn|kingdom)\b", re.I),
    re.compile(r"\btire\s*&\s*(?:wheel|auto|service)\b", re.I),
    re.compile(r"\bwheel\s+alignment\b", re.I),
    re.compile(r"\boil\s+change\b", re.I),
    re.compile(r"\blube\s+(?:center|shop|express|station|quick|n\s+lube|n\s+go)\b", re.I),
    re.compile(r"\bjiffy\s+lube\b", re.I),
    re.compile(r"\bvalvoline\b", re.I),
    re.compile(r"\bquick\s+lube\b", re.I),
    re.compile(r"\bcar\s+wash\b", re.I),
    re.compile(r"\bauto\s+(?:dealer|dealership|sales|center|store|outlet|parts\s+store)\b", re.I),
    re.compile(r"\bused\s+cars?\b", re.I),
    re.compile(r"\bpre.?owned\b", re.I),
    re.compile(r"\bpaint\s*(?:&|and)\s*body\b", re.I),
    # Hair / nail / personal services
    re.compile(r"\bhair\s+salon\b", re.I),
    re.compile(r"\bbarber\s+shop\b", re.I),
    re.compile(r"\bbarbershop\b", re.I),
    re.compile(r"\bnail\s+salon\b", re.I),
    re.compile(r"\bdentist\b", re.I),
    re.compile(r"\bdental\s+(?:office|clinic|care|group|center|practice|associates)\b", re.I),
    re.compile(r"\bveterinar(?:y|ian)\b", re.I),
    re.compile(r"\bvet\s+(?:clinic|hospital|center|care|services)\b", re.I),
    # Accommodation — bare (keep if also contains warehouse/storage/industrial)
    re.compile(r"\bhotel\b", re.I),
    re.compile(r"\bmotel\b", re.I),
    re.compile(r"\binn\b", re.I),
    re.compile(r"\blodge\b", re.I),
    re.compile(r"\bsuites\b", re.I),
    re.compile(r"\bhampton\s+inn\b", re.I),
    re.compile(r"\bholiday\s+inn\b", re.I),
    re.compile(r"\bmarriott\b", re.I),
    re.compile(r"\bhilton\b", re.I),
    # Non-industrial community/civic
    re.compile(r"\bchurch\b", re.I),
    re.compile(r"\bchapel\b", re.I),
    re.compile(r"\bcathedral\b", re.I),
    re.compile(r"\bsynagogue\b", re.I),
    re.compile(r"\bmosque\b", re.I),
    re.compile(r"\btemple\b", re.I),
    re.compile(r"\blibrary\b", re.I),
    re.compile(r"\bmuseum\b", re.I),
    re.compile(r"\belementary\s+school\b", re.I),
    re.compile(r"\bmiddle\s+school\b", re.I),
    re.compile(r"\bhigh\s+school\b", re.I),
    re.compile(r"\bprimary\s+school\b", re.I),
    re.compile(r"\bfire\s+(?:station|department|dept|house)\b", re.I),
    re.compile(r"\bpolice\s+(?:station|department|dept|precinct)\b", re.I),
    re.compile(r"\bcemetery\b", re.I),
    re.compile(r"\bmemorial\s+park\b", re.I),
    re.compile(r"\bplayground\b", re.I),
]

# Substation pattern: remove if name ends in "substation" without a company
# brand preceding it.  Allowed company brands that make it legitimate.
# Also catches "#N Substation" numbered substations.
_BARE_SUBSTATION_RE: re.Pattern = re.compile(
    r"^(?:#\s*\d+\s+substation|[\w\s\-'\.]+\s+substation)\s*(?:#\s*\d+)?$",
    re.I,
)
# Company brands that validate a substation as a named asset
_SUBSTATION_COMPANY_RE: re.Pattern = re.compile(
    r"\b("
    r"georgia\s*power|southern\s*company|duke\s*energy|dominion|entergy|"
    r"xcel|ameren|evergy|dte|firstenergy|american\s*electric|aep\b|"
    r"pacific\s*gas|pg&e|pge\b|con\s*ed|national\s*grid|"
    r"alabama\s*power|mississippi\s*power|gulf\s*power|"
    r"tennessee\s*valley|tva\b|cleco|clarksville\s*light|"
    r"progress\s*energy|alliant\s*energy|westar|evergy|"
    r"ks\s*power|ks\s*energy"
    r")\b",
    re.I,
)

# Words that redeem a name that would otherwise match a non-rail pattern
# (hotels with attached warehouses, etc.)
_HOTEL_OVERRIDE_RE: re.Pattern = re.compile(
    r"\b(warehouse|storage|industrial|manufacturing|distribution|terminal)\b",
    re.I,
)


def _normalize_brand(name: str) -> str:
    """Lowercase, strip, collapse multiple spaces, remove trailing punctuation."""
    n = name.lower().strip()
    # Remove trailing punctuation
    n = re.sub(r"[^\w\s]+$", "", n)
    n = re.sub(r"\s{2,}", " ", n)
    return n.strip()


def _is_non_rail_business(name: str) -> Tuple[bool, str]:
    """
    Return (should_remove, category_label) for a company name.

    Logic:
      1. Check bare brand exact match for each brand set.
         If matched AND the name contains an industrial override word → keep.
      2. Check keyword patterns.
      3. Check substation pattern.

    Returns True (remove) only when there's no industrial override.
    """
    norm = _normalize_brand(name)

    # ------------------------------------------------------------------ #
    # Gas stations / convenience                                          #
    # ------------------------------------------------------------------ #
    if norm in _GAS_STATION_BRANDS:
        if not _has_industrial_override(name):
            return True, "gas_station"

    # ------------------------------------------------------------------ #
    # Propane kiosks                                                      #
    # ------------------------------------------------------------------ #
    if norm in _PROPANE_KIOSK_BRANDS:
        if not _has_industrial_override(name):
            return True, "propane_kiosk"

    # ------------------------------------------------------------------ #
    # Retail                                                              #
    # ------------------------------------------------------------------ #
    if norm in _RETAIL_BRANDS:
        if not _has_industrial_override(name):
            return True, "retail"

    # ------------------------------------------------------------------ #
    # Fast food / restaurants                                             #
    # ------------------------------------------------------------------ #
    if norm in _FAST_FOOD_BRANDS:
        return True, "fast_food"

    # ------------------------------------------------------------------ #
    # Services (banks, postal retail)                                     #
    # ------------------------------------------------------------------ #
    if norm in _SERVICE_BRANDS:
        return True, "services"

    # ------------------------------------------------------------------ #
    # Keyword patterns                                                    #
    # ------------------------------------------------------------------ #
    for pattern in _NON_RAIL_PATTERN_RES:
        if pattern.search(name):
            # Hotel/motel patterns have a warehouse override
            label = "keyword_pattern"
            if re.search(r"\b(hotel|motel|inn|lodge|suites|marriott|hilton|holiday\s+inn|hampton)\b", name, re.I):
                if _hotel_override_re := _HOTEL_OVERRIDE_RE:
                    if _hotel_override_re.search(name):
                        break  # Keep — it's an industrial lodging/storage
                label = "accommodation"
            return True, label

    # ------------------------------------------------------------------ #
    # Bare substation (generic names like "Calera Substation")           #
    # ------------------------------------------------------------------ #
    if _BARE_SUBSTATION_RE.match(name):
        if not _SUBSTATION_COMPANY_RE.search(name):
            if not _has_industrial_override(name):
                return True, "bare_substation"

    return False, ""


# ---------------------------------------------------------------------------
# Junk-detection patterns
# ---------------------------------------------------------------------------

# Highways / routes — the name IS a road designation, nothing more.
_HIGHWAY_PATTERNS: List[re.Pattern] = [
    # "US 60", "US-52", "U.S. 101", "US 20 Business", "US 64 ALT"
    re.compile(r"^(US|U\.S\.)\s*[-#]?\s*\d+[\w\s\-\.;]*$", re.I),
    # "I-80", "I 80", "I80", "Interstate 40"
    re.compile(r"^I\s*[-]?\s*\d+[\w\s\-\.;]*$", re.I),
    re.compile(r"^Interstate\s+\d+[\w\s\-\.;]*$", re.I),
    # "State Route 9", "SR 45", "SH 30", "State Highway 1"
    re.compile(r"^(State\s+Route|SR|SH|State\s+Highway|State\s+Road)\s*[-#]?\s*\d+[\w\s\-\.;]*$", re.I),
    # "County Road 12", "CR 100", "CR N2185"
    re.compile(r"^(County\s+Road|CR|County\s+Route|County\s+Highway)\s*[-#]?\s*[A-Z0-9][\w\.\-;]*$", re.I),
    # "FM 1960", "RM 620"
    re.compile(r"^(FM|RM|Ranch\s+(?:to\s+Market|Road)|Farm\s+(?:to\s+Market|Road))\s*[-#]?\s*\d+[\w\s\-\.;]*$", re.I),
    # "Highway 101", "Hwy 30"
    re.compile(r"^(Highway|Hwy|HWY)\s+\d+[\w\s\-\.;]*$", re.I),
    # "Route 66", "RT 9", "RTE 1"
    re.compile(r"^(Route|RT|RTE)\s+\d+[\w\s\-\.;]*$", re.I),
    # "NY 481", "GA 30", "M 35" — bare state-abbreviation + number
    re.compile(
        r"^(NY|CA|TX|OH|IL|PA|GA|FL|WI|MN|IA|KS|MO|AR|LA|MS|AL|TN|KY|IN|MI|NC|"
        r"SC|VA|WV|MD|DE|NJ|CT|MA|RI|NH|VT|ME|ND|SD|NE|CO|UT|AZ|NM|NV|ID|MT|WY|"
        r"OR|WA|AK|HI|OK|M)\s*[-]?\s*\d+[\w\s\-\.;]*$",
        re.I,
    ),
    # "Business US 20", "Business Loop I-90"
    re.compile(r"^(Business|Loop|Spur|Alt|Alternate|Business\s+Loop)\s+(US|I|Highway|Hwy|Route)\s*[-#]?\s*\d+", re.I),
    # Compound routes: "US 82;AL 14", "US 6;US 34;NE 44"
    re.compile(r"^(US|I-|SR|FM|CR|RM|Highway|Hwy)\s*\d[\w\s\-]*(?:;\s*(US|I-|SR|SH|FM|CR|RM|[A-Z]{2})\s*[-]?\s*\d[\w\s\-]*)+$", re.I),
    # "Truck Route", "Access Road", "Service Road" (bare)
    re.compile(r"^(Truck\s+Route|Access\s+Road\s*[A-Z0-9]*|Service\s+Road)\s*\d*$", re.I),
]

# Street suffixes for bare-address detection
_STREET_SUFFIX_RE = re.compile(
    r"\b(street|avenue|boulevard|drive|road|lane|way|place|court|circle|"
    r"parkway|terrace|trail|pike|turnpike)s?\b$",
    re.I,
)

# Bare street / road / highway: optional number prefix + words + suffix.
# Covers generic named thoroughfares that are not companies:
#   "123 Main Street", "North Industrial Avenue", "Old River Road",
#   "Dwight D. Eisenhower Highway", "Heartland Expressway"
_BARE_STREET_RE = re.compile(
    r"^(\d+\s+)?[\w\s\-'\.]+\s+"
    r"(street|st|avenue|ave|boulevard|blvd|road|rd|drive|dr|lane|ln|"
    r"way|place|pl|court|ct|circle|cir|trail|trl|parkway|pkwy|"
    r"terrace|ter|pike|turnpike|highway|hwy|freeway|fwy|expressway|"
    r"throughway|thruway|causeway|connector|bypass|cutoff)$",
    re.I,
)

# Numbered cross-streets: "1st Street", "42nd Avenue"
_NUMBERED_STREET_RE = re.compile(
    r"^\d+(st|nd|rd|th)\s+(street|ave|avenue|road|drive|blvd|boulevard|place|lane|way|court)$",
    re.I,
)

# County / parish / township — administrative unit with no company info
# Requires the name to END with one of these words (possibly preceded by whitespace)
_COUNTY_TOWNSHIP_RE = re.compile(
    r"^[\w\s\-'\.]+\s+(county|parish|township|twp|borough|boro|municipality|precinct)$",
    re.I,
)

# Pure number (integer or decimal)
_PURE_NUMBER_RE = re.compile(r"^\d+(\.\d+)?$")

# Completely generic / placeholder values
_GENERIC_RE = re.compile(
    r"^(north|south|east|west|northeast|northwest|southeast|southwest|"
    r"the|a|an|unknown|unnamed|unnamed\s+facility|fixme|n\s*/\s*a|null|none|tbd)$",
    re.I,
)

# 2-char (or 3-char all-digits) brand abbreviations that are real companies
_KNOWN_SHORT_BRANDS: Set[str] = {"3M", "BP", "GE", "CF", "DX", "UPS", "CSX", "CN", "CP", "NS", "UP"}


def _is_highway_or_route(name: str) -> bool:
    for pat in _HIGHWAY_PATTERNS:
        if pat.match(name):
            return True
    return False


def _is_bare_street(name: str) -> bool:
    """True if name looks like a raw street address with no company context."""
    if _NUMBERED_STREET_RE.match(name):
        return True
    if _BARE_STREET_RE.match(name):
        return True
    return False


def _is_county_or_township(name: str) -> bool:
    return bool(_COUNTY_TOWNSHIP_RE.match(name))


def _is_junk_name(name: str) -> bool:
    """
    Return True only for names that are clearly NOT a company:
    highways, bare streets, county/township labels, pure numbers, 1-2 char
    non-brands, and pure generic placeholders.

    IMPORTANT: If the name contains an industrial keyword the function
    unconditionally returns False (safeguard applied by caller, but guard
    here too for safety).
    """
    n = name.strip()
    if not n:
        return True
    if _has_industrial_keyword(n):
        return False
    if _PURE_NUMBER_RE.match(n):
        return True
    if len(n) <= 2 and n.upper() not in _KNOWN_SHORT_BRANDS:
        return True
    if _GENERIC_RE.match(n):
        return True
    if _is_highway_or_route(n):
        return True
    if _is_bare_street(n):
        return True
    if _is_county_or_township(n):
        return True
    return False


# ---------------------------------------------------------------------------
# Name normalisation for deduplication
# ---------------------------------------------------------------------------

_LEGAL_SUFFIXES_RE = re.compile(
    r"\b(incorporated|inc|llc|l\.l\.c|corp|corporation|co|company|ltd|"
    r"limited|lp|l\.p|plc|p\.l\.c|pllc)\b\.?",
    re.I,
)
_NON_WORD_RE = re.compile(r"[^\w\s]")
_MULTI_SPACE_RE = re.compile(r"\s{2,}")


def _normalize_name(name: str) -> str:
    """
    Canonical form for dedup grouping:
    lowercase, remove legal suffixes, collapse punctuation/spaces.
    """
    n = name.lower().strip()
    n = _LEGAL_SUFFIXES_RE.sub(" ", n)
    n = _NON_WORD_RE.sub(" ", n)
    n = _MULTI_SPACE_RE.sub(" ", n)
    return n.strip()


# ---------------------------------------------------------------------------
# Source / quality ranking helpers
# ---------------------------------------------------------------------------

# Lower rank = better source
_SOURCE_RANK: Dict[str, int] = {"osm_batch": 0, "overture": 1, "nominatim": 2}
# Lower rank = better data_quality
_QUALITY_RANK: Dict[str, int] = {"complete": 0, "no_location": 1, "low_quality_name": 2}


def _row_score(row: dict) -> Tuple[int, int, int]:
    """
    Lower score = better row to keep.
    Criteria (in priority order):
      1. Has city  (0 = yes, 1 = no)
      2. Has sector (0 = yes, 1 = no)
      3. Source rank
    """
    has_city = 0 if row.get("city", "").strip() else 1
    has_sector = 0 if row.get("sector_guess", "").strip() else 1
    src = _SOURCE_RANK.get(row.get("source", "").strip(), 9)
    return (has_city, has_sector, src)


# ---------------------------------------------------------------------------
# Haversine distance
# ---------------------------------------------------------------------------

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres."""
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_float(val: str) -> Optional[float]:
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _has_coords(row: dict) -> bool:
    return (
        _parse_float(row.get("lat", "")) is not None
        and _parse_float(row.get("lon", "")) is not None
    )


# ---------------------------------------------------------------------------
# Step 2 — Smart city-level deduplication
# ---------------------------------------------------------------------------

def _location_key(row: dict) -> str:
    """
    Return a string location discriminator for a row.

    Priority:
      1. City name (lowercased) — most precise.
      2. Lat/lon snapped to a ~5 km grid cell — catches rows that share a
         state but are at genuinely different physical locations.
      3. state_key only — last resort when there are no coordinates.

    Using a ~5 km grid (0.045 deg) means two facilities must be within
    roughly 5 km to be considered the "same location" when city is missing.
    That is tight enough to avoid collapsing plants in different cities
    while still collapsing multiple OSM nodes for the same facility.
    """
    city = row.get("city", "").strip().lower()
    if city:
        return city
    lat = _parse_float(row.get("lat", ""))
    lon = _parse_float(row.get("lon", ""))
    if lat is not None and lon is not None:
        # ~5 km grid cells at mid-latitudes
        CELL_DEG = 0.045
        cell_lat = int(lat / CELL_DEG)
        cell_lon = int(lon / CELL_DEG)
        return f"grid:{cell_lat},{cell_lon}"
    return row.get("state_key", "").strip().lower()


def _city_level_dedup(rows: List[dict]) -> Tuple[List[dict], int]:
    """
    Group rows by (normalized_name, location_key).
    Within each group keep the single row with the best _row_score.

    location_key is city name when available, a ~5 km lat/lon grid cell
    when city is missing but coordinates exist, or state_key as a last
    resort.  This ensures two plants of the same company that are in
    different cities — or more than ~5 km apart with no city — are kept
    as separate entries.
    """
    groups: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for row in rows:
        norm = _normalize_name(row["company_name"])
        loc = _location_key(row)
        groups[(norm, loc)].append(row)

    kept: List[dict] = []
    removed = 0
    for group in groups.values():
        if len(group) == 1:
            kept.append(group[0])
        else:
            best = min(group, key=_row_score)
            kept.append(best)
            removed += len(group) - 1

    return kept, removed


# ---------------------------------------------------------------------------
# Step 3 — Near-duplicate proximity dedup (same normalized name, ≤100 m)
# ---------------------------------------------------------------------------

def _proximity_dedup(rows: List[dict], distance_m: float = 100.0) -> Tuple[List[dict], int]:
    """
    For rows that share a normalized name AND are within `distance_m` metres
    of each other, keep the best one and discard the rest.
    Rows without coordinates are never removed by this step.
    """
    with_coords = [r for r in rows if _has_coords(r)]
    without_coords = [r for r in rows if not _has_coords(r)]

    # Group by normalized name
    by_norm: Dict[str, List[dict]] = defaultdict(list)
    for row in with_coords:
        by_norm[_normalize_name(row["company_name"])].append(row)

    removed_ids: Set[int] = set()

    for group in by_norm.values():
        if len(group) < 2:
            continue
        # Sort best score first so the first surviving row is always the best
        group_sorted = sorted(group, key=_row_score)
        for i, anchor in enumerate(group_sorted):
            if id(anchor) in removed_ids:
                continue
            lat_a = _parse_float(anchor["lat"])
            lon_a = _parse_float(anchor["lon"])
            for candidate in group_sorted[i + 1:]:
                if id(candidate) in removed_ids:
                    continue
                lat_b = _parse_float(candidate["lat"])
                lon_b = _parse_float(candidate["lon"])
                if _haversine_m(lat_a, lon_a, lat_b, lon_b) <= distance_m:
                    removed_ids.add(id(candidate))

    kept = [r for r in with_coords if id(r) not in removed_ids]
    removed = len(removed_ids)
    return kept + without_coords, removed


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _sector_counts(rows: List[dict]) -> Counter:
    return Counter(r.get("sector_guess", "").strip() or "(none)" for r in rows)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("Rail Network Scanner — Final Dataset Builder (surgical)")
    print("=" * 70)

    # ------------------------------------------------------------------ #
    # Load                                                                #
    # ------------------------------------------------------------------ #
    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_rows: List[dict] = list(reader)
        fieldnames: List[str] = list(reader.fieldnames or [])

    start_count = len(all_rows)
    print(f"\nStarting rows (from classified_v2): {start_count:,}")

    # Keep the classified rows for comparison reporting later
    classified_rows = all_rows[:]
    classified_sector_counts = _sector_counts(classified_rows)

    # ------------------------------------------------------------------ #
    # Step 0 — Non-rail business filter                                   #
    # ------------------------------------------------------------------ #
    non_rail_removed: Dict[str, List[str]] = defaultdict(list)
    after_non_rail: List[dict] = []

    for row in all_rows:
        name = row["company_name"].strip()
        should_remove, category = _is_non_rail_business(name)
        if should_remove:
            non_rail_removed[category].append(name)
        else:
            after_non_rail.append(row)

    all_rows = after_non_rail
    total_non_rail = sum(len(v) for v in non_rail_removed.values())

    print(f"\n[Step 0] Non-rail business filter:")
    category_labels = {
        "gas_station": "Gas stations / convenience",
        "propane_kiosk": "Propane kiosks (AmeriGas, Blue Rhino)",
        "retail": "Retail stores",
        "fast_food": "Fast food / restaurants",
        "services": "Services (banks, postal retail, FedEx Office)",
        "keyword_pattern": "Keyword matches (auto repair, body shop, etc.)",
        "accommodation": "Hotels / motels",
        "bare_substation": "Generic bare substations",
    }
    for cat, label in category_labels.items():
        cnt = len(non_rail_removed.get(cat, []))
        if cnt:
            print(f"  {label:<50s} {cnt:>6,}")
    print(f"  {'Total non-rail removed':<50s} {total_non_rail:>6,}")
    print(f"  Remaining:                                               {len(all_rows):>6,}")

    # ------------------------------------------------------------------ #
    # Step 1 — Remove TRUE junk names                                     #
    # ------------------------------------------------------------------ #
    highway_removed: List[str] = []
    street_removed: List[str] = []
    county_removed: List[str] = []
    generic_removed: List[str] = []
    after_junk: List[dict] = []

    for row in all_rows:
        name = row["company_name"].strip()
        # Industrial safeguard checked inside _is_junk_name, but we classify
        # for reporting here.
        if not _is_junk_name(name):
            after_junk.append(row)
        elif _is_highway_or_route(name):
            highway_removed.append(name)
        elif _is_bare_street(name):
            street_removed.append(name)
        elif _is_county_or_township(name):
            county_removed.append(name)
        else:
            generic_removed.append(name)

    junk_total = len(highway_removed) + len(street_removed) + len(county_removed) + len(generic_removed)
    all_rows = after_junk

    print(f"\n[Step 1] Junk name removal:")
    print(f"  Highway/route names removed:    {len(highway_removed):>7,}")
    print(f"  Bare street names removed:      {len(street_removed):>7,}")
    print(f"  County/township names removed:  {len(county_removed):>7,}")
    print(f"  Generic/short/number removed:   {len(generic_removed):>7,}")
    print(f"  Total junk removed:             {junk_total:>7,}")
    print(f"  Remaining:                      {len(all_rows):>7,}")

    # ------------------------------------------------------------------ #
    # Step 2 — Smart city-level deduplication                             #
    # ------------------------------------------------------------------ #
    all_rows, removed_city_dedup = _city_level_dedup(all_rows)

    print(f"\n[Step 2] City-level dedup (same name + same city/state):")
    print(f"  Duplicate rows removed:         {removed_city_dedup:>7,}")
    print(f"  Remaining:                      {len(all_rows):>7,}")

    # ------------------------------------------------------------------ #
    # Step 3 — Proximity dedup (same normalized name, ≤100 m)             #
    # ------------------------------------------------------------------ #
    all_rows, removed_proximity = _proximity_dedup(all_rows, distance_m=100.0)

    print(f"\n[Step 3] Proximity dedup (same name, ≤100 m):")
    print(f"  Near-duplicate rows removed:    {removed_proximity:>7,}")
    print(f"  Remaining:                      {len(all_rows):>7,}")

    # ------------------------------------------------------------------ #
    # Write output                                                        #
    # ------------------------------------------------------------------ #
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nOutput written to: {OUTPUT_CSV}")

    # ------------------------------------------------------------------ #
    # Comprehensive report                                                #
    # ------------------------------------------------------------------ #
    final_count = len(all_rows)
    total_removed = start_count - final_count

    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)
    print(f"  Starting rows:                  {start_count:>7,}")
    print(f"  - Non-rail businesses removed:  {total_non_rail:>7,}")
    for cat, label in category_labels.items():
        cnt = len(non_rail_removed.get(cat, []))
        if cnt:
            print(f"    ({label}): {cnt:,}")
    print(f"  - Junk names removed:           {junk_total:>7,}")
    print(f"    (highway/route):              {len(highway_removed):>7,}")
    print(f"    (bare street):                {len(street_removed):>7,}")
    print(f"    (county/township):            {len(county_removed):>7,}")
    print(f"    (generic/short/number):       {len(generic_removed):>7,}")
    print(f"  - City-level dupes removed:     {removed_city_dedup:>7,}")
    print(f"  - Proximity dupes removed:      {removed_proximity:>7,}")
    print(f"  Total removed:                  {total_removed:>7,}")
    print(f"  Final row count:                {final_count:>7,}")
    print(f"  Unique company names (norm):    {len({_normalize_name(r['company_name']) for r in all_rows}):>7,}")

    print("\n--- Top 30 Company Names (by frequency) ---")
    name_counts = Counter(r["company_name"].strip() for r in all_rows)
    for i, (name, cnt) in enumerate(name_counts.most_common(30), 1):
        print(f"  {i:>2}. {name:<50s} {cnt:>4}x")

    # Sector comparison: classified_v2 vs final
    final_sector_counts = _sector_counts(all_rows)
    all_sectors = sorted(
        set(classified_sector_counts) | set(final_sector_counts),
        key=lambda s: -classified_sector_counts.get(s, 0),
    )

    print("\n--- Sector Counts: classified_v2 vs final ---")
    print(f"  {'Sector':<35s} {'Classified':>10s}  {'Final':>8s}  {'Removed':>8s}")
    print("  " + "-" * 65)
    for sector in all_sectors:
        c = classified_sector_counts.get(sector, 0)
        f = final_sector_counts.get(sector, 0)
        diff = c - f
        print(f"  {sector:<35s} {c:>10,}  {f:>8,}  {diff:>8,}")

    # % unclassified
    unclassified = sum(1 for r in all_rows if not r.get("sector_guess", "").strip()
                       or r.get("sector_guess", "").strip().lower() in ("unknown", "nan", "none"))
    pct_unknown = 100 * unclassified / final_count if final_count else 0
    print(f"\n  Unclassified (Unknown/blank) in final: {unclassified:,}  ({pct_unknown:.1f}%)")

    # ------------------------------------------------------------------ #
    # Spot checks by sector                                               #
    # ------------------------------------------------------------------ #
    def _spot_check(sector_name: str, n: int = 10) -> None:
        rows = [r for r in all_rows if r.get("sector_guess", "").strip() == sector_name]
        print(f"\n--- Spot Check: '{sector_name}' ({len(rows):,} total, showing {min(n, len(rows))}) ---")
        if not rows:
            print("  (no rows)")
            return
        # Prefer variety: de-dup normalized names
        seen: Set[str] = set()
        shown = 0
        for r in rows:
            key = _normalize_name(r["company_name"])
            if key in seen:
                continue
            seen.add(key)
            name = r["company_name"]
            city = r.get("city", "") or "(no city)"
            state = r.get("state", "") or r.get("state_key", "")
            print(f"  {name:<50s} | {city:<20s} {state}")
            shown += 1
            if shown >= n:
                break

    _spot_check("Automotive Manufacturing", 10)
    _spot_check("Energy", 10)
    _spot_check("Utilities", 10)

    print("\n--- Data Quality Breakdown (final) ---")
    dq_counts = Counter(r["data_quality"] for r in all_rows)
    for k, v in sorted(dq_counts.items()):
        pct = 100 * v / final_count if final_count else 0
        print(f"  {k:<25s} {v:>7,}  ({pct:.1f}%)")

    print("\n--- Source Breakdown (final) ---")
    src_counts = Counter(r.get("source", "").strip() for r in all_rows)
    for k, v in src_counts.most_common():
        pct = 100 * v / final_count if final_count else 0
        print(f"  {k:<25s} {v:>7,}  ({pct:.1f}%)")

    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
