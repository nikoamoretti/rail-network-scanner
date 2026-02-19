"""
Multi-layer industry classifier for rail-served companies.

Applies three classification layers in order, each processing only rows
that are still marked as Unknown after the previous layer:

  Layer 1 — Expanded keyword matching (name + facility_type)
  Layer 2 — Overture category mapping
  Layer 3 — Claude Haiku batch LLM classification (requires API key)

Typical usage::

    from src.industry_classifier import classify_industries

    classify_industries(
        csv_path="data/output/all_states_enhanced_rail_companies.csv",
        output_path="data/output/classified_rail_companies.csv",
        api_key="sk-ant-...",   # optional; omit to skip Layer 3
    )
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Layer 1 — Expanded keyword map
# ---------------------------------------------------------------------------

# Each value is a list of lowercase substrings.  The first sector whose list
# contains a match wins.  Order matters: more specific sectors are listed
# before generic catch-alls such as Manufacturing/Logistics.
EXPANDED_SECTOR_MAP: Dict[str, List[str]] = {
    "Rail Services": [
        "railroad", "railway", "rail yard", "railyard", "locomotive",
        "railcar", "rail car", "switching", "short line", "shortline",
        "bnsf", "union pacific", "csx", "norfolk southern", "ns rail",
        "canadian national", "canadian pacific", "cn rail", "cp rail",
        "kansas city southern", "kcs rail", "amtrak", "conrail",
        "wheeling lake erie", "iowa interstate", "belt railway",
        "terminal railroad", "chicago terminal", "indiana harbor",
    ],
    "Agriculture": [
        "grain", "elevator", "feed mill", "feed store", "seed",
        "farm", "farming", "coop", "co-op", "cooperative",
        "ethanol", "fertilizer", "agri", "silo", "crop",
        "corn", "wheat", "soybean", "soybeans", "cotton",
        "livestock", "poultry", "cattle", "hog", "swine",
        "dairy", "ranch", "harvest", "irrigation",
        "adm", "archer daniels", "cargill", "bunge", "chs inc",
        "land o lakes", "landolakes", "growmark", "ag processing",
        "agp", "gavilon", "zen-noh", "louis dreyfus",
        "midwest grain", "heartland grain", "prairie grain",
        "farmers coop", "farmers cooperative", "rural coop",
        "agway", "cenex", "pioneer seed", "dekalb", "asgrow",
        "farm bureau", "farmland",
    ],
    "Chemicals": [
        "chemical", "chem ", "chemicals", "polymer", "plastic",
        "resin", "chlor", "caustic", "solvent", "acid", "alkali",
        "ammonia", "nitrogen", "sulfur", "sulphur", "chlorine",
        "peroxide", "glycol", "methanol", "ethylene", "propylene",
        "pharma", "pharmaceutical", "drug", "biochem",
        "dow chemical", "dow inc", "basf", "dupont", "eastman",
        "huntsman", "olin corp", "westlake", "formosa plastics",
        "celanese", "ashland", "univar", "brenntag", "nexeo",
        "praxair", "air products", "air liquide", "linde",
        "ineos", "lanxess", "solvay", "arkema",
    ],
    "Energy": [
        "energy", "oil", "gas", "petroleum", "fuel", "propane",
        "pipeline", "power plant", "electric", "utility",
        "solar", "wind farm", "nuclear", "turbine", "generator",
        "refinery", "crude", "diesel", "gasoline", "lng",
        "lpg", "biofuel", "biodiesel", "ethanol plant",
        "coal plant", "natural gas", "power generation",
        "substation", "transmission", "electric co",
        "exxon", "mobil", "chevron", "bp ", "shell oil",
        "valero", "marathon", "phillips 66", "conocophillips",
        "tesoro", "holly frontier", "pbf energy", "motiva",
        "flint hills", "calumet", "delek", "par pacific",
        "plains all american", "enterprise products",
        "targa", "kinder morgan", "energy transfer",
        "sunoco", "citgo", "getty", "sinclair",
    ],
    "Metals": [
        "steel", "metal", "iron", "aluminum", "aluminium",
        "foundry", "forge", "forging", "scrap metal", "scrap yard",
        "copper", "zinc", "nickel", "alloy", "smelter",
        "rolling mill", "fabricat", "weld", "casting",
        "galvaniz", "tinplate", "wire mill", "pipe mill",
        "nucor", "arcelormittal", "us steel", "steel dynamics",
        "ak steel", "ssab", "gerdau", "commercial metals",
        "metals usa", "reliance steel", "worthington",
        "aleris", "novelis", "kaiser aluminum", "constellium",
        "haynes", "carpenter technology", "allegheny tech",
        "olympic steel", "service center", "metals service",
    ],
    "Building Materials": [
        "lumber yard", "concrete", "cement", "aggregate",
        "sand", "gravel", "stone", "brick", "asphalt",
        "roofing", "drywall", "gypsum", "glass", "tile",
        "insulation", "wallboard", "plasterboard",
        "ready mix", "readymix", "block plant", "block co",
        "paver", "masonry", "stucco",
        "vulcan", "martin marietta", "lafargeholcim",
        "holcim", "heidelberg", "cemex", "us concrete",
        "oldcastle", "crhamerica", "boral", "armstrong",
        "owens corning", "johns manville", "knauf",
        "usg", "national gypsum", "american gypsum",
        "apogee", "guardian glass", "ppg glass",
        "pratt & lambert", "sherwin", "quikrete",
    ],
    "Forest Products": [
        "paper", "pulp", "paperboard", "tissue", "newsprint",
        "lumber", "plywood", "veneer", "osb", "mdf",
        "sawmill", "log yard", "logging", "forestry",
        "cardboard", "corrugated", "packaging board",
        "weyerhaeuser", "georgia-pacific", "georgia pacific",
        "potlatch", "plum creek", "resolute forest",
        "domtar", "clearwater paper", "boise cascade",
        "packaging corp", "pca ", "rock-tenn", "rocktenn",
        "westrock", "smurfit", "temple-inland",
        "longview fibre", "sappi", "finch paper",
        "norampac", "cascades", "graphic packaging",
    ],
    "Food & Beverage": [
        "food", "beverage", "brewery", "brewing",
        "sugar", "flour", "mill", "meat", "packing",
        "slaughter", "bakery", "baking", "snack",
        "cereal", "frozen food", "canning", "cannery",
        "bottling", "distillery", "distill", "winery",
        "wine", "spirits", "malt", "maltings",
        "coca-cola", "coke ", "pepsi", "dr pepper",
        "kraft", "nestle", "tyson", "jbs ", "cargill beef",
        "swift", "conagra", "general mills", "kellogg",
        "post holdings", "campbell", "heinz", "unilever",
        "dole", "del monte", "birds eye", "green giant",
        "pilgrim", "perdue", "wayne farms",
        "smithfield", "hormel", "oscar mayer",
        "bob evans", "pierre foods", "advance foods",
        "ready pac", "fresh express", "driscolls",
        "miller brewing", "anheuser", "inbev", "molson",
        "coors", "boston beer", "craft brew",
        "archer daniels midland milling",
    ],
    "Mining": [
        "mine", "mining", "mineral", "coal",
        "quarry", "ore", "potash", "phosphate",
        "clay", "talc", "lithium", "bauxite",
        "silica", "kaolin", "bentonite", "diatomite",
        "titanium", "vanadium", "manganese", "chromite",
        "gold mine", "silver mine", "copper mine",
        "iron ore", "taconite", "magnetite",
        "freeport", "peabody", "arch coal", "consol",
        "alpha natural", "cloud peak", "foresight",
        "mosaic", "cf industries", "icl ", "intrepid potash",
        "us silica", "fairmount santrol", "hi-crush",
        "martin marietta magnesia", "imerys", "minerals tech",
    ],
    # Government is checked before Logistics so that "army depot",
    # "naval depot", etc. resolve to Government rather than Logistics.
    "Government": [
        "army", "navy", "air force", "marine corps",
        "military", "national guard", "pentagon",
        "department of defense", "dod ",
        "dept of defense", "naval base", "army base",
        "arsenal", "ordnance", "ammunition",
        "federal ", "u.s. government", "us government",
        "state of ", "county of ", "city of ",
        "municipal", "port authority",
        "army corps", "corps of engineers",
        "bureau of reclamation", "tva ",
        "tennessee valley",
    ],
    "Logistics": [
        "logistics", "logistic", "warehouse",
        "distribution", "distrib center",
        "freight", "transport", "terminal",
        "intermodal", "storage",
        "trucking", "shipping", "dock", "port",
        "container", "transload", "transloading",
        "cross dock", "crossdock", "fulfillment",
        "supply chain", "third party", "3pl",
        "fedex", "ups ", "xpo", "j.b. hunt", "jb hunt",
        "schneider", "werner", "swift transport",
        "knight transport", "landstar",
        "echo global", "coyote logistics", "total quality",
        "ruan", "penske logistics", "ryder",
        "amazon fulfillment", "walmart dc",
        "target dc", "home depot dc",
    ],
    "Automotive": [
        "auto", "automotive", "vehicle", "automobile",
        "car ", "truck assembly", "parts plant",
        "body stamping", "stamping plant",
        "toyota", "ford ", "ford motor", "gm ",
        "general motors", "chrysler", "stellantis",
        "fca ", "dodge", "jeep assembly",
        "honda", "nissan", "hyundai", "kia",
        "bmw", "mercedes", "volkswagen", "subaru",
        "volvo", "caterpillar", "deere", "john deere",
        "cummins", "delphi", "lear corp", "magna",
        "dana inc", "modine", "gentex", "dorman",
        "motorcraft", "federal mogul", "tenneco",
    ],
    "Water/Utilities": [
        "water", "wastewater", "sewage",
        "treatment plant", "purification",
        "water district", "water authority",
        "water utility", "water works", "waterworks",
        "water co", "water company",
        "american water", "veolia water", "suez water",
        "aqua america", "cal water", "essential utilities",
        "consolidated water", "middlesex water",
        "york water", "artesian water",
        "electric utility", "gas utility", "public utility",
        "power company", "electric company",
        "municipal utility", "rural electric",
        "cooperative utility",
    ],
    "Waste/Recycling": [
        "waste", "recycling", "recycle", "scrap",
        "salvage", "landfill", "disposal",
        "environmental services", "transfer station",
        "material recovery", "sorting facility",
        "compost", "composting",
        "waste management", "republic services",
        "clean harbors", "us ecology", "us ecology",
        "clean earth", "envirostar", "stericycle",
        "casella waste", "advanced disposal",
    ],
    "Manufacturing": [
        "manufacturing", "manufacture",
        "fabrication", "fabricating",
        "production", "assembly plant",
        "industrial park", "industrial complex",
        "works", "plant",
        "mill", "forge", "casting", "stamping",
        "precision parts", "machine shop",
        "contract manufacturing",
        "industrial",
    ],
}


def classify_by_keywords(name: str, facility_type: str = "") -> str:
    """
    Classify a company into a sector using expanded keyword matching.

    Checks both ``name`` and ``facility_type`` (concatenated) against
    ``EXPANDED_SECTOR_MAP``.  Returns the first matching sector or an
    empty string if nothing matches.

    Args:
        name: Company name.
        facility_type: Facility/category type string (e.g. "grain_elevator").

    Returns:
        Sector string such as "Agriculture", or "" if no match found.
    """
    haystack = f"{name} {facility_type}".lower()
    for sector, keywords in EXPANDED_SECTOR_MAP.items():
        if any(kw in haystack for kw in keywords):
            return sector
    return ""


# ---------------------------------------------------------------------------
# Layer 2 — Overture category map
# ---------------------------------------------------------------------------

# Maps Overture Maps category strings to rail-freight industry sectors.
# Covers all categories defined in src/overture_matcher.py RAIL_CATEGORIES
# plus common Overture place categories encountered in practice.
OVERTURE_CATEGORY_MAP: Dict[str, str] = {
    # Manufacturing / industrial
    "manufacturing_facility": "Manufacturing",
    "factory": "Manufacturing",
    "industrial_facility": "Manufacturing",
    "industrial_estate": "Manufacturing",
    "workshop": "Manufacturing",
    "fabrication": "Metals",
    # Warehousing / logistics
    "warehouse": "Logistics",
    "distribution_center": "Logistics",
    "freight_terminal": "Logistics",
    "logistics_facility": "Logistics",
    "storage_facility": "Logistics",
    "loading_dock": "Logistics",
    "intermodal_terminal": "Logistics",
    "cargo_terminal": "Logistics",
    # Energy / mining
    "mine": "Mining",
    "quarry": "Mining",
    "stone_quarry": "Mining",
    "gravel_pit": "Mining",
    "coal_plant": "Energy",
    "oil_and_gas": "Energy",
    "refinery": "Energy",
    "power_plant": "Energy",
    "natural_gas_facility": "Energy",
    "petroleum_facility": "Energy",
    "fuel_depot": "Energy",
    "propane_supplier": "Energy",
    # Agriculture
    "grain_elevator": "Agriculture",
    "feed_mill": "Agriculture",
    "silo": "Agriculture",
    "farm": "Agriculture",
    "agricultural_facility": "Agriculture",
    "fertilizer_plant": "Agriculture",
    "ethanol_plant": "Agriculture",
    "dairy_facility": "Agriculture",
    "sugar_mill": "Food & Beverage",
    "flour_mill": "Food & Beverage",
    "food_processing": "Food & Beverage",
    # Building materials
    "lumber_yard": "Forest Products",
    "cement_plant": "Building Materials",
    "concrete_plant": "Building Materials",
    "aggregate_supplier": "Building Materials",
    "building_supply": "Building Materials",
    # Metals / recycling
    "steel_mill": "Metals",
    "metal_fabrication": "Metals",
    "foundry": "Metals",
    "scrap_metal": "Waste/Recycling",
    "recycling_facility": "Waste/Recycling",
    # Chemicals
    "chemical_plant": "Chemicals",
    "chemical_facility": "Chemicals",
    # Wholesale / trade
    "wholesale": "Logistics",
    "wholesale_store": "Logistics",
    "hardware_store": "Building Materials",
    "trade_supplier": "Logistics",
    # Utilities / water
    "water_treatment": "Water/Utilities",
    "wastewater_treatment": "Water/Utilities",
    "pumping_station": "Water/Utilities",
    "water_works": "Water/Utilities",
    "electric_utility": "Water/Utilities",
    "substation": "Water/Utilities",
    # Automotive
    "auto_parts": "Automotive",
    "car_dealership": "Automotive",
    "car_manufacturing": "Automotive",
    "auto_manufacturing": "Automotive",
    # Rail
    "railway_station": "Rail Services",
    "train_station": "Rail Services",
    "rail_yard": "Rail Services",
}


def classify_by_overture_category(facility_type: str) -> str:
    """
    Map an Overture category string to a rail-freight sector.

    Normalises the input to lowercase and strips surrounding whitespace
    before looking it up in ``OVERTURE_CATEGORY_MAP``.

    Args:
        facility_type: Overture category string (e.g. "grain_elevator").

    Returns:
        Sector string such as "Agriculture", or "" if not found.
    """
    if not facility_type:
        return ""
    normalised = facility_type.strip().lower().replace(" ", "_")
    return OVERTURE_CATEGORY_MAP.get(normalised, "")


# ---------------------------------------------------------------------------
# Layer 3 — Claude Haiku batch LLM classification
# ---------------------------------------------------------------------------

# Valid sector labels the LLM must choose from
LLM_SECTORS = [
    "Agriculture",
    "Chemicals",
    "Energy",
    "Metals",
    "Building Materials",
    "Forest Products",
    "Food & Beverage",
    "Mining",
    "Logistics",
    "Automotive",
    "Manufacturing",
    "Water/Utilities",
    "Waste/Recycling",
    "Rail Services",
    "Government",
    "Unknown",
]

_LLM_SECTOR_LIST = ", ".join(LLM_SECTORS)

_BATCH_PROMPT_TEMPLATE = """Classify each company into exactly one of these rail-freight industry sectors based on its name, facility type, and location:

{sector_list}

Return ONLY a valid JSON object mapping the integer index to the sector string.
Example: {{"0": "Agriculture", "1": "Chemicals", "2": "Unknown"}}

Companies to classify:
{companies_json}"""


def _build_batch_prompt(batch: List[dict]) -> str:
    """Build the classification prompt for a single batch."""
    companies_json = json.dumps(
        {
            str(i): {
                "name": row.get("company_name", ""),
                "facility_type": row.get("facility_type", ""),
                "city": row.get("city", ""),
                "state": row.get("state", ""),
            }
            for i, row in enumerate(batch)
        },
        indent=2,
    )
    return _BATCH_PROMPT_TEMPLATE.format(
        sector_list=_LLM_SECTOR_LIST,
        companies_json=companies_json,
    )


def _parse_llm_response(content: str, batch_size: int) -> Dict[int, str]:
    """
    Extract the JSON mapping from the LLM response text.

    Handles responses that wrap the JSON in markdown code fences.

    Args:
        content: Raw text returned by the model.
        batch_size: Expected number of entries (used only for logging).

    Returns:
        Dict mapping int index to sector string.  Empty dict on parse failure.
    """
    # Strip markdown fences if present
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # Drop opening fence (```json or ```) and closing fence
        inner = [
            ln for ln in lines[1:]
            if not ln.strip().startswith("```")
        ]
        text = "\n".join(inner).strip()

    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning(f"LLM response JSON parse failed ({exc}): {text[:200]!r}")
        return {}

    result: Dict[int, str] = {}
    for k, v in raw.items():
        try:
            idx = int(k)
        except ValueError:
            continue
        sector = str(v).strip()
        if sector not in LLM_SECTORS:
            # Normalise common deviations before discarding
            sector = "Unknown"
        if 0 <= idx < batch_size:
            result[idx] = sector

    return result


def classify_batch_with_llm(
    companies: List[dict],
    api_key: str,
    batch_size: int = 50,
) -> Dict[int, str]:
    """
    Classify companies using Claude Haiku via the Anthropic API.

    Sends ``companies`` in batches of ``batch_size`` to Claude Haiku.
    Returns a mapping from the original list index to the classified sector.
    If a batch fails the entries in that batch are silently skipped (no
    partial result from a failed batch is recorded).

    Args:
        companies: List of dicts, each with keys:
            ``company_name``, ``facility_type``, ``city``, ``state``.
        api_key: Anthropic API key (``sk-ant-...``).
        batch_size: Number of companies per API call (max 50 recommended).

    Returns:
        Dict mapping original list index (int) to sector string.

    Raises:
        ImportError: If the ``anthropic`` package is not installed.
    """
    try:
        from anthropic import Anthropic  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "The 'anthropic' package is required for Layer 3 classification.\n"
            "Install it with: pip install anthropic"
        ) from exc

    client = Anthropic(api_key=api_key)
    results: Dict[int, str] = {}

    total_batches = (len(companies) + batch_size - 1) // batch_size
    requests_this_minute = 0
    minute_start = time.monotonic()

    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(companies))
        batch = companies[start_idx:end_idx]

        # Rate-limit: max 50 requests per minute
        requests_this_minute += 1
        elapsed = time.monotonic() - minute_start
        if requests_this_minute >= 50 and elapsed < 60.0:
            sleep_for = 60.0 - elapsed + 1.0
            logger.info(f"Rate limit pause: sleeping {sleep_for:.1f}s")
            time.sleep(sleep_for)
            requests_this_minute = 0
            minute_start = time.monotonic()
        elif elapsed >= 60.0:
            requests_this_minute = 1
            minute_start = time.monotonic()

        prompt = _build_batch_prompt(batch)

        try:
            response = client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.content[0].text
        except Exception as exc:
            logger.warning(
                f"LLM batch {batch_num + 1}/{total_batches} failed: {exc}. Skipping."
            )
            continue

        batch_results = _parse_llm_response(content, len(batch))

        # Remap local batch indices to global indices
        for local_idx, sector in batch_results.items():
            global_idx = start_idx + local_idx
            results[global_idx] = sector

        classified = sum(1 for s in batch_results.values() if s != "Unknown")
        logger.info(
            f"LLM batch {batch_num + 1}/{total_batches}: "
            f"classified {classified}/{len(batch)} entries"
        )

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_unknown(value: object) -> bool:
    """Return True if a sector value is absent / Unknown / NaN."""
    if value is None:
        return True
    import math
    try:
        if isinstance(value, float) and math.isnan(value):
            return True
    except (TypeError, ValueError):
        pass
    s = str(value).strip()
    return s == "" or s.lower() in ("unknown", "nan")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def classify_industries(
    csv_path: str,
    output_path: str,
    api_key: str = "",
) -> None:
    """
    Classify rail-company industries using three stacked layers.

    Reads a CSV produced by the rail-network scanner pipeline, applies
    classification layers in order (each layer only touches rows that
    are still Unknown), then writes the enriched CSV to ``output_path``.

    Args:
        csv_path: Path to the input CSV file.
        output_path: Path where the enriched CSV will be written.
        api_key: Anthropic API key.  Layer 3 is skipped when empty.
    """
    try:
        import pandas as pd  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "pandas is required. Install with: pip install pandas"
        ) from exc

    df = pd.read_csv(csv_path, dtype=str)
    logger.info(f"Loaded {len(df)} rows from {csv_path}")

    # Ensure the sector column exists
    if "sector_guess" not in df.columns:
        df["sector_guess"] = ""

    total_rows = len(df)

    def unknown_mask(frame: "pd.DataFrame") -> "pd.Series":
        return frame["sector_guess"].apply(_is_unknown)

    # ------------------------------------------------------------------
    # Layer 1: keyword matching
    # ------------------------------------------------------------------
    mask1 = unknown_mask(df)
    count_before = mask1.sum()

    def _apply_keywords(row: "pd.Series") -> str:
        current = row["sector_guess"]
        if not _is_unknown(current):
            return current
        return classify_by_keywords(
            str(row.get("company_name", "")),
            str(row.get("facility_type", "")),
        )

    df.loc[mask1, "sector_guess"] = df[mask1].apply(_apply_keywords, axis=1)

    classified_l1 = count_before - unknown_mask(df).sum()
    logger.info(
        f"Layer 1 (keywords): classified {classified_l1}/{count_before} unknowns "
        f"({count_before - classified_l1} remain)"
    )

    # ------------------------------------------------------------------
    # Layer 2: Overture category mapping
    # ------------------------------------------------------------------
    mask2 = unknown_mask(df)
    count_before2 = mask2.sum()

    def _apply_overture(row: "pd.Series") -> str:
        current = row["sector_guess"]
        if not _is_unknown(current):
            return current
        return classify_by_overture_category(str(row.get("facility_type", "")))

    df.loc[mask2, "sector_guess"] = df[mask2].apply(_apply_overture, axis=1)

    classified_l2 = count_before2 - unknown_mask(df).sum()
    logger.info(
        f"Layer 2 (Overture): classified {classified_l2}/{count_before2} unknowns "
        f"({count_before2 - classified_l2} remain)"
    )

    # ------------------------------------------------------------------
    # Layer 3: Claude Haiku LLM classification
    # ------------------------------------------------------------------
    if api_key:
        mask3 = unknown_mask(df)
        count_before3 = int(mask3.sum())

        if count_before3 > 0:
            unknown_indices = df.index[mask3].tolist()
            unknown_rows = df.loc[mask3][
                ["company_name", "facility_type", "city", "state"]
            ].to_dict("records")

            logger.info(
                f"Layer 3 (LLM): sending {count_before3} unknowns to Claude Haiku"
            )

            try:
                llm_results = classify_batch_with_llm(unknown_rows, api_key)
            except ImportError as exc:
                logger.error(f"Layer 3 skipped — {exc}")
                llm_results = {}

            for local_idx, sector in llm_results.items():
                if not _is_unknown(sector):
                    df.at[unknown_indices[local_idx], "sector_guess"] = sector

            classified_l3 = count_before3 - unknown_mask(df).sum()
            logger.info(
                f"Layer 3 (LLM): classified {classified_l3}/{count_before3} unknowns "
                f"({count_before3 - classified_l3} remain)"
            )
        else:
            logger.info("Layer 3 (LLM): no unknowns remain, skipping")
    else:
        logger.info("Layer 3 (LLM): skipped (no API key provided)")

    # ------------------------------------------------------------------
    # Summary and output
    # ------------------------------------------------------------------
    final_unknown = int(unknown_mask(df).sum())
    classified_total = total_rows - final_unknown
    logger.info(
        f"Classification complete: {classified_total}/{total_rows} rows classified "
        f"({final_unknown} remain Unknown)"
    )

    df.to_csv(output_path, index=False)
    logger.info(f"Enriched CSV written to {output_path}")


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Classify rail-company industries using multi-layer approach."
    )
    parser.add_argument(
        "csv_path",
        help="Path to input CSV (e.g. data/output/all_states_enhanced_rail_companies.csv)",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output path.  Defaults to <csv_path stem>_classified.csv in the same dir.",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="Anthropic API key for Layer 3 (Claude Haiku).  Omit to skip LLM layer.",
    )
    args = parser.parse_args()

    from pathlib import Path  # noqa: PLC0415 (late import fine here)

    input_path = Path(args.csv_path)
    if not input_path.exists():
        print(f"Error: file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_path = (
        args.output
        or str(input_path.parent / f"{input_path.stem}_classified{input_path.suffix}")
    )

    classify_industries(
        csv_path=str(input_path),
        output_path=output_path,
        api_key=args.api_key,
    )
