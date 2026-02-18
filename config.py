"""
Rail Network Scanner configuration.

Region definitions, API settings, and constants.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SPURS_DIR = DATA_DIR / "spurs"
COMPANIES_DIR = DATA_DIR / "companies"
SATELLITE_DIR = DATA_DIR / "satellite"
OUTPUT_DIR = DATA_DIR / "output"
CACHE_DIR = PROJECT_ROOT / ".cache"

# Ensure directories exist
for d in [RAW_DIR, SPURS_DIR, COMPANIES_DIR, SATELLITE_DIR, OUTPUT_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --- API Keys ---
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# --- Overpass API ---
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
OVERPASS_URL = OVERPASS_URLS[0]
OVERPASS_RATE_LIMIT = 1 / 5  # 1 request per 5 seconds
OVERPASS_TIMEOUT = 300  # seconds

# --- Google Places API ---
PLACES_NEARBY_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
PLACES_RATE_LIMIT = 2.5  # requests per second (conservative)
PLACES_SEARCH_RADIUS_M = 150  # meters from spur endpoint

# Industrial place types for filtering
PLACES_INDUSTRIAL_TYPES = {
    "accounting", "car_dealer", "car_repair", "car_wash",
    "electrician", "food", "gas_station", "general_contractor",
    "grocery_or_supermarket", "hardware_store", "home_goods_store",
    "industrial", "locksmith", "moving_company", "painter",
    "plumber", "roofing_contractor", "storage", "store",
    "transit_station", "trucking_company", "warehouse",
}

# --- Region Definitions ---
# Each region is an Overpass area selector
REGIONS = {
    "illinois": {
        "name": "Illinois",
        "overpass_area": '["name"="Illinois"]["admin_level"="4"]',
        "bbox": (36.97, -91.51, 42.51, -87.02),  # (south, west, north, east)
        "description": "MVP state - high rail density, mix of ag and industrial",
    },
    "indiana": {
        "name": "Indiana",
        "overpass_area": '["name"="Indiana"]["admin_level"="4"]',
        "bbox": (37.77, -88.10, 41.76, -84.78),
        "description": "Heavy steel and manufacturing rail traffic",
    },
    "ohio": {
        "name": "Ohio",
        "overpass_area": '["name"="Ohio"]["admin_level"="4"]',
        "bbox": (38.40, -84.82, 42.32, -80.52),
        "description": "Major NS and CSX corridors",
    },
    "texas": {
        "name": "Texas",
        "overpass_area": '["name"="Texas"]["admin_level"="4"]',
        "bbox": (25.84, -106.65, 36.50, -93.51),
        "description": "Largest rail network by mileage, energy and chemicals",
    },
}

# --- Validation ---
KNOWN_ACCOUNTS_PATH = Path(
    "/Users/nicoamoretti/nico_repo/company-finder/high_rail_accounts.csv"
)

# --- Logging ---
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
