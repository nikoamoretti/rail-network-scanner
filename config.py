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
    "alabama": {
        "name": "Alabama",
        "overpass_area": '["name"="Alabama"]["admin_level"="4"]',
        "bbox": (30.14, -88.47, 35.01, -84.89),
        "description": "NS and CSX serve heavy manufacturing and auto industry spurs",
    },
    "arizona": {
        "name": "Arizona",
        "overpass_area": '["name"="Arizona"]["admin_level"="4"]',
        "bbox": (31.33, -114.82, 37.00, -109.04),
        "description": "BNSF and UP transcontinental routes through Tucson and Phoenix corridors",
    },
    "arkansas": {
        "name": "Arkansas",
        "overpass_area": '["name"="Arkansas"]["admin_level"="4"]',
        "bbox": (33.00, -94.62, 36.50, -89.64),
        "description": "UP and short-line network serving agriculture and timber",
    },
    "california": {
        "name": "California",
        "overpass_area": '["name"="California"]["admin_level"="4"]',
        "bbox": (32.53, -124.41, 42.01, -114.13),
        "description": "BNSF and UP intermodal hubs at LA and Bay Area ports",
    },
    "colorado": {
        "name": "Colorado",
        "overpass_area": '["name"="Colorado"]["admin_level"="4"]',
        "bbox": (36.99, -109.06, 41.00, -102.04),
        "description": "UP and BNSF coal and intermodal routes through Denver",
    },
    "connecticut": {
        "name": "Connecticut",
        "overpass_area": '["name"="Connecticut"]["admin_level"="4"]',
        "bbox": (40.98, -73.73, 42.05, -71.79),
        "description": "Dense short-line network and Pan Am Railways freight corridors",
    },
    "delaware": {
        "name": "Delaware",
        "overpass_area": '["name"="Delaware"]["admin_level"="4"]',
        "bbox": (38.45, -75.79, 39.84, -75.05),
        "description": "NS and short lines serving chemical and port facilities",
    },
    "florida": {
        "name": "Florida",
        "overpass_area": '["name"="Florida"]["admin_level"="4"]',
        "bbox": (24.54, -87.63, 31.00, -80.03),
        "description": "CSX and FEC serve phosphate, intermodal, and auto facilities",
    },
    "georgia": {
        "name": "Georgia",
        "overpass_area": '["name"="Georgia"]["admin_level"="4"]',
        "bbox": (30.36, -85.61, 35.00, -80.84),
        "description": "NS and CSX hub at Atlanta with heavy auto and intermodal traffic",
    },
    "idaho": {
        "name": "Idaho",
        "overpass_area": '["name"="Idaho"]["admin_level"="4"]',
        "bbox": (41.99, -117.24, 49.00, -111.04),
        "description": "UP main line and short lines serving timber and agricultural shippers",
    },
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
    "iowa": {
        "name": "Iowa",
        "overpass_area": '["name"="Iowa"]["admin_level"="4"]',
        "bbox": (40.37, -96.64, 43.50, -90.14),
        "description": "BNSF and UP grain and ethanol network across the corn belt",
    },
    "kansas": {
        "name": "Kansas",
        "overpass_area": '["name"="Kansas"]["admin_level"="4"]',
        "bbox": (36.99, -102.05, 40.00, -94.59),
        "description": "BNSF and UP wheat and grain corridors through Wichita and Dodge City",
    },
    "kentucky": {
        "name": "Kentucky",
        "overpass_area": '["name"="Kentucky"]["admin_level"="4"]',
        "bbox": (36.50, -89.57, 39.15, -81.96),
        "description": "CSX coal and intermodal main lines through Louisville",
    },
    "louisiana": {
        "name": "Louisiana",
        "overpass_area": '["name"="Louisiana"]["admin_level"="4"]',
        "bbox": (28.92, -94.04, 33.02, -88.82),
        "description": "UP and KCS serving Gulf Coast petrochemical and port facilities",
    },
    "maine": {
        "name": "Maine",
        "overpass_area": '["name"="Maine"]["admin_level"="4"]',
        "bbox": (43.06, -71.08, 47.46, -66.95),
        "description": "Pan Am Railways and short lines serving paper and forest products",
    },
    "maryland": {
        "name": "Maryland",
        "overpass_area": '["name"="Maryland"]["admin_level"="4"]',
        "bbox": (37.91, -79.49, 39.72, -74.98),
        "description": "CSX and NS serve Baltimore port and chemical facilities",
    },
    "massachusetts": {
        "name": "Massachusetts",
        "overpass_area": '["name"="Massachusetts"]["admin_level"="4"]',
        "bbox": (41.24, -73.53, 42.89, -69.93),
        "description": "Pan Am Railways and short lines with dense industrial spur network",
    },
    "michigan": {
        "name": "Michigan",
        "overpass_area": '["name"="Michigan"]["admin_level"="4"]',
        "bbox": (41.70, -90.42, 48.19, -82.41),
        "description": "NS, CSX, and CN serving auto plants and Great Lakes ports",
    },
    "minnesota": {
        "name": "Minnesota",
        "overpass_area": '["name"="Minnesota"]["admin_level"="4"]',
        "bbox": (43.50, -97.24, 49.38, -89.49),
        "description": "BNSF and CP grain and taconite ore routes to Twin Ports",
    },
    "mississippi": {
        "name": "Mississippi",
        "overpass_area": '["name"="Mississippi"]["admin_level"="4"]',
        "bbox": (30.17, -91.66, 35.01, -88.10),
        "description": "CN, KCS, and NS serve timber, paper, and agricultural corridors",
    },
    "missouri": {
        "name": "Missouri",
        "overpass_area": '["name"="Missouri"]["admin_level"="4"]',
        "bbox": (35.99, -95.77, 40.61, -89.10),
        "description": "BNSF and UP main lines converge at Kansas City interchange hub",
    },
    "montana": {
        "name": "Montana",
        "overpass_area": '["name"="Montana"]["admin_level"="4"]',
        "bbox": (44.36, -116.05, 49.00, -104.04),
        "description": "BNSF northern transcontinental route with coal and grain traffic",
    },
    "nebraska": {
        "name": "Nebraska",
        "overpass_area": '["name"="Nebraska"]["admin_level"="4"]',
        "bbox": (39.99, -104.05, 43.00, -95.31),
        "description": "UP main line and BNSF grain routes through Omaha hub",
    },
    "nevada": {
        "name": "Nevada",
        "overpass_area": '["name"="Nevada"]["admin_level"="4"]',
        "bbox": (35.00, -120.01, 42.00, -114.04),
        "description": "UP and BNSF transcontinental routes with mining branch lines",
    },
    "new hampshire": {
        "name": "New Hampshire",
        "overpass_area": '["name"="New Hampshire"]["admin_level"="4"]',
        "bbox": (42.70, -72.56, 45.31, -70.61),
        "description": "Short-line network serving regional manufacturing and forest products",
    },
    "new jersey": {
        "name": "New Jersey",
        "overpass_area": '["name"="New Jersey"]["admin_level"="4"]',
        "bbox": (38.93, -75.56, 41.36, -73.89),
        "description": "NS and CSX dense corridor with chemical and port industrial spurs",
    },
    "new mexico": {
        "name": "New Mexico",
        "overpass_area": '["name"="New Mexico"]["admin_level"="4"]',
        "bbox": (31.33, -109.05, 37.00, -103.00),
        "description": "BNSF and UP transcontinental routes with potash mining spurs",
    },
    "new york": {
        "name": "New York",
        "overpass_area": '["name"="New York"]["admin_level"="4"]',
        "bbox": (40.50, -79.76, 45.01, -71.86),
        "description": "CSX and NS water-level routes with heavy intermodal and industrial traffic",
    },
    "north carolina": {
        "name": "North Carolina",
        "overpass_area": '["name"="North Carolina"]["admin_level"="4"]',
        "bbox": (33.84, -84.32, 36.59, -75.46),
        "description": "NS and CSX serve automotive, textile, and agricultural shippers",
    },
    "north dakota": {
        "name": "North Dakota",
        "overpass_area": '["name"="North Dakota"]["admin_level"="4"]',
        "bbox": (45.93, -104.05, 49.00, -96.55),
        "description": "BNSF and CP heavy grain and crude oil unit train corridors",
    },
    "ohio": {
        "name": "Ohio",
        "overpass_area": '["name"="Ohio"]["admin_level"="4"]',
        "bbox": (38.40, -84.82, 42.32, -80.52),
        "description": "Major NS and CSX corridors",
    },
    "oklahoma": {
        "name": "Oklahoma",
        "overpass_area": '["name"="Oklahoma"]["admin_level"="4"]',
        "bbox": (33.62, -103.00, 37.00, -94.43),
        "description": "BNSF and UP serve energy, agriculture, and intermodal corridors",
    },
    "oregon": {
        "name": "Oregon",
        "overpass_area": '["name"="Oregon"]["admin_level"="4"]',
        "bbox": (41.99, -124.57, 46.26, -116.46),
        "description": "UP and BNSF serve Pacific ports with timber and intermodal traffic",
    },
    "pennsylvania": {
        "name": "Pennsylvania",
        "overpass_area": '["name"="Pennsylvania"]["admin_level"="4"]',
        "bbox": (39.72, -80.52, 42.27, -74.69),
        "description": "NS and CSX heavy industrial corridors with dense spur network",
    },
    "rhode island": {
        "name": "Rhode Island",
        "overpass_area": '["name"="Rhode Island"]["admin_level"="4"]',
        "bbox": (41.15, -71.86, 42.02, -71.12),
        "description": "Short-line freight serving Providence port and light manufacturing",
    },
    "south carolina": {
        "name": "South Carolina",
        "overpass_area": '["name"="South Carolina"]["admin_level"="4"]',
        "bbox": (32.05, -83.35, 35.22, -78.54),
        "description": "CSX and NS serve auto plants, port of Charleston, and textile mills",
    },
    "south dakota": {
        "name": "South Dakota",
        "overpass_area": '["name"="South Dakota"]["admin_level"="4"]',
        "bbox": (42.48, -104.06, 45.95, -96.44),
        "description": "BNSF and short lines serving grain elevators and ethanol plants",
    },
    "tennessee": {
        "name": "Tennessee",
        "overpass_area": '["name"="Tennessee"]["admin_level"="4"]',
        "bbox": (34.98, -90.31, 36.68, -81.65),
        "description": "NS and CSX serve auto assembly, chemical, and intermodal facilities",
    },
    "texas": {
        "name": "Texas",
        "overpass_area": '["name"="Texas"]["admin_level"="4"]',
        "bbox": (25.84, -106.65, 36.50, -93.51),
        "description": "Largest rail network by mileage, energy and chemicals",
    },
    "utah": {
        "name": "Utah",
        "overpass_area": '["name"="Utah"]["admin_level"="4"]',
        "bbox": (36.99, -114.05, 42.00, -109.04),
        "description": "UP and short lines serving mining, potash, and intermodal corridors",
    },
    "vermont": {
        "name": "Vermont",
        "overpass_area": '["name"="Vermont"]["admin_level"="4"]',
        "bbox": (42.73, -73.44, 45.02, -71.50),
        "description": "New England Central and short lines serving agriculture and paper mills",
    },
    "virginia": {
        "name": "Virginia",
        "overpass_area": '["name"="Virginia"]["admin_level"="4"]',
        "bbox": (36.54, -83.68, 39.47, -75.24),
        "description": "NS and CSX coal and intermodal corridors through Roanoke and Richmond",
    },
    "washington": {
        "name": "Washington",
        "overpass_area": '["name"="Washington"]["admin_level"="4"]',
        "bbox": (45.54, -124.73, 49.00, -116.92),
        "description": "BNSF and UP serve Seattle and Tacoma ports with grain and intermodal",
    },
    "west virginia": {
        "name": "West Virginia",
        "overpass_area": '["name"="West Virginia"]["admin_level"="4"]',
        "bbox": (37.20, -82.64, 40.64, -77.72),
        "description": "CSX and NS coal mine branches dominate one of the densest rail networks",
    },
    "wisconsin": {
        "name": "Wisconsin",
        "overpass_area": '["name"="Wisconsin"]["admin_level"="4"]',
        "bbox": (42.49, -92.89, 47.08, -86.25),
        "description": "CN, CP, and BNSF serve paper mills, dairy ag, and Great Lakes ports",
    },
    "wyoming": {
        "name": "Wyoming",
        "overpass_area": '["name"="Wyoming"]["admin_level"="4"]',
        "bbox": (40.99, -111.06, 45.01, -104.05),
        "description": "BNSF and UP Powder River Basin coal — highest tonnage corridor in North America",
    },
}

# --- Validation ---
KNOWN_ACCOUNTS_PATH = Path(
    "/Users/nicoamoretti/nico_repo/company-finder/high_rail_accounts.csv"
)

# --- Logging ---
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
