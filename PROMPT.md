# Rail Network Scanner — Project Prompt for AI Coding Agent

## Goal

Build a tool that scans the North American railroad network using OpenStreetMap data and reverse geocoding to discover every company with direct rail access (sidings, spurs, loading facilities). The output is a comprehensive database of rail-served companies that can be used for B2B sales targeting.

**The core concept:** Use OpenStreetMap's already-tagged railroad spurs and sidings to find where industrial facilities connect to the rail network, then use Google Places API to identify the company at each location. Optionally validate with satellite imagery + AI vision for unidentified facilities.

## Context

We have an existing project at `/Users/nicoamoretti/nico_repo/company-finder/` that works in the opposite direction — it takes a list of known companies and searches for evidence of rail connectivity. That project has:

- `capture_satellite_maps.py` — Production-ready Playwright script that captures Google Maps satellite imagery at given lat/lon coordinates. Uses headless Chromium, handles popups, switches to satellite view, zooms, and saves PNGs. **This is reusable — study it and adapt its patterns.**
- `rail_intelligence_improved.py` — DuckDuckGo scraper that searches for rail keywords/railroad names per company. Reusable for enrichment after we discover companies.
- `high_rail_accounts.csv` — 347 confirmed rail-served companies we've already identified. Use this as validation data to test accuracy.
- `geocode_facilities.py` / `get_facility_coordinates.py` — Geocoding utilities.

Read the `CLAUDE.md` and `SATELLITE_CAPTURE_IMPROVEMENTS.md` in that project for additional context before starting.

## Architecture

### Phase 1: Get the Rail Network as Coordinates
- Use the **OpenStreetMap Overpass API** to extract all railroad spurs and sidings in a target region
- OSM tags: `railway=spur` for industrial spur tracks, `railway=siding` for sidings
- **Key insight:** OSM already distinguishes between mainline track and industrial spurs/sidings. We don't need satellite imagery for initial discovery — we just need to reverse geocode the spur endpoints.
- Store the rail network as GeoJSON
- For MVP: Start with **Illinois** (high density of rail-served industrial facilities across ag, chemicals, metals)

### Phase 2: Extract Industrial Spur/Siding Endpoint Locations
- From the OSM data, identify all `railway=spur` and `railway=siding` segments
- For each spur, get the endpoint coordinates (the end that connects to the facility, not the mainline junction)
- The facility-side endpoint is typically the node that is NOT shared with a mainline way
- Store as a list of lat/lon pairs with metadata (OSM way ID, tags, etc.)

### Phase 3: Identify Companies at Spur Locations
- For each spur endpoint, use the **Google Places API** `nearbysearch` endpoint to identify what business/facility is at those coordinates
- Use a small search radius (100-200m)
- Filter for business types likely to be rail shippers (manufacturing, warehousing, industrial, etc.)
- Store: company name, address, place_id, business type, lat/lon, confidence level

### Phase 4: Satellite Imagery Validation (Optional Enhancement — skip for MVP)
- For spur locations where Places API returned no results or low-confidence results
- Adapt `capture_satellite_maps.py` from the existing project to capture tiles at spur endpoints
- Send images to Claude Vision API with prompt: "This satellite image shows a location where a railroad spur connects to a facility. Describe what type of industrial facility this appears to be (grain elevator, chemical plant, lumber yard, warehouse, etc.). Is there visible rail infrastructure?"
- Use AI classification to enrich the facility record

### Phase 5: Enrichment & Deduplication
- Cross-reference discovered companies against our existing `high_rail_accounts.csv` (347 known companies)
- Deduplicate (same company may have multiple spurs/facilities)
- Group facilities by parent company
- Score each facility: confirmed rail spur (from OSM) + confirmed business (from Places API) = HIGH confidence
- Enrich with web search for company revenue, industry, employee count (can reuse `rail_intelligence_improved.py` patterns)

## Tech Stack

- **Python 3.11+**
- **Overpass API** (OpenStreetMap) — free, no API key needed, but rate-limited
- **Google Maps Places API** — requires API key, costs ~$0.032 per nearby search request
- **Playwright** — for satellite image capture if needed (existing pattern from company-finder)
- **Anthropic API** — for vision classification of satellite imagery (Phase 4 only)
- **GeoPandas / Shapely** — for geographic data manipulation
- **pandas** — for data processing
- Results stored in **CSV** and **GeoJSON**

## Project Structure

```
rail-network-scanner/
├── CLAUDE.md                    # Project documentation (keep updated as you build)
├── README.md
├── requirements.txt
├── .env.example                 # Template for API keys
├── config.py                    # Region definitions, constants, API config
├── src/
│   ├── __init__.py
│   ├── osm_extractor.py         # Phase 1: Pull rail spurs/sidings from OSM Overpass API
│   ├── spur_finder.py           # Phase 2: Extract spur endpoint coordinates
│   ├── company_identifier.py    # Phase 3: Reverse geocode spur endpoints via Google Places
│   ├── satellite_capture.py     # Phase 4: Capture satellite imagery (adapt from company-finder)
│   ├── vision_classifier.py     # Phase 4: AI vision classification of facilities
│   ├── enrichment.py            # Phase 5: Enrich, deduplicate, score
│   └── utils/
│       ├── __init__.py
│       ├── geo_utils.py         # Geographic helpers (distance calc, bbox, coordinate transforms)
│       ├── cache.py             # JSON file-based caching for all API responses
│       └── rate_limiter.py      # Rate limiting wrapper for API calls
├── data/
│   ├── raw/                     # Raw OSM extracts (GeoJSON)
│   ├── spurs/                   # Extracted spur endpoint locations
│   ├── companies/               # Identified companies per region
│   ├── satellite/               # Satellite imagery (Phase 4)
│   └── output/                  # Final enriched datasets
├── scripts/
│   ├── run_mvp.py               # End-to-end MVP pipeline for Illinois
│   ├── run_region.py            # Run pipeline for any configured region
│   └── validate_against_known.py # Compare results vs high_rail_accounts.csv
└── tests/
    ├── test_osm_extractor.py
    ├── test_spur_finder.py
    └── test_company_identifier.py
```

## MVP Scope

For the first working version:

1. **One state: Illinois** (high rail density, mix of ag and industrial)
2. Extract all `railway=spur` and `railway=siding` from OSM for Illinois
3. Reverse geocode each spur endpoint using Google Places API
4. Output a CSV: `company_name, address, lat, lon, facility_type, osm_way_id, google_place_id, confidence`
5. Cross-reference against existing 347 known accounts to measure discovery accuracy
6. **Skip** satellite imagery and vision classification for MVP

**Success criteria:** Discover at least 50 rail-served companies in Illinois that are NOT already in our existing 347-account list.

## Example Overpass API Query

```
[out:json][timeout:300];
area["name"="Illinois"]["admin_level"="4"]->.state;
(
  way["railway"="spur"](area.state);
  way["railway"="siding"](area.state);
);
out body;
>;
out skel qt;
```

This returns all railway spurs and sidings in Illinois with their node coordinates.

## Critical Implementation Notes

### Rate Limiting & Caching
- **Overpass API**: Max ~10K elements per query, ~2 requests per minute. Cache all responses.
- **Google Places API**: ~$0.032 per request. For 1000 spurs in Illinois, that's ~$32. Cache every response by lat/lon (rounded to 4 decimal places) so we never re-fetch.
- Build a `cache.py` utility that wraps all API calls with file-based JSON caching. Check cache before every request.

### Rate Limiter
- Build a generic rate limiter class that can be configured per-API
- Overpass: 1 request per 5 seconds
- Google Places: 10 requests per second (their limit), but we should use 2-3/sec to be safe

### Start Small, Validate, Scale
- Get Illinois working and validated before expanding to other states
- Make regions config-driven so adding new states is just a config entry
- Log everything with Python's `logging` module, not print statements

### API Keys
- `GOOGLE_MAPS_API_KEY` from environment variable (`.env` file)
- `ANTHROPIC_API_KEY` from environment variable (Phase 4 only)
- Never hardcode keys. Provide `.env.example` template.

### Determining Facility-Side Endpoint
The key algorithmic challenge is figuring out which end of a spur connects to the facility vs. the mainline. Approach:
1. For each spur way, get all its nodes
2. Check if the first or last node is shared with any `railway=rail` (mainline) way
3. The node NOT shared with mainline is the facility-side endpoint
4. If both or neither are shared, use the midpoint of the spur as the search location

### Output Format
Final CSV columns:
```
company_name, address, city, state, zip, lat, lon, facility_type, google_place_id, 
osm_way_id, spur_length_m, connected_to_mainline, confidence, discovery_date,
in_existing_list (bool), sector_guess
```

## Getting Started Order

1. Read the existing `company-finder` project (especially `capture_satellite_maps.py`, `rail_intelligence_improved.py`, `CLAUDE.md`)
2. Set up project structure, `requirements.txt`, config
3. Build `src/utils/cache.py` and `src/utils/rate_limiter.py` first — everything depends on these
4. Build `src/osm_extractor.py` — query Overpass API for Illinois spurs/sidings
5. Build `src/spur_finder.py` — extract facility-side endpoint coordinates
6. Build `src/company_identifier.py` — Google Places reverse geocode
7. Build `scripts/run_mvp.py` — end-to-end pipeline
8. Build `scripts/validate_against_known.py` — accuracy measurement
9. Run MVP on Illinois, analyze results
10. Update CLAUDE.md with results and findings
