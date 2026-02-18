# Rail Network Scanner

## Overview

Scans the North American railroad network using OpenStreetMap data and Google Places API to discover companies with direct rail access (sidings, spurs, loading facilities). Outputs a database of rail-served companies for B2B sales targeting.

## Architecture

### Pipeline Phases

| Phase | Module | Description |
|-------|--------|-------------|
| 1 | `src/osm_extractor.py` | Pull railway spurs/sidings from OSM Overpass API |
| 2 | `src/spur_finder.py` | Extract facility-side endpoint coordinates |
| 3 | `src/company_identifier.py` | Identify businesses via Google Places API |
| 4 | `src/satellite_capture.py` | Satellite imagery capture (future) |
| 4 | `src/vision_classifier.py` | AI vision classification (future) |
| 5 | `src/enrichment.py` | Deduplicate, score, cross-reference |

### Key Algorithm

**Spur endpoint detection** (`spur_finder.py`):
1. Get all nodes for each spur way
2. Find junction nodes (shared between multiple ways)
3. The endpoint NOT at a junction is the facility side
4. If ambiguous, use the spur midpoint

### Utilities

- `src/utils/cache.py` — JSON file-based caching for all API responses (`.cache/` dir)
- `src/utils/rate_limiter.py` — Token-bucket rate limiter per API
- `src/utils/geo_utils.py` — Haversine distance, bounding boxes, coordinate helpers

## Quick Start

```bash
# Install dependencies
pip3 install -r requirements.txt

# Set up API keys
cp .env.example .env
# Edit .env with your GOOGLE_MAPS_API_KEY

# Run MVP pipeline (Illinois)
python3 scripts/run_mvp.py

# Run for any region
python3 scripts/run_region.py illinois
python3 scripts/run_region.py --list

# OSM extraction only (no API cost)
python3 scripts/run_region.py illinois --skip-places

# Validate against known accounts
python3 scripts/validate_against_known.py illinois

# Run tests
python3 -m pytest tests/ -v
```

## API Keys

- `GOOGLE_MAPS_API_KEY` — Required for Phase 3 (Places API, ~$0.032/request)
- `ANTHROPIC_API_KEY` — Phase 4 only (vision classification, not yet implemented)

## Rate Limits

| API | Configured Rate | Notes |
|-----|----------------|-------|
| Overpass | 1 req / 5 sec | Free, no key needed |
| Google Places | 2.5 req / sec | ~$32 per 1000 endpoints |

All responses are cached in `.cache/` — re-running is free after first fetch.

## Regions

Config-driven in `config.py`. Currently defined:
- **illinois** (MVP) — High rail density, ag + industrial
- **indiana** — Heavy steel and manufacturing
- **ohio** — Major NS and CSX corridors
- **texas** — Largest rail network, energy + chemicals

## Output

Final CSV at `data/output/{region}_rail_companies.csv`:
```
company_name, address, city, state, zip, lat, lon, facility_type,
google_place_id, osm_way_id, spur_length_m, connected_to_mainline,
confidence, discovery_date, in_existing_list, sector_guess
```

## Related Projects

- `/Users/nicoamoretti/nico_repo/company-finder/` — Works in opposite direction (known companies → rail evidence)
- `high_rail_accounts.csv` — 347 known rail-served companies for validation

## Current Status

- [x] Phase 1: OSM extraction
- [x] Phase 2: Spur endpoint detection
- [x] Phase 3: Google Places identification
- [ ] Phase 4: Satellite imagery + vision (future)
- [x] Phase 5: Enrichment + deduplication
- [x] MVP pipeline script
- [x] Validation script
- [x] Tests (25 passing)
- [ ] MVP run on Illinois (pending GOOGLE_MAPS_API_KEY in .env)

## Illinois OSM Extraction Results (2026-02-18)

| Metric | Count |
|--------|-------|
| Total spur/siding ways | 12,767 |
| Mainline node IDs fetched | 107,110 |
| Endpoints by mainline exclusion | 3,177 |
| Endpoints by midpoint (ambiguous) | 9,590 |
| Duplicates removed (within 50m) | 3,372 |
| **Final unique endpoints** | **9,395** |

Estimated Places API cost: ~$300 for all 9,395 endpoints.

## Key Discovery: OSM Tagging

**US OSM data does NOT use `railway=spur`**. Instead it uses:
- `railway=rail` + `service=spur` for industrial spurs
- `railway=rail` + `service=siding` for sidings
- `railway=rail` + `service=yard` for yard tracks

The query was updated to match both `railway=spur` (non-US) and `railway=rail` + `service=spur` (US) patterns.

## Overpass API Notes

- Primary server (`overpass-api.de`) frequently returns 504 Gateway Timeout
- Fallback server (`overpass.kumi.systems`) is more reliable but slightly stale data
- Area queries (`area["name"="Illinois"]`) don't work reliably — use bbox instead
- All responses are cached in `.cache/overpass/` for fast re-runs
