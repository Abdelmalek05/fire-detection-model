"""Live smoke test for FIRMS + Open-Meteo. Run manually; not part of pytest."""
import sys

import requests

from firerisk.config import load_config

cfg = load_config()

r = requests.get(
    f"https://firms.modaps.eosdis.nasa.gov/api/data_availability/csv/{cfg.map_key}/ALL",
    timeout=60,
)
print("FIRMS availability:", r.status_code)
print(r.text.splitlines()[0] if r.ok else r.text[:200])
if not r.ok:
    sys.exit(1)

w, s, e, n = cfg.bbox
r = requests.get(
    f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{cfg.map_key}"
    f"/{cfg.firms_source}/{w},{s},{e},{n}/{cfg.firms_day_range}/2023-07-20",
    timeout=180,
)
print("FIRMS area:", r.status_code, len(r.text.splitlines()), "lines")

r = requests.get(
    "https://archive-api.open-meteo.com/v1/archive",
    params={
        "latitude": "36.7", "longitude": "4.1",
        "start_date": "2023-08-01", "end_date": "2023-08-05",
        "daily": "temperature_2m_max,precipitation_sum",
        "timezone": "Africa/Algiers",
    },
    timeout=60,
)
print("Open-Meteo:", r.status_code, r.json().get("daily", {}).get("time"))
