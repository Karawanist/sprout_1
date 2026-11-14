"""NASA POWER daily point API client for Sprout study regions."""
from __future__ import annotations

from datetime import date
from pathlib import Path
import time
import requests
import pandas as pd

from src.data_collection import REGIONS

API_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
AG_PARAMETERS = ["T2M_MAX", "T2M_MIN", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN"]
WETNESS_PARAMETERS = ["GWETROOT"]
POWER_PARAMETERS = AG_PARAMETERS + WETNESS_PARAMETERS

def _request_point(latitude: float, longitude: float, start: date, end: date,
                   parameters: list[str], community: str) -> pd.DataFrame:
    params = {
        "parameters": ",".join(parameters),
        "community": community,
        "longitude": longitude,
        "latitude": latitude,
        "start": start.strftime("%Y%m%d"),
        "end": end.strftime("%Y%m%d"),
        "format": "JSON",
        "time-standard": "UTC",
    }
    response = requests.get(API_URL, params=params, timeout=(10, 60))
    response.raise_for_status()
    payload = response.json()
    if payload.get("messages") and not payload.get("properties", {}).get("parameter"):
        raise ValueError("NASA POWER returned: " + "; ".join(map(str, payload["messages"])))
    values = payload.get("properties", {}).get("parameter")
    if not isinstance(values, dict) or not values:
        raise ValueError("NASA POWER response did not include daily parameter values.")
    frame = pd.DataFrame(values)
    frame.index = pd.to_datetime(frame.index.astype(str), format="%Y%m%d", errors="coerce")
    frame.index.name = "date"
    frame = frame.reset_index()
    for col in frame.columns:
        if col != "date":
            frame[col] = pd.to_numeric(frame[col], errors="coerce").replace(-999, float("nan"))
    return frame.dropna(subset=["date"])

def fetch_region_power_daily(region: str, start: date, end: date) -> tuple[pd.DataFrame, list[str]]:
    """Fetch daily weather and root-zone wetness for one supported study point."""
    if region not in REGIONS:
        raise ValueError(f"Unsupported study region: {region}")
    latitude, longitude = REGIONS[region]
    result = _request_point(latitude, longitude, start, end, POWER_PARAMETERS, "AG")
    result["region"] = region
    return result.sort_values("date").reset_index(drop=True), []

def fetch_study_regions_history(start: date, end: date, output_dir="data/raw/api_power") -> list[Path]:
    """Download complete study-point history sequentially into a separate raw folder."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    outputs = []
    for index, region in enumerate(REGIONS):
        frame, warnings = fetch_region_power_daily(region, start, end)
        if warnings:
            raise RuntimeError(f"{region}: " + " ".join(warnings))
        path = destination / f"power_{region}.csv"
        frame.to_csv(path, index=False)
        outputs.append(path)
        if index < len(REGIONS) - 1:
            time.sleep(1)
    return outputs
