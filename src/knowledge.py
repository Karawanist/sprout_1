"""Rule-based comparisons against explicitly unverified plant profiles."""
import json
from pathlib import Path
import pandas as pd

def load_profiles(path="data/knowledge/plant_profiles.json"):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)

def compare_to_profile(recent_weather: pd.DataFrame, profile: dict) -> list[str]:
    notes = []
    bounds = profile.get("optimal_mean_temp_c", profile.get("temp_optimal_c"))
    if bounds and {"T2M_MAX", "T2M_MIN"}.issubset(recent_weather.columns):
        lo, hi = bounds
        mean_temp = ((recent_weather.T2M_MAX + recent_weather.T2M_MIN) / 2).mean()
        if pd.notna(mean_temp) and mean_temp > hi:
            notes.append(f"Recent mean air temperature ({mean_temp:.1f} °C) is above the cited ecological range ({lo}–{hi} °C).")
        elif pd.notna(mean_temp) and mean_temp < lo:
            notes.append(f"Recent mean air temperature ({mean_temp:.1f} °C) is below the cited ecological range ({lo}–{hi} °C).")
        if not notes:
            notes.append("Recent mean air temperature falls within the cited ecological range; this is not a crop suitability verdict.")
    if not profile.get("locally_validated", False):
        notes.append("This FAO profile has not been validated for the local cultivar, field, or management conditions.")
    return notes or ["No temperature comparison is available for this profile and data."]
