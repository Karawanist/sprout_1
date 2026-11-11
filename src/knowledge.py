"""Rule-based comparisons against explicitly unverified plant profiles."""
import json
from pathlib import Path
import pandas as pd

def load_profiles(path="data/knowledge/plant_profiles.json"):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)

def compare_to_profile(recent_weather: pd.DataFrame, profile: dict) -> list[str]:
    notes = []
    bounds = profile.get("temp_optimal_c")
    if bounds and {"T2M_MAX", "T2M_MIN"}.issubset(recent_weather.columns):
        lo, hi = bounds
        max_t, min_t = recent_weather.T2M_MAX.mean(), recent_weather.T2M_MIN.mean()
        if pd.notna(max_t) and max_t > hi:
            notes.append(f"Recent daytime highs (average {max_t:.1f} °C) are above the documented range ({lo}–{hi} °C).")
        if pd.notna(min_t) and min_t < lo - 8:
            notes.append(f"Recent nights (average low {min_t:.1f} °C) are quite cool for this plant.")
        if not notes:
            notes.append("Recent temperatures fall within the profile's documented range.")
    if not profile.get("verified", False):
        notes.append("This plant profile has not been verified against a cited source.")
    return notes or ["No temperature comparison is available for this profile and data."]
