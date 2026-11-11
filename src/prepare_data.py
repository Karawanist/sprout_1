"""Build clean weather and NDVI feature tables from downloaded source CSVs."""
from pathlib import Path
import pandas as pd
from src.data_collection import combine_power_files
from src.features import clean_weather, load_ndvi_csv, build_features

def main():
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    weather = combine_power_files()
    clean = clean_weather(weather)
    clean.to_csv("data/processed/power_daily_regions_clean.csv", index=False)
    ndvi_path = Path("data/raw/appeears_ndvi.csv")
    if not ndvi_path.exists():
        raise SystemExit("Weather prepared. Add data/raw/appeears_ndvi.csv from AppEEARS, then rerun this command to build NDVI features.")
    ndvi = load_ndvi_csv(ndvi_path)
    features = build_features(clean, ndvi)
    if features.empty:
        raise SystemExit("No joined observations. Check region labels, dates, and weather coverage.")
    features.to_csv("data/processed/ndvi_weather_features.csv", index=False)
    print(f"Saved {len(clean)} daily weather rows and {len(features)} joined NDVI periods.")

if __name__ == "__main__": main()
