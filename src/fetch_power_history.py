"""Fetch historical POWER parameters missing from earlier manual CSV exports."""
from datetime import date
from src.power_api import fetch_study_regions_history

def main():
    paths = fetch_study_regions_history(date(2004, 12, 1), date(2024, 12, 31))
    print("Saved POWER API source files:")
    for path in paths:
        print(f"  {path}")

if __name__ == "__main__":
    main()
