# Source data setup

This project uses the NASA POWER API for weather. You can also keep manually downloaded CSV originals here unchanged.

## Fetch the full POWER history through the API

From the repository root, run:

```powershell
.venv\Scripts\python.exe -m src.fetch_power_history
```

The script downloads daily values from 2004-12-01 through 2024-12-31 for the five study points. It requests `T2M_MAX`, `T2M_MIN`, `PRECTOTCORR`, `ALLSKY_SFC_SW_DWN`, and `GWETROOT` together. Files are written under `data/raw/api_power/`; existing manual downloads are not overwritten. Data fetching is sequential and NASA POWER requires no key.

## NASA POWER

Download daily Agroclimatology CSVs for the five points in `study_points.csv`, ideally from 2004-12-01 through 2024-12-31. Request `T2M_MAX`, `T2M_MIN`, `PRECTOTCORR`, `ALLSKY_SFC_SW_DWN`, and `GWETROOT`. Save each file using these exact names:

- `power_delta_gharbia.csv`
- `power_delta_menoufia.csv`
- `power_delta_sharqia.csv`
- `power_middle_egypt_minya.csv`
- `power_upper_egypt_sohag.csv`

## NASA AppEEARS MODIS NDVI

Create an AppEEARS Point Samples CSV request with the uploaded study points, dates 2005-01-01 through 2024-12-31, product `MOD13Q1.061`, layers `_250m_16_days_NDVI` and `_250m_16_days_pixel_reliability`. Save its CSV as `appeears_ndvi.csv`. The CSV should include a date, a region/category matching the names above, NDVI, and (ideally) pixel reliability.

Then run `python -m src.prepare_data`. POWER is processed even if the AppEEARS download has not arrived yet. Preparation creates derived files in `data/processed/`; it never changes these raw inputs. Poor quality NDVI observations are dropped, not imputed.

The chat app retrieves recent weather directly from NASA POWER and falls back to `data/processed/power_daily_regions_clean.csv` if the API is unavailable. The MODIS NDVI target is still required for training Sprout's greenness model; POWER alone cannot supply that target.
