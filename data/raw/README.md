# Source data setup

This project intentionally uses downloaded CSV files. Keep downloaded originals here unchanged.

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
