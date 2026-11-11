<<<<<<< HEAD
# 🌱 Sprout

Sprout is an Egypt-focused agricultural decision-support prototype built around downloaded NASA POWER weather and MODIS NDVI CSV files. It is educational and supports decisions; it does not guarantee yields or outcomes.

## What the MVP supports

- Explore recent NASA POWER weather and root-zone wetness for five example Nile Delta and Nile Valley study points.
- Compare recent temperature observations with a plant profile, clearly marked unverified until a person adds checked values and citations.
- Train a Random Forest to estimate **vegetation greenness anomaly** from the 32 days of weather before a MODIS 16-day composite.
- Review historical model estimates only for the supported study regions, alongside observed MODIS greenness and a tree-disagreement range.
- Ask questions in the app's text box. The MVP uses transparent keyword routing and templates, not an LLM; it summarizes available weather and supported historical greenness, and declines unsupported advice requests.

NDVI is a satellite-derived greenness measure. It is not yield, crop type, field condition, crop suitability, or irrigation need. POWER's grid is coarse and most farmland in the study area is irrigated; weather may explain only a modest share of greenness changes. Consult local agricultural experts for important decisions.

## Quick start

Requires Python 3.10 or newer. Create an environment and install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Download the CSVs described in [`data/raw/README.md`](data/raw/README.md). Keep source downloads unchanged. Prepare derived tables:

```powershell
.venv\Scripts\python.exe -m src.prepare_data
```

If the AppEEARS NDVI file is not ready, preparation still cleans and saves the POWER table, then prints the next step. Once both sources are present, it writes `data/processed/ndvi_weather_features.csv`.

Train and evaluate the anomaly model:

```powershell
.venv\Scripts\python.exe -m src.train
```

Run the geographic transfer diagnostic separately:

```powershell
.venv\Scripts\python.exe -m src.evaluate_regions
```

The training script uses training-year-only NDVI climatology and time splits with a 32-day gap. It reports validation metrics against a constant baseline and a Ridge baseline, evaluates the test set once, and saves artifacts under `models/`. A useful result has to beat the baseline on held-out data; do not present metrics until training on real downloads succeeds.

Launch the app:

```powershell
.venv\Scripts\streamlit.exe run app/streamlit_app.py
```

The weather and profile panels work after POWER preparation. The model panel appears after NDVI preparation and successful training. Missing files produce a setup message.

The text box supports questions about recent conditions and historical greenness. Include a region name (for example, `delta_gharbia`) and optionally a date in `YYYY-MM-DD` form. It cannot answer open-ended questions or predict yield, irrigation, crop suitability, pests, or disease.

## Project layout

```text
data/raw/          Downloaded NASA CSVs and study point list (source data)
data/processed/    Reproducible cleaned and joined tables
data/knowledge/    Structured plant profiles (unverified placeholders)
src/               Loading, preparation, features, training, prediction, knowledge
models/            Locally trained model and training climatology
app/               Streamlit interface
tests/             Small unit checks for core rule behavior
```

## Data collection

Use NASA POWER's daily Agroclimatology CSV export for each point with `T2M_MAX`, `T2M_MIN`, `PRECTOTCORR`, `ALLSKY_SFC_SW_DWN`, and `GWETROOT`. For NDVI, use AppEEARS Point Samples with `MOD13Q1.061`, `_250m_16_days_NDVI`, and `_250m_16_days_pixel_reliability`. Exact dates, labels, and filenames are in [`data/raw/README.md`](data/raw/README.md). Data collection is deliberately manual; API automation is a later improvement.

The study points are rough starting coordinates, not verified field locations. Inspect them on a satellite map before treating them as representative farmland. AppEEARS category labels must match the region IDs in `study_points.csv`.

## Data integrity and limitations

- Raw downloads are not edited; generated tables are written to `data/processed/`.
- Short weather gaps (up to three daily values) are interpolated. Invalid weather rows and poor-quality or impossible NDVI values are excluded.
- Weather inputs cover only the 32 days before each MODIS composite. No weather from the target period or future is used.
- The model target is `ndvi_anomaly`: observed NDVI minus the region's median for the same composite day-of-year, computed from training years.
- The model does not recommend crops, predict yield, schedule irrigation, diagnose disease, or support arbitrary coordinates.
- The tree percentile spread is disagreement among trees, not a formal confidence interval.
- Plant profiles intentionally contain no numeric recommendations. Fill them only with values verified against authoritative, locally relevant sources; keep `verified` false until checked and cite exact sources.
- No trained model or performance claim is included before authentic source CSVs are supplied.

## Evaluation before use

Before treating this as an MVP, run the workflow with real data, record train/validation/test MAE and R², confirm the model beats the zero-anomaly and constant baselines on held-out years, and perform leave-one-region-out validation. The geographic diagnostic uses the available anomaly table and should be reported with its documented climatology limitation. Inspect missingness and each downloaded CSV's schema. Treat weak or negative performance as a valid result and do not deploy unsupported predictions.

This repository is an educational prototype, not guaranteed agricultural advice. NASA data outages or absent source files are reported in the interface; this offline CSV version does not make live NASA calls.
=======
# sprout
>>>>>>> 3b15e5108ca2a9288b51e09e8810f776362d1e3e
