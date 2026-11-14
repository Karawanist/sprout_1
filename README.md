# 🌱 Sprout

Sprout is an Egypt-focused agricultural decision-support prototype. It combines NASA POWER weather, FAO EcoCrop plant profiles, and a MODIS NDVI anomaly model. It supports decisions; it does not guarantee outcomes.

## What it can do

- Fetch recent weather from the NASA POWER API for five example Nile Delta and Nile Valley study points.
- Download historical weather, including precipitation and root-zone wetness, for model preparation.
- Compare recent mean air temperature with FAO EcoCrop's global ecological ranges for tomato, wheat, and maize. Profiles are source-cited but not validated for Egyptian cultivars, fields, or management.
- Train a Random Forest to estimate historical vegetation greenness anomaly from the preceding 32 days of weather, after authentic MODIS NDVI observations are supplied.
- Use the chat box to answer from the available weather, profile, and model evidence. Optional Gemini (free tier) or OpenAI can phrase answers using only that evidence.

NDVI is a satellite-derived greenness measure. It is not yield, crop type, field condition, crop suitability, or irrigation need. POWER's weather and wetness values have different source resolutions. Consult local agricultural experts before making important decisions.

## Setup

Requires Python 3.10 or newer. In PowerShell from the project folder:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Get NASA POWER weather through the API

The API client requests all five fields in one call to NASA's daily point endpoint using the AG community setting. POWER describes root-zone wetness in its hydrology documentation; its effective resolution differs from the meteorological fields. Historical requests are sequential. See [`data/raw/README.md`](data/raw/README.md).

API references: [NASA POWER daily point API](https://power.larc.nasa.gov/docs/services/api/temporal/daily/) and [NASA POWER hydrology data notes](https://power.larc.nasa.gov/docs/methodology/data/sources_hydrology/).

Fetch daily history for the study regions (the originals are saved separately under `data/raw/api_power/`):

```powershell
.venv\Scripts\python.exe -m src.fetch_power_history
```

The app also fetches its recent period live from NASA POWER and falls back to the prepared CSV if the API is temporarily unavailable.

Prepare the weather table:

```powershell
.venv\Scripts\python.exe -m src.prepare_data
```

This saves cleaned weather first. Model features additionally need an AppEEARS MODIS NDVI CSV at `data/raw/appeears_ndvi.csv`, using `MOD13Q1.061` and region labels from `study_points.csv`. Without this observed target, no model is trained.

## Train and evaluate the NDVI model

After both the full POWER history and NDVI file are present:

```powershell
.venv\Scripts\python.exe -m src.prepare_data
.venv\Scripts\python.exe -m src.train
.venv\Scripts\python.exe -m src.evaluate_regions
```

The model predicts `ndvi_anomaly`: observed NDVI minus a training-years-only regional seasonal baseline. It is a historical estimate for the study regions, not a yield or forecast model. Report metrics honestly; do not present a predictive capability unless it beats the baselines on held-out years.

## Optional LLM (Gemini free tier or OpenAI)

The LLM is optional and only rewrites the answer constructed from NASA and plant-profile evidence. It does not search for facts, invent crop advice, or change model values. Gemini is the suggested no-cost-to-start option; Google currently offers a free tier with model-specific usage limits. Free-tier content may be used to improve Google's products. Avoid entering personal or sensitive information. Review [current Gemini API pricing and limits](https://ai.google.dev/gemini-api/docs/pricing).

Create a Gemini API key in [Google AI Studio](https://aistudio.google.com/apikey). For local use, create `.streamlit/secrets.toml` (this file is ignored by Git):

```toml
GEMINI_API_KEY = "your-key"
GEMINI_MODEL = "gemini-2.5-flash-lite"
```

For Streamlit Community Cloud, open **Manage app → Settings → Secrets** and add the same two lines. Then select **Gemini (free tier)** and enable **Use LLM to phrase responses** in the app.

OpenAI remains available if configured; it uses the Responses API and may incur charges:

```toml
OPENAI_API_KEY = "your-key"
OPENAI_MODEL = "gpt-6-astra"
```

Never commit the key to GitHub. Without a key, the app keeps using its deterministic evidence-based answers.

## Run the app

```powershell
.venv\Scripts\streamlit.exe run app/streamlit_app.py
```

The text box accepts plain-language questions, but answers are limited to the loaded evidence. The LLM is not a substitute for local agronomic advice.

## Plant profile sources

The structured profiles in `data/knowledge/plant_profiles.json` cite [FAO EcoCrop tomato](https://ecocrop.apps.fao.org/ecocrop/srv/en/dataSheet?id=1379), [wheat](https://ecocrop.apps.fao.org/ecocrop/srv/en/dataSheet?id=2114), and [maize](https://ecocrop.apps.fao.org/ecocrop/srv/en/dataSheet?id=2175). The displayed temperature ranges are EcoCrop screening ranges, not local variety-specific setpoints. Annual rainfall ranges are not crop water requirements, particularly for irrigated farming.

## Deployment

Push the app code and required runtime artifacts to the GitHub branch connected to Streamlit Community Cloud. For the model panel, include the trained `models/` artifacts and `data/processed/ndvi_weather_features.csv`; those files are ignored by default and must be explicitly included for deployment. Configure `GEMINI_API_KEY` or `OPENAI_API_KEY` in Streamlit Secrets if using the LLM.

## Limitations

- Study coordinates are rough example points, not verified farm locations.
- NASA POWER is regional data; it does not observe an individual field.
- The MODIS label measures vegetation greenness, not crop yield.
- Profile ranges are globally sourced and not locally validated for Egypt.
- The model does not predict yield, recommend crops, schedule irrigation, diagnose pests/disease, or support arbitrary coordinates.
- The LLM is constrained to explain supplied evidence and can still phrase that evidence imperfectly; verify important decisions independently.
