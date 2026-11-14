# 🌱 Sprout

Sprout is an Egypt-focused agricultural decision-support prototype. It combines NASA POWER weather, FAO EcoCrop plant profiles, and a MODIS NDVI anomaly model. It supports decisions; it does not guarantee outcomes.

## What it can do

- Fetch recent weather from the NASA POWER API for five example Nile Delta and Nile Valley study points.
- Download historical weather, including precipitation and root-zone wetness, for model preparation.
- Compare recent mean air temperature with FAO EcoCrop's global ecological ranges for tomato, wheat, and maize. Profiles are source-cited but not validated for Egyptian cultivars, fields, or management.
- Train a Random Forest to estimate historical vegetation greenness anomaly from the preceding 32 days of weather, after authentic MODIS NDVI observations are supplied.
- Use the chat box to answer from the available weather, profile, and model evidence. A local Ollama model or optional OpenAI can phrase answers using only that evidence.

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

## Optional local LLM with Ollama

The LLM is optional and only rewrites the answer constructed from NASA and plant-profile evidence. It does not search for facts, invent crop advice, or change model values. Ollama runs the model on the computer where Sprout is running, so no API key or per-request API billing is needed.

Install [Ollama for Windows](https://ollama.com/download), then open PowerShell and download a small starter model:

```powershell
ollama pull llama3.2:3b
ollama list
```

Keep the Ollama app/server running, start Sprout, then select **Ollama (local)** and enable **Use LLM to phrase responses**. The defaults are `llama3.2:3b` and `http://localhost:11434`. Optional overrides can be set in `.streamlit/secrets.toml`:

```toml
OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_BASE_URL = "http://localhost:11434"
```

This local setup is for a Sprout process running on the same computer as Ollama. A Streamlit Community Cloud app cannot reach Ollama running on your personal computer. Each person can run Sprout and Ollama locally, or you would need to host an Ollama server somewhere accessible to the app.

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

## Data and reference sources

### 🛰️ NASA sources

- [NASA POWER Daily API](https://power.larc.nasa.gov/docs/services/api/temporal/daily/) — recent live and historical daily weather for the five example study points. Sprout requests maximum/minimum air temperature, corrected precipitation, all-sky surface shortwave radiation, and root-zone wetness.
- [NASA POWER hydrology data notes](https://power.larc.nasa.gov/docs/methodology/data/sources_hydrology/) — background and resolution notes for the `GWETROOT` root-zone wetness field.
- [NASA AppEEARS](https://appeears.earthdatacloud.nasa.gov/) — point-sample service used to obtain the optional historical satellite CSV.
- [NASA/USGS MODIS MOD13Q1.061 product](https://lpdaac.usgs.gov/products/mod13q1v061/) — 250 m, 16-day vegetation indices. Sprout uses NDVI and pixel-reliability information when preparing its historical model data.

### 🌾 Plant profile source

Structured profiles in `data/knowledge/plant_profiles.json` cite [FAO EcoCrop tomato](https://ecocrop.apps.fao.org/ecocrop/srv/en/dataSheet?id=1379), [common wheat](https://ecocrop.apps.fao.org/ecocrop/srv/en/dataSheet?id=2114), and [maize](https://ecocrop.apps.fao.org/ecocrop/srv/en/dataSheet?id=2175). These are global screening ranges, not locally validated recommendations or variety-specific setpoints. Annual rainfall ranges are not crop water requirements, particularly for irrigated farming.

The five study coordinates are project example points stored in `data/raw/study_points.csv`, not an externally validated set of farm locations. The optional language model is [Ollama](https://ollama.com/) running `llama3.2:3b` locally; it only rephrases evidence supplied by Sprout and is not a source of agronomic or weather facts.

## Deployment

Push the app code and required runtime artifacts to the GitHub branch connected to Streamlit Community Cloud. For the model panel, include the trained `models/` artifacts and `data/processed/ndvi_weather_features.csv`; those files are ignored by default and must be explicitly included for deployment. The Ollama option only works in deployments that can reach an Ollama server; configure `OPENAI_API_KEY` in Streamlit Secrets if using OpenAI.

## Limitations

- Study coordinates are rough example points, not verified farm locations.
- NASA POWER is regional data; it does not observe an individual field.
- The MODIS label measures vegetation greenness, not crop yield.
- Profile ranges are globally sourced and not locally validated for Egypt.
- The model does not predict yield, recommend crops, schedule irrigation, diagnose pests/disease, or support arbitrary coordinates.
- The LLM is constrained to explain supplied evidence and can still phrase that evidence imperfectly; verify important decisions independently.
