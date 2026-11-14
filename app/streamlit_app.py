"""Streamlit interface for the CSV-first Sprout prototype."""
import sys
import os
from datetime import date, timedelta
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import pandas as pd
import streamlit as st
from src.knowledge import load_profiles, compare_to_profile
from src.predict import predict_with_range
from src.data_collection import REGIONS
from src.power_api import fetch_region_power_daily
from src.llm import rephrase_evidence, rephrase_with_ollama

st.set_page_config(page_title="Sprout", page_icon="🌱", layout="wide")
st.title("🌱 Sprout")
st.caption("Agricultural decision-support prototype using live NASA POWER data. It supports decisions; it does not guarantee outcomes.")
st.info("NDVI is a satellite measure of vegetation greenness. It is not yield, crop type, irrigation need, or a field-level diagnosis.")

try:
    profiles = load_profiles(ROOT / "data/knowledge/plant_profiles.json")
except (OSError, ValueError) as exc:
    st.error(f"Plant profile data could not be loaded: {exc}"); st.stop()
weather_path = ROOT / "data/processed/power_daily_regions_clean.csv"
try:
    local_weather = pd.read_csv(weather_path, parse_dates=["date"])
except (FileNotFoundError, pd.errors.EmptyDataError, ValueError):
    local_weather = pd.DataFrame()

with st.sidebar:
    st.header("Study region")
    region = st.selectbox("Region", sorted(REGIONS))
    days = st.slider("Recent period (days)", 14, 90, 30)
    crop = st.selectbox("Plant profile", list(profiles), format_func=lambda key: profiles[key]["display_name"])

@st.cache_data(ttl=21600, show_spinner=False)
def get_power_data(selected_region, first_day, last_day):
    return fetch_region_power_daily(selected_region, first_day, last_day)

end_day = date.today() - timedelta(days=2)
start_day = end_day - timedelta(days=days - 1)
api_warnings = []
try:
    weather, api_warnings = get_power_data(region, start_day, end_day)
    weather_source = "live NASA POWER API"
except Exception:
    weather_source = "saved NASA POWER CSV fallback"
    if not local_weather.empty and "region" in local_weather:
        weather = local_weather[local_weather.region == region].copy()
        weather = weather[weather.date >= pd.Timestamp(start_day)]
    else:
        weather = pd.DataFrame()
    st.warning("The NASA POWER API is unavailable right now; Sprout will use saved weather data if available.")
for warning in api_warnings:
    st.caption(warning)

if not weather.empty and "date" in weather:
    recent = weather.sort_values("date").tail(days).set_index("date")
else:
    recent = pd.DataFrame()
    st.warning("No weather observations are available for this region and period. The plant profile and model sections remain available.")
if not recent.empty:
    st.caption(f"Showing {len(recent)} daily observations through {recent.index.max().date()} from the {weather_source}.")
left, right = st.columns(2)
with left:
    st.subheader("NASA POWER observations")
    available = [c for c in ["T2M_MAX", "T2M_MIN"] if c in recent]
    if not recent.empty and available:
        st.line_chart(recent[available])
    else:
        st.caption("Temperature observations are unavailable.")
    if "PRECTOTCORR" in recent and recent.PRECTOTCORR.notna().any():
        st.bar_chart(recent["PRECTOTCORR"])
        st.write(f"Total precipitation: **{recent.PRECTOTCORR.sum():.1f} mm**.")
    else:
        st.caption("Precipitation is unavailable for this period.")
    if "GWETROOT" in recent and recent.GWETROOT.notna().any():
        st.write(f"Average root-zone soil wetness index: **{recent.GWETROOT.mean():.2f}** (0 = dry, 1 = wet).")
    else:
        st.caption("Root-zone wetness is unavailable for this period.")
with right:
    st.subheader(f"Profile comparison: {profiles[crop]['display_name']}")
    for note in compare_to_profile(recent.reset_index(), profiles[crop]): st.write(f"• {note}")
    profile = profiles[crop]
    st.write(f"Scientific name: *{profile.get('scientific_name', 'not listed')}*")
    if profile.get("optimal_mean_temp_c"):
        st.write(f"FAO EcoCrop optimal ecological temperature range: {profile['optimal_mean_temp_c'][0]}–{profile['optimal_mean_temp_c'][1]} °C.")
    st.caption("These global profile ranges are not locally validated crop recommendations; rainfall ranges are not irrigation requirements.")
    with st.expander(f"More {profile['display_name']} profile details"):
        if profile.get("annual_rainfall_optimal_mm"):
            lo, hi = profile["annual_rainfall_optimal_mm"]
            st.write(f"FAO EcoCrop annual rainfall screening range: {lo}–{hi} mm/year. This is not an irrigation requirement.")
        if profile.get("soil_ph_optimal"):
            lo, hi = profile["soil_ph_optimal"]
            st.write(f"Soil pH screening range: {lo}–{hi}.")
        soil_values = [f"{key.replace('_', ' ').title()}: {value}" for key, value in profile.get("soil", {}).items() if value]
        if soil_values:
            st.write("Soil profile: " + "; ".join(soil_values) + ".")
        if profile.get("crop_cycle_days"):
            lo, hi = profile["crop_cycle_days"]
            st.write(f"Listed crop-cycle range: {lo}–{hi} days; actual cultivar and season vary.")
        for note in profile.get("notes", []):
            st.write(f"• {note}")
    sources = profile.get("sources", [])
    if sources:
        st.markdown("**Profile source**")
        for source in sources:
            st.markdown(f"- [{source['label']}]({source['url']})")

st.subheader("Model: vegetation greenness vs. normal")
model_path, table_path, clim_path = ROOT / "models/sprout_rf_v1.joblib", ROOT / "data/processed/ndvi_weather_features.csv", ROOT / "models/ndvi_climatology.csv"
bundle = table = clim = None
if not all(p.exists() for p in (model_path, table_path, clim_path)):
    st.info("The model panel will appear after data preparation and successful training. It only supports historical periods in the study regions.")
else:
    try:
        bundle = joblib.load(model_path)  # Load only a model artifact created by this project.
        table = pd.read_csv(table_path, parse_dates=["date"])
        clim = pd.read_csv(clim_path)
        subset = table[table.region == region].sort_values("date")
        if subset.empty:
            st.info("No joined NDVI periods are available for this region.")
        else:
            selected = st.select_slider("Historical composite start", options=subset.date.dt.date.tolist())
            row = subset[subset.date.dt.date == selected].iloc[0]
            X = row[bundle["features"]].to_frame().T.astype(float)
            mean, low, high = predict_with_range(bundle["model"], X)
            normal = clim[(clim.region == region) & (clim.doy == row.doy)]
            if normal.empty: st.warning("No training climatology is available for this period.")
            else:
                actual = row.ndvi - normal.ndvi_clim.iloc[0]
                st.write(f"Sprout estimates **{mean[0]:+.3f} NDVI** vs. normal (tree disagreement range {low[0]:+.3f} to {high[0]:+.3f}).")
                st.write(f"NASA MODIS observed **{actual:+.3f} NDVI** vs. normal.")
                st.caption("Positive values mean greener than the region's usual level for that time of year. The tree range shows model disagreement, not a formal confidence interval.")
    except (OSError, ValueError, KeyError, IndexError) as exc:
        st.warning(f"Model results are unavailable: {exc}")

with st.expander("What Sprout cannot tell you"):
    st.write("This prototype does not predict yield, crop suitability, irrigation schedules, pests, disease, field conditions, or locations beyond its Egyptian study regions. NASA POWER is coarse regional data, and most study farmland is irrigated, so weather may explain only a modest share of greenness changes. Consult a local agricultural advisor.")

with st.expander("Data and reference sources"):
    st.markdown("### 🛰️ NASA sources")
    st.info("NASA datasets power Sprout's weather charts and, when MODIS data is supplied, its historical vegetation-greenness model.")
    st.markdown(
        "- [NASA POWER Daily API](https://power.larc.nasa.gov/docs/services/api/temporal/daily/) — recent live and historical daily weather at the five example study points. Sprout requests maximum/minimum air temperature, corrected precipitation, all-sky surface shortwave radiation, and root-zone wetness. "
        "\n- [NASA POWER hydrology data notes](https://power.larc.nasa.gov/docs/methodology/data/sources_hydrology/) — background and resolution notes for `GWETROOT` root-zone wetness. "
        "\n- [NASA AppEEARS](https://appeears.earthdatacloud.nasa.gov/) — point-sample service used to obtain the optional historical satellite CSV. "
        "\n- [NASA/USGS MODIS MOD13Q1.061 product](https://lpdaac.usgs.gov/products/mod13q1v061/) — 250 m, 16-day vegetation indices; Sprout uses NDVI and pixel-reliability information when preparing model data."
    )
    st.markdown("### 🌾 Plant profile source")
    st.markdown(
        "- [FAO EcoCrop tomato](https://ecocrop.apps.fao.org/ecocrop/srv/en/dataSheet?id=1379) "
        "\n- [FAO EcoCrop common wheat](https://ecocrop.apps.fao.org/ecocrop/srv/en/dataSheet?id=2114) "
        "\n- [FAO EcoCrop maize](https://ecocrop.apps.fao.org/ecocrop/srv/en/dataSheet?id=2175)"
    )
    st.caption("EcoCrop ranges are global screening references, not locally validated crop recommendations. Study-point coordinates are example points listed in data/raw/study_points.csv.")
    st.markdown("### 💬 Local language model")
    st.markdown("- [Ollama](https://ollama.com/) runs the optional `llama3.2:3b` phrasing model on the computer hosting Sprout. The model only rewrites the data-backed answer; it is not a source for weather or crop-profile facts.")

st.subheader("Ask Sprout")
st.caption("Ask about recent conditions, sourced plant profiles, or supported historical greenness. The optional LLM only rephrases Sprout's computed evidence; it does not provide independent agricultural facts.")
try:
    ollama_model = st.secrets.get("OLLAMA_MODEL", "llama3.2:3b") or os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
    ollama_base_url = st.secrets.get("OLLAMA_BASE_URL", "http://localhost:11434") or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    openai_key = st.secrets.get("OPENAI_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
    openai_model = st.secrets.get("OPENAI_MODEL", "gpt-6-astra") or os.environ.get("OPENAI_MODEL", "gpt-6-astra")
except Exception:
    ollama_model = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
    ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    openai_model = os.environ.get("OPENAI_MODEL", "gpt-6-astra")
providers = ["Ollama (local)"]
if openai_key:
    providers.append("OpenAI")
selected_provider = st.selectbox("Optional LLM provider", providers, index=0)
use_llm = bool(selected_provider and st.checkbox("Use LLM to phrase responses", value=False))
if selected_provider == "Ollama (local)":
    st.caption(f"Runs locally through Ollama ({ollama_model}); no API key or per-request API charge. Ollama must be running on the same computer as this app.")
    st.markdown("New to Ollama? [Download Ollama](https://ollama.com/download), install it, then run `ollama pull llama3.2:3b` in PowerShell. Each person running Sprout locally needs Ollama installed on their computer.")
elif selected_provider == "OpenAI":
    st.caption("When enabled, Sprout sends your question and computed answer evidence to OpenAI. Avoid personal or sensitive information; API usage may incur charges.")
if "sprout_messages" not in st.session_state:
    st.session_state.sprout_messages = []
for message in st.session_state.sprout_messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def answer_question(question: str) -> str:
    q = question.lower()
    unsupported = ("yield", "harvest", "irrigat", "water schedule", "disease", "pest", "grow ", "suitable", "recommend a crop")
    if any(term in q for term in unsupported):
        return ("I can't answer that with this prototype. It does not predict yield, crop suitability, irrigation needs, pests, or disease. "
                "I can summarize NASA POWER conditions or, when trained data is available, estimate historical vegetation greenness versus normal.")

    chosen_region = region
    for name in REGIONS:
        readable = name.replace("_", " ")
        if name in q or readable in q:
            chosen_region = name
            break
    if weather.empty or not {"region", "date"}.issubset(weather.columns):
        return "I don't have live or saved NASA POWER observations right now. Please try again when the data source is available."
    subset_weather = weather[weather.region == chosen_region].sort_values("date")
    if subset_weather.empty:
        return f"I don't currently have live or saved weather observations for {chosen_region}. Check the NASA POWER connection and try again."

    crop_key = crop
    for key, profile in profiles.items():
        if key in q or profile["display_name"].lower() in q:
            crop_key = key
            break
    asks_greenness = any(term in q for term in ("ndvi", "green", "greenness", "anomaly", "model estimate"))
    if asks_greenness:
        if bundle is None or table is None or clim is None:
            return "The greenness model isn't ready: it needs MODIS NDVI data and a successfully trained model. Weather-only data can't train this target."
        import re
        match = re.search(r"\b\d{4}-\d{2}-\d{2}\b", question)
        subset = table[table.region == chosen_region].sort_values("date")
        if subset.empty:
            return f"There are no joined MODIS weather and NDVI observations for {chosen_region}."
        if match:
            asked_date = pd.Timestamp(match.group(0))
            candidates = subset[subset.date <= asked_date]
            if candidates.empty:
                return f"I don't have a trained historical composite on or before {asked_date.date()} for {chosen_region}."
            row = candidates.iloc[-1]
        else:
            row = subset.iloc[-1]
        X = row[bundle["features"]].to_frame().T.astype(float)
        mean, low, high = predict_with_range(bundle["model"], X)
        normal = clim[(clim.region == chosen_region) & (clim.doy == row.doy)]
        if normal.empty:
            return "The model has no training climatology for this composite date."
        observed = row.ndvi - normal.ndvi_clim.iloc[0]
        return (f"For {chosen_region.replace('_', ' ')} on the historical composite starting {row.date.date()}, "
                f"Sprout estimates vegetation greenness at {mean[0]:+.3f} NDVI versus normal "
                f"(tree disagreement range {low[0]:+.3f} to {high[0]:+.3f}). NASA MODIS observed {observed:+.3f}. "
                "NDVI is greenness, not yield; the range is not a formal confidence interval.")

    latest = subset_weather.tail(days)
    summary = (f"NASA POWER shows {len(latest)} available daily observations for {chosen_region.replace('_', ' ')} "
               f"through {latest.date.max().date()}.")
    if "T2M_MAX" in latest:
        summary += f" Average daily high: {latest.T2M_MAX.mean():.1f} °C."
    if "PRECTOTCORR" in latest and latest.PRECTOTCORR.notna().any():
        summary += f" Total precipitation: {latest.PRECTOTCORR.sum():.1f} mm."
    else:
        summary += " Precipitation is unavailable in the current NASA POWER response."
    if any(term in q for term in ("plant", "crop", "tomato", "wheat", "maize", "profile")):
        notes = compare_to_profile(latest, profiles[crop_key])
        summary += " Profile notes for " + profiles[crop_key]["display_name"] + ": " + " ".join(notes)
        profile = profiles[crop_key]
        if profile.get("optimal_mean_temp_c"):
            lo, hi = profile["optimal_mean_temp_c"]
            summary += f" FAO EcoCrop ecological mean-temperature range: {lo}–{hi} °C."
        if profile.get("annual_rainfall_optimal_mm"):
            lo, hi = profile["annual_rainfall_optimal_mm"]
            summary += f" FAO EcoCrop annual rainfall screening range: {lo}–{hi} mm/year; this is not irrigation guidance."
        if profile.get("soil_ph_optimal"):
            lo, hi = profile["soil_ph_optimal"]
            summary += f" FAO EcoCrop soil pH screening range: {lo}–{hi}."
        if profile.get("notes"):
            summary += " " + " ".join(profile["notes"])
        source = profiles[crop_key].get("sources", [{}])[0]
        if source.get("url"):
            summary += f" Profile source: {source['label']} ({source['url']})."
        summary += " Profile data is global and not locally validated for Egyptian cultivars or management."
    summary += " POWER is regional data, not a measurement of an individual field."
    return summary

if question := st.chat_input("Ask about recent conditions or historical greenness…"):
    st.session_state.sprout_messages.append({"role": "user", "content": question})
    response = answer_question(question)
    if use_llm:
        try:
            if selected_provider == "Ollama (local)":
                response = rephrase_with_ollama(question, response, ollama_model, ollama_base_url)
            else:
                response = rephrase_evidence(question, response, openai_key, openai_model)
        except Exception:
            response += f"\n\n({selected_provider} phrasing is unavailable. For Ollama, start the Ollama app and ensure `{ollama_model}` is installed; this answer uses Sprout's rule-based response.)"
    st.session_state.sprout_messages.append({"role": "assistant", "content": response})
    st.rerun()
