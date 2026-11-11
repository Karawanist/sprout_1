"""Streamlit interface for the CSV-first Sprout prototype."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import pandas as pd
import streamlit as st
from src.knowledge import load_profiles, compare_to_profile
from src.predict import predict_with_range
from src.data_collection import REGIONS

st.set_page_config(page_title="Sprout", page_icon="🌱", layout="wide")
st.title("🌱 Sprout")
st.caption("Agricultural decision-support prototype using downloaded NASA CSV data. It supports decisions; it does not guarantee outcomes.")
st.info("NDVI is a satellite measure of vegetation greenness. It is not yield, crop type, irrigation need, or a field-level diagnosis.")

try:
    profiles = load_profiles(ROOT / "data/knowledge/plant_profiles.json")
except (OSError, ValueError) as exc:
    st.error(f"Plant profile data could not be loaded: {exc}"); st.stop()
weather_path = ROOT / "data/processed/power_daily_regions_clean.csv"
try:
    weather = pd.read_csv(weather_path, parse_dates=["date"])
except FileNotFoundError:
    st.warning("NASA POWER data is not ready yet. Add the regional CSVs described in `data/raw/README.md`, then run `python -m src.prepare_data`.")
    st.stop()
if weather.empty or "region" not in weather:
    st.error("The processed weather file has no usable regional rows. Check the downloaded CSVs and rerun preparation."); st.stop()

with st.sidebar:
    st.header("Study region")
    region = st.selectbox("Region", sorted(weather.region.dropna().unique()))
    days = st.slider("Recent period (days)", 14, 90, 30)
    crop = st.selectbox("Plant profile", list(profiles), format_func=lambda key: profiles[key]["display_name"])

recent = weather[weather.region == region].sort_values("date").tail(days).set_index("date")
if recent.empty:
    st.warning("No weather observations are available for this region."); st.stop()
st.caption(f"Showing {len(recent)} available daily observations through {recent.index.max().date()}. Update by downloading newer NASA POWER CSVs.")
left, right = st.columns(2)
with left:
    st.subheader("NASA POWER observations")
    available = [c for c in ["T2M_MAX", "T2M_MIN"] if c in recent]
    if available: st.line_chart(recent[available])
    if "PRECTOTCORR" in recent: st.bar_chart(recent["PRECTOTCORR"])
    if "PRECTOTCORR" in recent: st.write(f"Total precipitation: **{recent.PRECTOTCORR.sum():.1f} mm**.")
    if "GWETROOT" in recent: st.write(f"Average root-zone soil wetness index: **{recent.GWETROOT.mean():.2f}** (0 = dry, 1 = wet).")
with right:
    st.subheader(f"Profile comparison: {profiles[crop]['display_name']}")
    for note in compare_to_profile(recent.reset_index(), profiles[crop]): st.write(f"• {note}")
    sources = profiles[crop].get("sources", [])
    if sources: st.caption("Profile sources: " + "; ".join(sources))

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

st.subheader("Ask Sprout")
st.caption("Ask about recent conditions or a supported historical greenness estimate. Sprout uses fixed, transparent rules here; it does not provide open-ended agricultural advice.")
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
    subset_weather = weather[weather.region == chosen_region].sort_values("date")
    if subset_weather.empty:
        return f"I don't have weather observations for {chosen_region}. Choose one of the five study regions."

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
    if "PRECTOTCORR" in latest:
        summary += f" Total precipitation: {latest.PRECTOTCORR.sum():.1f} mm."
    if any(term in q for term in ("plant", "crop", "tomato", "wheat", "maize", "profile")):
        notes = compare_to_profile(latest, profiles[crop_key])
        summary += " Profile notes for " + profiles[crop_key]["display_name"] + ": " + " ".join(notes)
    summary += " POWER is regional data, not a measurement of an individual field."
    return summary

if question := st.chat_input("Ask about recent conditions or historical greenness…"):
    st.session_state.sprout_messages.append({"role": "user", "content": question})
    response = answer_question(question)
    st.session_state.sprout_messages.append({"role": "assistant", "content": response})
    st.rerun()
