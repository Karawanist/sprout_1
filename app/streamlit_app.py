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
