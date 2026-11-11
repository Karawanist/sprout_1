"""Cleaning, weather-window features, and leakage-aware splits."""
import numpy as np
import pandas as pd

WEATHER_COLS = ["T2M_MAX", "T2M_MIN", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN", "GWETROOT"]
FEATURES = ["tmax_mean", "tmin_mean", "precip_sum", "solar_mean", "soil_wet_mean", "hot_days", "doy_sin", "doy_cos"]
TARGET = "ndvi_anomaly"

def clean_weather(weather):
    weather = weather.copy()
    supplied = set(weather.columns)
    weather["date"] = pd.to_datetime(weather["date"], errors="coerce")
    for col in WEATHER_COLS:
        if col not in weather:
            weather[col] = np.nan
        weather[col] = pd.to_numeric(weather[col], errors="coerce").replace(-999, np.nan)
    weather = weather.dropna(subset=["date"]).sort_values(["region", "date"])
    weather[WEATHER_COLS] = weather.groupby("region")[WEATHER_COLS].transform(lambda s: s.interpolate(limit=3))
    # Keep valid observations when a download omitted optional parameters.
    # Model preparation checks for all required parameters separately.
    valid = weather.T2M_MAX.between(-10, 60)
    if "PRECTOTCORR" in supplied:
        valid &= weather.PRECTOTCORR.isna() | weather.PRECTOTCORR.between(0, 300)
    if "GWETROOT" in supplied:
        valid &= weather.GWETROOT.isna() | weather.GWETROOT.between(0, 1)
    return weather.loc[valid].copy()

def load_ndvi_csv(path):
    raw = pd.read_csv(path)
    date_col = next((c for c in raw if c.lower() == "date"), None)
    region_col = next((c for c in raw if c.lower() in ("category", "region")), None)
    ndvi_col = next((c for c in raw if c.lower().endswith("ndvi") or c.lower() == "ndvi"), None)
    rel_col = next((c for c in raw if "reliability" in c.lower()), None)
    if not all((date_col, region_col, ndvi_col)):
        raise ValueError("NDVI CSV needs Date, Category/region, and an NDVI column. See data/raw/README.md.")
    out = raw.rename(columns={date_col:"date", region_col:"region", ndvi_col:"ndvi"})
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["ndvi"] = pd.to_numeric(out["ndvi"], errors="coerce")
    if out.ndvi.abs().max() > 2:
        out["ndvi"] *= 0.0001
    if rel_col:
        out["reliability"] = pd.to_numeric(out[rel_col], errors="coerce")
        out = out[out.reliability.isin([0, 1])]
    return out[out.ndvi.between(-0.2, 1.0)].dropna(subset=["date", "region", "ndvi"])[["region", "date", "ndvi"]]

def build_features(weather, ndvi, window=32):
    rows = []
    weather = weather.copy(); weather["date"] = pd.to_datetime(weather.date)
    ndvi = ndvi.copy(); ndvi["date"] = pd.to_datetime(ndvi.date)
    for region, obs in ndvi.groupby("region"):
        w = weather[weather.region == region].set_index("date").sort_index()
        for r in obs.itertuples():
            win = w.loc[r.date - pd.Timedelta(days=window):r.date - pd.Timedelta(days=1)]
            if len(win) < 0.8 * window:
                continue
            rows.append({"region":region,"date":r.date,"year":r.date.year,"doy":r.date.dayofyear,"ndvi":r.ndvi,
                         "tmax_mean":win.T2M_MAX.mean(),"tmin_mean":win.T2M_MIN.mean(),"precip_sum":win.PRECTOTCORR.sum(),
                         "solar_mean":win.ALLSKY_SFC_SW_DWN.mean(),"soil_wet_mean":win.GWETROOT.mean(),"hot_days":int((win.T2M_MAX > 35).sum())})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["doy_sin"] = np.sin(2*np.pi*df.doy/365.25); df["doy_cos"] = np.cos(2*np.pi*df.doy/365.25)
    return df

def add_ndvi_anomaly(df, train_end_year=2017):
    train = df[df.year <= train_end_year]
    clim = train.groupby(["region", "doy"]).ndvi.median().rename("ndvi_clim").reset_index()
    out = df.merge(clim, on=["region", "doy"], how="left")
    out["ndvi_anomaly"] = out.ndvi - out.ndvi_clim
    return out.dropna(subset=["ndvi_anomaly"]), clim

def time_split(df, train_end=2017, val_end=2020, gap_days=32):
    gap = pd.Timedelta(days=gap_days)
    a, b = pd.Timestamp(f"{train_end}-12-31"), pd.Timestamp(f"{val_end}-12-31")
    return (df[df.date <= a-gap].copy(), df[(df.date > a+gap)&(df.date <= b-gap)].copy(), df[df.date > b+gap].copy())
