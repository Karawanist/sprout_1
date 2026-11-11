"""Diagnostic leave-one-region-out validation for the anomaly model.

This answers whether the weather relationship transfers to a held-out region.
It is a diagnostic rather than an unbiased final score because the simple
climatology in the input feature table was estimated across training years.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from src.features import FEATURES, TARGET, add_ndvi_anomaly

def main():
    path = Path("data/processed/ndvi_weather_features.csv")
    if not path.exists():
        raise SystemExit(f"Missing {path}; prepare real source CSVs first.")
    df, _ = add_ndvi_anomaly(pd.read_csv(path, parse_dates=["date"]), train_end_year=2017)
    results = []
    for region in sorted(df.region.unique()):
        train, held = df[df.region != region], df[df.region == region]
        if train.empty or held.empty:
            continue
        model = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=0, n_jobs=-1)
        model.fit(train[FEATURES], train[TARGET])
        prediction = model.predict(held[FEATURES])
        results.append({"held_out_region":region,"rows":len(held),"rf_mae":mean_absolute_error(held[TARGET],prediction),
                        "zero_anomaly_mae":mean_absolute_error(held[TARGET],np.zeros(len(held)))})
    if not results:
        raise SystemExit("Could not form a leave-one-region-out fold.")
    print(pd.DataFrame(results).to_string(index=False))
    print("Diagnostic only: report these results with the stated climatology limitation in the module docstring.")

if __name__ == "__main__": main()
