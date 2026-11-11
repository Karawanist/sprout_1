"""Train Sprout's NDVI anomaly model when processed inputs are available."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from src.features import FEATURES, TARGET, add_ndvi_anomaly, time_split

def main():
    path = Path("data/processed/ndvi_weather_features.csv")
    if not path.exists():
        raise SystemExit(f"Missing {path}. Add POWER and AppEEARS CSVs, then run python -m src.prepare_data.")
    df, clim = add_ndvi_anomaly(pd.read_csv(path, parse_dates=["date"]), train_end_year=2017)
    train, val, test = time_split(df)
    if min(len(train), len(val), len(test)) == 0:
        raise SystemExit(f"Not enough date coverage for the configured splits (rows: {len(train)}, {len(val)}, {len(test)}).")
    X, y = train[FEATURES], train[TARGET]
    dummy = np.full(len(val), y.mean())
    print(f"Rows train/val/test: {len(train)}/{len(val)}/{len(test)}; dummy validation MAE: {mean_absolute_error(val[TARGET], dummy):.4f}")
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X, y)
    print(f"Ridge validation MAE={mean_absolute_error(val[TARGET],ridge.predict(val[FEATURES])):.4f}")
    model = RandomForestRegressor(n_estimators=300,min_samples_leaf=5,random_state=42,n_jobs=-1).fit(X,y)
    val_mae = mean_absolute_error(val[TARGET],model.predict(val[FEATURES]))
    print(f"Random Forest validation MAE={val_mae:.4f} R2={r2_score(val[TARGET],model.predict(val[FEATURES])):.3f}")
    print(f"Random Forest test MAE={mean_absolute_error(test[TARGET],model.predict(test[FEATURES])):.4f} R2={r2_score(test[TARGET],model.predict(test[FEATURES])):.3f}")
    print("Feature importance:\n",pd.Series(model.feature_importances_,index=FEATURES).sort_values(ascending=False))
    Path("models").mkdir(exist_ok=True); clim.to_csv("models/ndvi_climatology.csv",index=False)
    joblib.dump({"model":model,"features":FEATURES,"target":TARGET,"train_end_year":2017,"val_mae":float(val_mae)},"models/sprout_rf_v1.joblib",compress=3)

if __name__ == "__main__": main()
