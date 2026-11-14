"""Load downloaded NASA POWER CSV files and combine study regions."""
from pathlib import Path
import pandas as pd

REGIONS = {
    "delta_gharbia": (30.85, 31.05),
    "delta_menoufia": (30.50, 30.95),
    "delta_sharqia": (30.65, 31.75),
    "middle_egypt_minya": (28.10, 30.75),
    "upper_egypt_sohag": (26.55, 31.70),
}

def load_power_csv(path):
    """Read a POWER CSV, including its optional metadata header."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    end = next((i for i, line in enumerate(lines) if "-END HEADER-" in line), -1)
    # The NASA metadata header is skipped explicitly. Do not use pandas'
    # comment option here: it would truncate ISO dates at their hyphens.
    df = pd.read_csv(path, skiprows=end + 1 if end >= 0 else 0, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]
    if {"YEAR", "MO", "DY"}.issubset(df.columns):
        df["date"] = pd.to_datetime(dict(year=df.YEAR, month=df.MO, day=df.DY), errors="coerce")
    elif {"YEAR", "DOY"}.issubset(df.columns):
        doy = pd.to_numeric(df.DOY, errors="coerce").astype("Int64").astype(str).str.zfill(3)
        df["date"] = pd.to_datetime(df.YEAR.astype(str) + doy, format="%Y%j", errors="coerce")
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        raise ValueError(f"No date columns found in {path.name}: {df.columns.tolist()}")
    df = df.drop(columns=[c for c in ["YEAR", "MO", "DY", "DOY"] if c in df])
    for col in df.columns:
        if col != "date":
            df[col] = pd.to_numeric(df[col], errors="coerce").replace(-999, pd.NA)
    return df.dropna(subset=["date"])

def combine_power_files(raw_dir="data/raw", output="data/processed/power_daily_regions.csv"):
    frames = []
    for region in REGIONS:
        api_path = Path(raw_dir) / "api_power" / f"power_{region}.csv"
        manual_path = Path(raw_dir) / f"power_{region}.csv"
        path = api_path if api_path.exists() else manual_path
        if path.exists():
            frame = load_power_csv(path)
            frame["region"] = region
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No power_<region>.csv files found in {raw_dir}")
    result = pd.concat(frames, ignore_index=True).sort_values(["region", "date"])
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return result
