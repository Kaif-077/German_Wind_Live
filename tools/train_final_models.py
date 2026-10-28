"""Train the two live models once on ALL research data (2024-02-17 to 2026-09-30) and seed the history file.
Needs the research project (Power_Market_Analytics) and its raw data. Usage:
    python tools/train_final_models.py /path/to/Power_Market_Analytics
Outputs: models/final_models.joblib, history/actual_wind.csv
"""
import sys
from pathlib import Path

research = Path(sys.argv[1])
sys.path.insert(0, str(research / "src"))
import joblib
import pandas as pd
from pmp import data, features as F
from pmp.models import make_gbm

df = F.build_dataset(data.load_weather(), data.load_actuals(), data.load_entsoe_forecast())
fs = F.feature_sets(df)
ok = df["y"].notna()

# 1) weather + recent-observation model: predicts capacity factor (generation / capacity proxy)
cols_w = fs["weather_plus_obs"]
m_w = make_gbm(max_leaf_nodes=15).fit(df.loc[ok, cols_w], (df["y"] / df["cap_proxy"])[ok])
# 2) hybrid: learns the correction to ENTSO-E's day-ahead forecast
cols_h = fs["hybrid_with_entsoe"]
ok_h = ok & df["entsoe_fc"].notna()
m_h = make_gbm(max_leaf_nodes=7, max_iter=200, learning_rate=0.03, min_samples_leaf=100).fit(
    df.loc[ok_h, cols_h], (df["y"] - df["entsoe_fc"])[ok_h])

Path("models").mkdir(exist_ok=True)
joblib.dump({"weather_obs": m_w, "hybrid": m_h, "cols_weather_obs": cols_w, "cols_hybrid": cols_h,
             "trained_through": str(df.index.max()), "n_train_weather_obs": int(ok.sum()), "n_train_hybrid": int(ok_h.sum())},
            "models/final_models.joblib", compress=3)

act = data.load_actuals()["wind_total"].dropna()
act = act[act.index >= act.index.max() - pd.Timedelta(days=500)]
Path("history").mkdir(exist_ok=True)
act.rename("wind_total_mw").to_csv("history/actual_wind.csv", index_label="time_utc")
print("trained through", df.index.max(), "| rows", int(ok.sum()), int(ok_h.sum()), "| history rows", len(act), act.index.min(), act.index.max())
