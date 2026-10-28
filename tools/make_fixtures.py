"""Build the small test fixtures from the research project: inputs for one day plus the research code's own
features and predictions for that day (the oracle the live code must match).
    python tools/make_fixtures.py /path/to/Power_Market_Analytics
"""
import sys
from pathlib import Path

research = Path(sys.argv[1])
sys.path.insert(0, str(research / "src"))
import joblib
import pandas as pd
from pmp import data, features as F

DAY = pd.Timestamp("2026-09-28", tz="UTC")
out = Path("tests/fixtures"); out.mkdir(parents=True, exist_ok=True)
w = data.load_weather()
w = w[(w["time_utc"] >= DAY - pd.Timedelta(days=1)) & (w["time_utc"] < DAY + pd.Timedelta(days=1))]
w[["time_utc", "location", "wind_speed_100m", "wind_direction_100m", "wind_gusts_10m", "temperature_2m", "surface_pressure"]].to_csv(out / "weather_long.csv", index=False)
df = F.build_dataset(data.load_weather(), data.load_actuals(), data.load_entsoe_forecast())
day = df[(df.index >= DAY) & (df.index < DAY + pd.Timedelta(days=1))]
day["entsoe_fc"].rename("entsoe_mw").to_csv(out / "entsoe_forecast.csv", index_label="time_utc")
m = joblib.load("models/final_models.joblib")
exp = day[m["cols_hybrid"]].copy()
exp["pred_weather_model"] = m["weather_obs"].predict(day[m["cols_weather_obs"]]) * day["cap_proxy"]
exp["pred_hybrid"] = day["entsoe_fc"] + m["hybrid"].predict(day[m["cols_hybrid"]])
exp.to_csv(out / "expected.csv", index_label="time_utc")
print("fixtures written for", DAY.date(), len(day), "hours")
