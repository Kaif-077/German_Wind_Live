"""Build the small test fixtures from the research project: inputs for one German day plus the research code's own
features and predictions for that day (the oracle the live code must match).
    python tools/make_fixtures.py /path/to/Power_Market_Analytics
"""
import sys
from pathlib import Path

research = Path(sys.argv[1])
sys.path.insert(0, str(research / "src"))
sys.path.insert(0, "src")
import joblib
import pandas as pd
from pmp import data, features as F
from wfl.forecast import target_index

IDX = target_index("2026-09-28")  # 24 hours, 2026-09-27 22:00 UTC to 2026-09-28 21:00 UTC
out = Path("tests/fixtures"); out.mkdir(parents=True, exist_ok=True)
w = data.load_weather()
w = w[(w["time_utc"] >= IDX[0].floor("D") - pd.Timedelta(days=1)) & (w["time_utc"] < IDX[-1].floor("D") + pd.Timedelta(days=1))]
w[["time_utc", "location", "wind_speed_100m", "wind_direction_100m", "wind_gusts_10m", "temperature_2m", "surface_pressure"]].to_csv(out / "weather_long.csv", index=False)
df = F.build_dataset(data.load_weather(), data.load_actuals(), data.load_entsoe_forecast())
day = df.loc[IDX]
day["entsoe_fc"].rename("entsoe_mw").to_csv(out / "entsoe_forecast.csv", index_label="time_utc")
m = joblib.load("models/final_models.joblib")
exp = day[m["cols_hybrid"]].copy()
exp["pred_weather_model"] = m["weather_obs"].predict(day[m["cols_weather_obs"]]) * day["cap_proxy"]
exp["pred_hybrid"] = day["entsoe_fc"] + m["hybrid"].predict(day[m["cols_hybrid"]])
exp.to_csv(out / "expected.csv", index_label="time_utc")
print("fixtures written for German day 2026-09-28:", len(day), "hours")
