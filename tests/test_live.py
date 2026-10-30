"""Offline tests: the live code must reproduce the research code exactly, and the daily runner must behave.
Run with:  python tests/test_live.py"""
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from run_daily import run
from wfl.forecast import build_features, load_models, make_forecast, target_index

FX = ROOT / "tests" / "fixtures"
IDX = target_index("2026-09-28")  # a German day: 2026-09-27 22:00 UTC to 2026-09-28 21:00 UTC
assert len(IDX) == 24 and len(target_index("2026-10-25")) == 25 and len(target_index("2026-03-29")) == 23  # clock-change days
models = load_models(ROOT / "models" / "final_models.joblib")
hist = pd.read_csv(ROOT / "history" / "actual_wind.csv", index_col=0, parse_dates=True, float_precision="round_trip").iloc[:, 0]
hist.index = pd.to_datetime(hist.index, utc=True)
weather = pd.read_csv(FX / "weather_long.csv", parse_dates=["time_utc"], float_precision="round_trip")
entsoe = pd.read_csv(FX / "entsoe_forecast.csv", index_col=0, parse_dates=True, float_precision="round_trip").iloc[:, 0]
entsoe.index = pd.to_datetime(entsoe.index, utc=True)
exp = pd.read_csv(FX / "expected.csv", index_col=0, parse_dates=True, float_precision="round_trip")
exp.index = pd.to_datetime(exp.index, utc=True)

# 1) same features and same predictions as the research code
out, X = make_forecast(IDX, weather, hist, entsoe, models)
pd.testing.assert_frame_equal(X[models["cols_weather_obs"]], exp[models["cols_weather_obs"]], check_exact=False, atol=1e-6, rtol=1e-9, check_names=False, check_freq=False, check_dtype=False)
assert np.allclose(out["weather_model_mw"], exp["pred_weather_model"], atol=1e-3)
assert np.allclose(out["hybrid_mw"], exp["pred_hybrid"], atol=1e-3)

# 2) without ENTSO-E there is a weather-model forecast and no hybrid
out2, _ = make_forecast(IDX, weather, hist, None, models)
assert out2["hybrid_mw"].isna().all() and out2["weather_model_mw"].notna().all()

# 3) future actuals must not change the forecast (scramble everything after the information cut-off)
cut = IDX[-1].floor("D") - pd.Timedelta(hours=25)  # the latest information cut-off among the rows
scr = hist.copy(); scr[scr.index > cut] = np.random.default_rng(0).uniform(0, 50000, int((scr.index > cut).sum()))
out3, _ = make_forecast(IDX, weather, scr, entsoe, models)
assert np.allclose(out["weather_model_mw"], out3["weather_model_mw"]) and np.allclose(out["hybrid_mw"], out3["hybrid_mw"])


class FakeFetch:  # stands in for the network
    def __init__(self, entsoe_published=True):
        self.entsoe_published, self.calls = entsoe_published, []

    def fetch_weather(self, start, end):  # the fixture days, relabelled to whatever day is asked for
        self.calls.append("weather"); w = weather.copy(); w["time_utc"] += end - IDX[-1].floor("D"); return w

    def fetch_entsoe_forecast(self, idx):
        self.calls.append("entsoe")
        if not self.entsoe_published:
            return None
        e = entsoe.copy(); e.index = e.index + (idx[0] - IDX[0]); return e

    def fetch_actual_wind(self, start, end):
        self.calls.append("actual"); raise AssertionError("history is already long enough in this test")


# 4) runner: evening run for 28 Sep (ENTSO-E not yet out), then a later run fills in the hybrid, then next day it is scored
tmp = Path(tempfile.mkdtemp())
for d in ("models", "history"):
    shutil.copytree(ROOT / d, tmp / d)
logs = []
run("2026-09-27 17:00", tmp, FakeFetch(entsoe_published=False), log=logs.append)
f = tmp / "forecasts" / "2026-09-28.csv"
fc = pd.read_csv(f, index_col="time_utc")
assert fc["hybrid_mw"].isna().all() and fc["weather_model_mw"].notna().all()
first = fc["weather_model_mw"].copy()
run("2026-09-27 19:00", tmp, FakeFetch(entsoe_published=True), log=logs.append)
fc = pd.read_csv(f, index_col="time_utc")
assert fc["hybrid_mw"].notna().all() and np.allclose(fc["weather_model_mw"], first, atol=0.1)  # original forecast kept
run("2026-09-27 20:00", tmp, FakeFetch(entsoe_published=True), log=logs.append)  # nothing left to do
assert any("already complete" in m for m in logs)
run("2026-09-29 17:30", tmp, FakeFetch(entsoe_published=True), target="2026-09-30", log=logs.append)
sb = pd.read_csv(tmp / "scoreboard.csv")
assert list(sb["date"]) == ["2026-09-28"] and sb.loc[0, "n_hours"] == 24
a = hist.reindex(IDX)
assert abs(sb.loc[0, "mae_entsoe"] - (entsoe - a).abs().mean()) < 0.5
av = pd.read_csv(tmp / "logs" / "entsoe_availability.csv")
assert list(av["entsoe_wind_forecast_available"]) == [False, True, True, True]
# 5) too late: once the German day has started, no new forecast is made
tmp2 = Path(tempfile.mkdtemp())
for d in ("models", "history"):
    shutil.copytree(ROOT / d, tmp2 / d)
late = []
run("2026-09-28 06:00", tmp2, FakeFetch(entsoe_published=True), target="2026-09-28", log=late.append)
assert any("too late" in m for m in late) and not list((tmp2 / "forecasts").glob("*.csv"))
shutil.rmtree(tmp); shutil.rmtree(tmp2)
print("OK: live features and predictions match the research code; runner, scoring and availability log behave as expected")
