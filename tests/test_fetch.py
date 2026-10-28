"""Tests of how fetch.py reads what Open-Meteo and ENTSO-E return, using stand-ins (no network).
Run with:  python tests/test_fetch.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from wfl import config as C
from wfl import fetch

DAY = pd.Timestamp("2026-10-29", tz="UTC")

# Open-Meteo: JSON with hourly lists -> long table with the 8 locations
calls = []
def fake_get(url, params, tries=4):
    calls.append(params)
    t = pd.date_range("2026-10-28", periods=48, freq="h").strftime("%Y-%m-%dT%H:%M").tolist()
    return {"hourly": {"time": t, **{v: list(np.arange(48, dtype=float)) for v in C.WEATHER_VARS}}}
fetch._get = fake_get
w = fetch.fetch_weather(DAY - pd.Timedelta(days=1), DAY)
assert len(w) == 48 * 8 and set(w["location"]) == set(C.LOCATIONS) and str(w["time_utc"].dt.tz) == "UTC"
assert all(c["timezone"] == "GMT" and c["wind_speed_unit"] == "ms" and c["start_date"] == "2026-10-28" for c in calls)

# ENTSO-E stand-in: 15-minute data, as the real service returns since late 2025
class NoMatchingDataError(Exception):
    pass

def quarter(day, cols, n=96):
    idx = pd.date_range(day, periods=n, freq="15min").tz_convert("Europe/Berlin")  # the real library answers in local time
    return pd.DataFrame({c: np.full(n, v, dtype=float) for c, v in cols.items()}, index=idx)

class Fake:
    mode = "ok"
    def query_wind_and_solar_forecast(self, code, *, start, end, **kw):
        if self.mode == "none":
            raise NoMatchingDataError()
        n = 96 if self.mode == "ok" else 40  # 'partial' = only part of the day published
        return quarter(start, {"Solar": 100.0, "Wind Offshore": 1000.0, "Wind Onshore": 5000.0}, n)
    def query_generation(self, code, *, start, end, psr_type=None, **kw):
        v = {"B19": 8000.0, "B18": 2000.0}[psr_type]
        d = quarter(start, {"Wind Onshore" if psr_type == "B19" else "Wind Offshore": v}, 96 * 2)
        return d
fake = Fake()
fetch._client = lambda: fake
s = fetch.fetch_entsoe_forecast(DAY)
assert len(s) == 24 and (s == 6000.0).all() and str(s.index.tz) == "UTC"
fake.mode = "none"; assert fetch.fetch_entsoe_forecast(DAY) is None       # not published yet
fake.mode = "partial"; assert fetch.fetch_entsoe_forecast(DAY) is None    # incomplete day is not used
a = fetch.fetch_actual_wind(DAY, DAY + pd.Timedelta(days=2))
assert len(a) == 48 and (a == 10000.0).all()

# column layouts returned by entsoe-py before and after July 2025, and a mixed frame
idx = pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC")
old = pd.DataFrame({("Wind Onshore", "Actual Aggregated"): [1.0, 2, 3, 4], ("Wind Onshore", "Actual Consumption"): [0.0] * 4}, index=idx)
old.columns = pd.MultiIndex.from_tuples(old.columns)
assert fetch._merge_split_columns(old).tolist() == [1.0, 2, 3, 4]
new = pd.DataFrame({"Wind Onshore": [5.0, 6, 7, 8]}, index=idx)
assert fetch._merge_split_columns(new).tolist() == [5.0, 6, 7, 8]
mixed = pd.DataFrame({"Wind Onshore | Actual Aggregated": [1, 2, np.nan, np.nan], "Wind Onshore | Actual Consumption": [0.0] * 4,
                      "Wind Onshore": [np.nan, np.nan, 3, 4]}, index=idx)
assert fetch._merge_split_columns(mixed).tolist() == [1.0, 2, 3, 4]
print("OK: weather, ENTSO-E forecast (15-minute, missing, partial) and actual-generation parsing behave as expected")
