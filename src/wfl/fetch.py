"""Network calls (Open-Meteo and ENTSO-E). Thin on purpose: all logic lives in forecast.py."""
import os
import time

import pandas as pd
import requests

from . import config as C

OM_URL = "https://api.open-meteo.com/v1/forecast"
ZONE = "DE_LU"


def _get(url, params, tries=4):
    for k in range(tries):
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(5 * 2 ** k)


def fetch_weather(start, end):
    """Latest weather forecasts at the 8 locations for the UTC dates start..end (inclusive), long format."""
    parts = []
    for loc, (lat, lon) in C.LOCATIONS.items():
        h = _get(OM_URL, {"latitude": lat, "longitude": lon, "hourly": ",".join(C.WEATHER_VARS), "timezone": "GMT",
                          "wind_speed_unit": "ms", "start_date": f"{start:%Y-%m-%d}", "end_date": f"{end:%Y-%m-%d}"})["hourly"]
        d = pd.DataFrame(h).rename(columns={"time": "time_utc"})
        d["time_utc"] = pd.to_datetime(d["time_utc"], utc=True)
        d["location"] = loc
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def _client():
    from entsoe import EntsoePandasClient

    return EntsoePandasClient(api_key=os.environ["ENTSOE_API_KEY"])


def _hourly(s):
    s.index = pd.to_datetime(s.index, utc=True)
    return s[~s.index.duplicated()].sort_index().resample("h").mean()


def fetch_entsoe_forecast(day):
    """ENTSO-E day-ahead wind forecast (onshore + offshore, MW) for the UTC day, or None if not published yet."""
    try:
        df = _client().query_wind_and_solar_forecast(ZONE, start=day, end=day + pd.Timedelta(days=1))
    except Exception as e:  # noqa: BLE001
        if type(e).__name__ == "NoMatchingDataError":
            return None
        raise
    s = _hourly(df["Wind Onshore"] + df["Wind Offshore"])
    s = s.reindex(pd.date_range(day, periods=24, freq="h", tz="UTC"))
    return s if s.notna().all() else None


def _merge_split_columns(df):
    """entsoe-py output differs before and after July 2025; 'Consumption' columns are always zero."""
    if isinstance(df, pd.Series):
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" | ".join(str(x) for x in c) for c in df.columns]
    cols = [c for c in df.columns if "Consumption" not in c]
    out = df[cols[0]].copy()
    for c in cols[1:]:
        out = out.combine_first(df[c])
    return out


def fetch_actual_wind(start, end):
    """Actual wind generation (onshore + offshore, MW, hourly UTC) for start <= t < end."""
    c = _client()
    parts = []
    for psr in ("B19", "B18"):
        parts.append(_hourly(_merge_split_columns(c.query_generation(ZONE, start=start, end=end, psr_type=psr))))
    return (parts[0] + parts[1]).dropna().rename("wind_total_mw")
