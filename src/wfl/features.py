"""Feature engineering for the day-ahead wind forecast.

Information cutoff: a forecast for any hour of day D is made before day D-1 starts (00:00 UTC),
which is earlier than the day-ahead auction closes (12:00 CET). So features built from ACTUAL
generation may only use hours up to D-2 23:00 UTC. Weather features are the day-ahead
forecasts themselves, so they are allowed for the whole of day D.
"""
import numpy as np
import pandas as pd

from . import config as C

CUTOFF_LAG = pd.Timedelta(hours=25)  # day start minus 25 h = last actual hour we may use


def _wide(df: pd.DataFrame, var: str, idx: pd.DatetimeIndex) -> pd.DataFrame:
    return df.pivot(index="time_utc", columns="location", values=var).reindex(idx)


def weather_features(weather: pd.DataFrame) -> pd.DataFrame:
    full_idx = pd.date_range(weather["time_utc"].min(), weather["time_utc"].max(), freq="h", tz="UTC")
    ws = _wide(weather, "wind_speed_100m", full_idx)
    wd = np.deg2rad(_wide(weather, "wind_direction_100m", full_idx))
    gust = _wide(weather, "wind_gusts_10m", full_idx)
    temp = _wide(weather, "temperature_2m", full_idx)
    pres = _wide(weather, "surface_pressure", full_idx)

    f = {}
    for loc in C.ALL_LOCS:
        f[f"ws_{loc}"] = ws[loc]
        f[f"wdsin_{loc}"] = np.sin(wd[loc])
        f[f"wdcos_{loc}"] = np.cos(wd[loc])
        f[f"gust_{loc}"] = gust[loc]
        f[f"pres_{loc}"] = pres[loc]
        f[f"temp_{loc}"] = temp[loc]
    X = pd.DataFrame(f, index=full_idx)

    on, off, allw = ws[C.ONSHORE_LOCS], ws[C.OFFSHORE_LOCS], ws[C.WIND_LOCS]
    X["ws_mean_on"] = on.mean(axis=1)
    X["ws_mean_off"] = off.mean(axis=1)
    X["ws_mean_all"] = allw.mean(axis=1)
    X["ws_std_all"] = allw.std(axis=1)
    X["ws_max_all"] = allw.max(axis=1)
    X["ws3_mean_on"] = (on ** 3).mean(axis=1)
    X["ws3_mean_off"] = (off ** 3).mean(axis=1)
    # Backward-looking changes only (no peeking at later forecast hours).
    for h in (3, 6):
        X[f"ramp{h}_on"] = X["ws_mean_on"].diff(h)
        X[f"ramp{h}_off"] = X["ws_mean_off"].diff(h)
    X["pres_mean"] = pres[C.WIND_LOCS].mean(axis=1)
    X["pres_change6"] = X["pres_mean"].diff(6)

    X["hour"] = X.index.hour
    doy = X.index.dayofyear
    X["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    X["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return X


def observation_features(gen: pd.Series, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Features from past actual generation, respecting the cutoff."""
    gen = gen.sort_index()
    roll24 = gen.rolling(24, min_periods=24).mean()
    # Installed-capacity proxy: 98th percentile of the last 365 days (a full year, so it does not
    # follow the wind seasons). Needs at least 120 days of history.
    cap = gen.rolling(24 * 365, min_periods=24 * 120).quantile(0.98)
    cut = index.floor("D") - CUTOFF_LAG
    out = pd.DataFrame(index=index)
    out["cap_proxy"] = cap.reindex(cut).to_numpy()
    out["gen_cut_last"] = gen.reindex(cut).to_numpy()
    out["gen_cut_mean24"] = roll24.reindex(cut).to_numpy()
    out["gen_lag48"] = gen.reindex(index - pd.Timedelta(hours=48)).to_numpy()
    return out


def build_dataset(weather: pd.DataFrame, actuals: pd.DataFrame, entsoe_fc: pd.DataFrame) -> pd.DataFrame:
    X = weather_features(weather)
    X = X.loc[C.DATA_START:C.DATA_END]
    obs = observation_features(actuals["wind_total"], X.index)
    df = X.join(obs)
    df["y"] = actuals["wind_total"].reindex(df.index)
    df["entsoe_fc"] = entsoe_fc["entsoe_fc"].reindex(df.index)
    return df


WEATHER_COLS_PREFIXES = ("ws_", "wdsin_", "wdcos_", "gust_", "pres_", "temp_", "ws3_", "ramp", "doy_")


def feature_sets(df: pd.DataFrame) -> dict:
    weather_cols = [c for c in df.columns if c.startswith(WEATHER_COLS_PREFIXES) or c == "hour"]
    obs_cols = ["gen_cut_last", "gen_cut_mean24", "gen_lag48"]
    return {
        "weather_only": weather_cols + ["cap_proxy"],
        "weather_plus_obs": weather_cols + ["cap_proxy"] + obs_cols,
        "hybrid_with_entsoe": weather_cols + ["cap_proxy"] + obs_cols + ["entsoe_fc"],
    }
