"""Core of the live job: turn fetched data into a 24-hour forecast, and score finished days. No network here,
so everything in this file can be tested offline."""
import joblib
import numpy as np
import pandas as pd

from . import features as F


TZ = "Europe/Berlin"


def load_models(path):
    return joblib.load(path)


def target_index(local_date):
    """Hourly UTC timestamps of one GERMAN calendar day (23, 24 or 25 hours on clock-change days). ENTSO-E publishes
    whole German days, so the forecast day must be a German day, not a UTC day."""
    d = pd.Timestamp(str(local_date)[:10])
    start, end = d.tz_localize(TZ), (d + pd.Timedelta(days=1)).tz_localize(TZ)
    return pd.date_range(start, end, freq="h", inclusive="left").tz_convert("UTC")


def build_features(idx, weather_long, actual_wind):
    """Feature matrix for the hours in `idx` (UTC). `weather_long` must also cover the 6 hours before the first hour.
    `actual_wind` is the hourly history of actual generation; only values before the information cut-off are used
    (see features.observation_features)."""
    X = F.weather_features(weather_long).reindex(idx)
    if X.isna().any().any():
        raise ValueError("weather data missing for some hours of the target day")
    return X.join(F.observation_features(actual_wind, idx))


def make_forecast(idx, weather_long, actual_wind, entsoe_fc, models):
    """Returns (forecast table, feature matrix). entsoe_fc: hourly Series for the day, or None if not published yet."""
    X = build_features(idx, weather_long, actual_wind)
    cap = X["cap_proxy"]
    if cap.isna().any():
        raise ValueError("not enough actual-generation history for the capacity proxy")
    out = pd.DataFrame(index=X.index)
    out["weather_model_mw"] = models["weather_obs"].predict(X[models["cols_weather_obs"]]) * cap.to_numpy()
    out["entsoe_mw"] = entsoe_fc.reindex(X.index) if entsoe_fc is not None else np.nan
    if out["entsoe_mw"].notna().all():
        Xh = X.assign(entsoe_fc=out["entsoe_mw"])
        out["hybrid_mw"] = out["entsoe_mw"] + models["hybrid"].predict(Xh[models["cols_hybrid"]])
    else:
        out["hybrid_mw"] = np.nan
    return out, X


def score_day(fc: pd.DataFrame, actual_wind: pd.Series):
    """Mean absolute error of each forecast column against actual generation; None until every actual hour exists."""
    a = actual_wind.reindex(fc.index)
    if len(fc) < 23 or a.isna().any():
        return None
    res = {"n_hours": len(fc)}
    for col, name in (("weather_model_mw", "mae_weather_model"), ("hybrid_mw", "mae_hybrid"), ("entsoe_mw", "mae_entsoe")):
        res[name] = float((fc[col] - a).abs().mean()) if fc[col].notna().all() else np.nan
    return res
