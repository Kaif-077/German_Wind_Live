"""Daily job: update the actual-generation history, score finished forecasts, make tomorrow's forecast.

    python run_daily.py                 # forecast for tomorrow (UTC date), run in the evening after ENTSO-E publishes
    python run_daily.py --target 2026-10-30
Outputs are plain CSV files in forecasts/, scoreboard.csv, history/ and logs/ so the dashboard needs no database.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from wfl.forecast import load_models, make_forecast, score_day

H = pd.Timedelta(hours=1)


def _read_series(path):
    d = pd.read_csv(path, index_col=0)
    d.index = pd.to_datetime(d.index, utc=True)
    return d.iloc[:, 0].sort_index()


def run(now, root, fetch, target=None, dry=False, log=print):
    root = Path(root)
    now = pd.Timestamp(now)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    today = now.normalize()
    target = today + pd.Timedelta(days=1) if target is None else pd.Timestamp(target, tz="UTC")
    models = load_models(root / "models" / "final_models.joblib")

    # 1) actual generation up to the end of yesterday
    hist = _read_series(root / "history" / "actual_wind.csv")
    if hist.index.max() < today - H:
        new = fetch.fetch_actual_wind(hist.index.max() + H, today)
        hist = pd.concat([hist, new[new.index > hist.index.max()]]).sort_index()
        log(f"history extended to {hist.index.max()}")
        if not dry:
            hist.rename("wind_total_mw").to_csv(root / "history" / "actual_wind.csv", index_label="time_utc")

    # 2) score every finished forecast that is not on the scoreboard yet
    sb_path = root / "scoreboard.csv"
    sb = pd.read_csv(sb_path) if sb_path.exists() else pd.DataFrame(
        columns=["date", "n_hours", "mae_weather_model", "mae_hybrid", "mae_entsoe", "scored_at_utc"])
    for f in sorted((root / "forecasts").glob("*.csv")):
        if f.stem in set(sb["date"].astype(str)) or pd.Timestamp(f.stem, tz="UTC") >= today:
            continue  # already scored, or the day is not over yet
        fc = pd.read_csv(f, index_col="time_utc", parse_dates=["time_utc"])
        res = score_day(fc, hist)
        if res:
            sb = pd.concat([sb, pd.DataFrame([{"date": f.stem, **res, "scored_at_utc": f"{now:%Y-%m-%d %H:%M}"}])], ignore_index=True)
            log(f"scored {f.stem}: {res}")
    if not dry:
        sb.to_csv(sb_path, index=False)

    # 3) tomorrow's forecast (never overwrite a forecast that already exists; only fill in a missing hybrid)
    path = root / "forecasts" / f"{target:%Y-%m-%d}.csv"
    existing = pd.read_csv(path, index_col="time_utc", parse_dates=["time_utc"]) if path.exists() else None
    entsoe = fetch.fetch_entsoe_forecast(target)
    av = root / "logs" / "entsoe_availability.csv"
    if not dry:
        av.parent.mkdir(exist_ok=True)
        pd.DataFrame([{"run_utc": f"{now:%Y-%m-%d %H:%M}", "target_date": f"{target:%Y-%m-%d}",
                       "entsoe_wind_forecast_available": entsoe is not None}]).to_csv(av, mode="a", header=not av.exists(), index=False)
    if existing is not None and not existing["hybrid_mw"].isna().any():
        log(f"forecast for {target:%Y-%m-%d} already complete")
        return
    if existing is not None and entsoe is None:
        log(f"forecast for {target:%Y-%m-%d} exists; ENTSO-E forecast still not published")
        return
    weather = fetch.fetch_weather(target - pd.Timedelta(days=1), target)
    out, _ = make_forecast(target, weather, hist, entsoe, models)
    if existing is not None:  # keep the original weather-model forecast, add ENTSO-E and hybrid
        out["weather_model_mw"] = existing["weather_model_mw"].to_numpy()
        out["made_at_utc"] = existing["made_at_utc"].to_numpy()
    else:
        out["made_at_utc"] = f"{now:%Y-%m-%d %H:%M}"
    out["hybrid_made_at_utc"] = f"{now:%Y-%m-%d %H:%M}" if out["hybrid_mw"].notna().all() else ""
    out.index.name = "time_utc"
    log(f"forecast for {target:%Y-%m-%d}: weather model mean {out['weather_model_mw'].mean():.0f} MW, "
        f"ENTSO-E {'available' if entsoe is not None else 'not yet available'}")
    if not dry:
        path.parent.mkdir(exist_ok=True)
        out.round(1).to_csv(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", help="UTC date YYYY-MM-DD (default: tomorrow)")
    ap.add_argument("--dry-run", action="store_true", help="print what would happen, write nothing")
    a = ap.parse_args()
    from wfl import fetch

    run(pd.Timestamp.now(tz="UTC"), ROOT, fetch, a.target, a.dry_run)
