"""Public dashboard: tomorrow's German wind forecast and a track record of forecasts saved BEFORE the day.
Reads only the CSV files the daily job commits. Run locally:  streamlit run app.py"""
from pathlib import Path

import pandas as pd
import streamlit as st

R = Path(__file__).parent
st.set_page_config(page_title="German wind forecast, live track record", layout="wide")
st.title("German wind power: tomorrow's forecast and a live track record")
st.caption("All times UTC. Forecasts are saved by an automatic daily job before the day they cover; "
           "the scoreboard compares them with ENTSO-E's actual generation afterwards. Not financial advice.")
t1, t2, t3 = st.tabs(["Tomorrow", "Track record", "About"])

files = sorted((R / "forecasts").glob("*.csv"))
with t1:
    if not files:
        st.info("No forecast yet. The first one appears after the daily job has run.")
    else:
        f = files[-1]
        d = pd.read_csv(f, index_col="time_utc", parse_dates=["time_utc"])
        st.subheader(f"Forecast for {f.stem}")
        st.line_chart(d[["weather_model_mw", "entsoe_mw", "hybrid_mw"]].rename(columns={
            "weather_model_mw": "Weather model", "entsoe_mw": "ENTSO-E day-ahead", "hybrid_mw": "Hybrid (ENTSO-E + correction)"}))
        made = d["made_at_utc"].iloc[0]
        hy = d["hybrid_made_at_utc"].dropna().astype(str).iloc[0] if d["hybrid_made_at_utc"].notna().any() else "not yet"
        st.write(f"Weather model forecast made at **{made} UTC**. Hybrid forecast made at **{hy}**"
                 f"{'' if hy != 'not yet' else ' (ENTSO-E has not published its forecast yet)'}.")
        st.write(f"Average forecast: weather model {d['weather_model_mw'].mean():,.0f} MW"
                 + (f", ENTSO-E {d['entsoe_mw'].mean():,.0f} MW." if d["entsoe_mw"].notna().any() else "."))

with t2:
    sbp = R / "scoreboard.csv"
    sb = pd.read_csv(sbp) if sbp.exists() else pd.DataFrame()
    if sb.empty:
        st.info("No finished days yet. A forecast is scored once its day is over and actual generation is published.")
    else:
        sb = sb.sort_values("date")
        full = sb.dropna(subset=["mae_weather_model", "mae_hybrid", "mae_entsoe"])
        st.metric("Days scored", len(sb))
        if len(full):
            c = st.columns(3)
            c[0].metric("ENTSO-E average error (MW)", f"{full['mae_entsoe'].mean():,.0f}")
            c[1].metric("Hybrid average error (MW)", f"{full['mae_hybrid'].mean():,.0f}")
            c[2].metric("Weather model average error (MW)", f"{full['mae_weather_model'].mean():,.0f}")
            st.write(f"Hybrid beat ENTSO-E on {(full['mae_hybrid'] < full['mae_entsoe']).mean():.0%} of {len(full)} days.")
            if len(full) < 14:
                st.warning("Fewer than 14 days: too few to say anything reliable yet.")
        st.line_chart(sb.set_index("date")[["mae_weather_model", "mae_hybrid", "mae_entsoe"]].rename(columns={
            "mae_weather_model": "Weather model", "mae_hybrid": "Hybrid", "mae_entsoe": "ENTSO-E"}))
        st.dataframe(sb.round(0), use_container_width=True)

with t3:
    st.markdown("""
**What this is.** A small live version of a research project (German wind generation, day-ahead). Two models were trained once on data to
30 September 2026 and are not retrained: a *weather model* (weather forecasts at 8 locations plus recent generation) and a *hybrid* that
adds a learned correction to ENTSO-E's own day-ahead forecast.

**Honest limits.**
- ENTSO-E publishes its wind forecast after the day-ahead auction closes, so the hybrid is not usable before the auction.
- Live weather inputs differ from those used in the research test, so live errors can differ from the research results.
- Early in the track record there are too few days to conclude anything.
- Hourly values are for UTC days, not German days.
""")
    av = R / "logs" / "entsoe_availability.csv"
    if av.exists():
        a = pd.read_csv(av)
        first = a[a["entsoe_wind_forecast_available"]].groupby("target_date")["run_utc"].min().rename("first seen by the job (UTC)")
        st.subheader("When did ENTSO-E's wind forecast first appear?")
        st.dataframe(first.reset_index().tail(14), use_container_width=True)
