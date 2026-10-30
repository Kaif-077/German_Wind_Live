# German wind power: live forecast and public track record

A small live companion to the research project [Power_Market_Analytics](https://github.com/Kaif-077/Power_Market_Analytics).
Every day an automatic job (GitHub Actions) forecasts the next day's German wind generation, saves the forecast **before the day happens**,
and later scores it against ENTSO-E's actual generation. The dashboard shows tomorrow's forecast and the running scoreboard.

**Live dashboard:** (add your Streamlit link here after deploying)

## What runs each day (`run_daily.py`)
1. Add yesterday's actual wind generation (ENTSO-E) to `history/actual_wind.csv`.
2. Score every finished forecast in `forecasts/` and add a row to `scoreboard.csv`.
3. Fetch the latest weather forecast (Open-Meteo, 8 locations) and ENTSO-E's day-ahead wind forecast for tomorrow, and save
   `forecasts/<date>.csv`: a **weather model** forecast and, once ENTSO-E has published, a **hybrid** forecast (ENTSO-E plus a learned correction).
   An existing forecast is never overwritten, so the commit history proves when each was made.
4. Log whether ENTSO-E's forecast was available at each run (`logs/entsoe_availability.csv`), which measures its publication time day by day.

The job is scheduled four times a day (08:30, 11:00, 13:30 and 15:30 UTC). GitHub often starts scheduled jobs hours late, which is why the times are early; ENTSO-E's forecast appeared between 13:04 and 17:20 UTC on the days checked.

## Models
Two gradient-boosting models trained once on data from 17 February 2024 to 30 September 2026 and not retrained (`models/final_models.joblib`;
`tools/train_final_models.py` shows how, using the research project's data). Features and settings are the research project's.

## Tests (no network needed)
`python tests/test_live.py` checks that the live code reproduces the research code's features and predictions exactly for one day, that future
actuals cannot change a forecast, and that the runner keeps old forecasts, fills in the hybrid later, scores only finished days and logs availability.
`python tests/test_fetch.py` checks how the network layer reads Open-Meteo and ENTSO-E answers (15-minute data, missing or partial days, the two column layouts of entsoe-py) using stand-ins.
The real calls cannot be tested offline; the first live runs are their test.

## Honest limits
- The hybrid needs ENTSO-E's forecast, which appears after the day-ahead auction closes, so it is **not usable before the auction**.
- Live weather inputs are the latest forecast run, not the exact "day before" product used in the research test, so live errors can differ.
- Each forecast covers one German calendar day (23 to 25 hours). Fewer than 14 scored days say nothing reliable; do not quote results before that.
- Not financial advice. No trading signal is produced.

## Setup (GitHub)
1. Settings, Secrets and variables, Actions: add a secret `ENTSOE_API_KEY` with your ENTSO-E key.
2. Settings, Actions, General, Workflow permissions: Read and write.
3. Actions tab, `daily-forecast`, Run workflow, to start the first run by hand and read its log.
4. Deploy `app.py` on Streamlit Community Cloud (it needs only `requirements.txt` and the committed CSV files).

## Known issue fixed on 30 October 2026
The first version asked ENTSO-E for a UTC day. ENTSO-E publishes whole German days, so the last hour of every UTC day was always missing and the job never saw the forecast. Forecast files from before the fix (2026-10-29 and 2026-10-30, UTC days) have no ENTSO-E or hybrid values, and their availability log rows are invalid.
