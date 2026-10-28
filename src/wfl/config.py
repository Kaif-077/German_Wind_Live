"""Settings shared by the live job. Copied from the research project (Power_Market_Analytics)."""
DATA_START = "2024-02-17 00:00"
DATA_END = "2026-09-30 23:00"
WEATHER_VARS = ["wind_speed_100m", "wind_direction_100m", "wind_gusts_10m", "temperature_2m", "surface_pressure"]
ONSHORE_LOCS = ["sh_west_coast", "lower_saxony", "brandenburg", "saxony_anhalt", "nrw_hesse"]
OFFSHORE_LOCS = ["north_sea_offshore", "baltic_offshore"]
WIND_LOCS = ONSHORE_LOCS + OFFSHORE_LOCS
ALL_LOCS = WIND_LOCS + ["bavaria_south"]
LOCATIONS = {  # latitude, longitude
    "sh_west_coast": (54.48, 9.05), "lower_saxony": (53.37, 7.20), "north_sea_offshore": (54.01, 6.60),
    "baltic_offshore": (54.77, 14.12), "brandenburg": (52.50, 13.50), "saxony_anhalt": (52.10, 11.60),
    "nrw_hesse": (51.50, 8.50), "bavaria_south": (48.50, 11.50),
}
