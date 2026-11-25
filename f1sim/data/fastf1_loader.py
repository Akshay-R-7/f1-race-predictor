"""Load real F1 data with FastF1 (run this on a machine with internet access).

    pip install fastf1
    python -m f1sim.data.fastf1_loader --from 2018 --to 2025 --db f1.sqlite

FastF1 downloads timing data from the official F1 live-timing service and caches
it in ./fastf1_cache, so the first run is slow and later runs are fast.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .db import connect, write

log = logging.getLogger(__name__)

# Rough circuit classification; edit to taste. Unknown circuits default to "balanced".
TRACK_TYPE = {
    "Monaco": "street", "Baku": "street", "Marina Bay": "street", "Jeddah": "street",
    "Melbourne": "street", "Miami": "street", "Las Vegas": "street",
    "Monza": "high_speed", "Spa-Francorchamps": "high_speed", "Silverstone": "high_speed",
    "Spielberg": "high_speed", "Montréal": "high_speed",
    "Hungaroring": "technical", "Budapest": "technical", "Barcelona": "technical",
    "Suzuka": "technical", "Zandvoort": "technical", "Imola": "technical",
}
OVERTAKE = {"street": 0.6, "technical": 0.7, "high_speed": 0.25, "balanced": 0.4}
PIT_LOSS = {"street": 21.0, "technical": 22.0, "high_speed": 21.0, "balanced": 22.0}


def quali_gaps(quali_results: pd.DataFrame) -> dict[str, float]:
    """Gap to pole in seconds, using each driver's best time across Q1-Q3."""
    best = quali_results[["Q1", "Q2", "Q3"]].min(axis=1).dt.total_seconds()
    pole = best.min()
    return {abbr: float(max(0.0, b - pole)) if pd.notna(b) else 5.0
            for abbr, b in zip(quali_results["Abbreviation"], best)}


def pace_deltas(laps: pd.DataFrame) -> dict[str, float]:
    """Median clean-lap time per driver minus the field median of those medians.

    'Clean' means: accurate timing, not an in/out lap, green flag (TrackStatus == '1').
    """
    clean = laps[(laps["IsAccurate"]) & laps["PitInTime"].isna() & laps["PitOutTime"].isna()
                 & (laps["TrackStatus"] == "1")].copy()
    clean["t"] = clean["LapTime"].dt.total_seconds()
    med = clean.groupby("Driver")["t"].median()
    return (med - med.median()).round(3).to_dict()


def tyre_degradation(laps: pd.DataFrame) -> dict[str, float]:
    """Median slope of lap time vs tyre age across a driver's stints (s per lap of age)."""
    out = {}
    clean = laps[(laps["IsAccurate"]) & (laps["TrackStatus"] == "1")].copy()
    clean["t"] = clean["LapTime"].dt.total_seconds()
    for drv, g in clean.groupby("Driver"):
        slopes = []
        for _, stint in g.groupby("Stint"):
            if len(stint) >= 6:
                slopes.append(np.polyfit(stint["TyreLife"], stint["t"], 1)[0])
        if slopes:
            out[drv] = float(np.clip(np.median(slopes), -0.2, 1.0))
    return out


def load_season(year: int):
    import fastf1  # imported here so the rest of the project works without it
    races, results = [], []
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    for _, ev in schedule.iterrows():
        rnd = int(ev["RoundNumber"])
        try:
            q = fastf1.get_session(year, rnd, "Q"); q.load(laps=False, telemetry=False, weather=False, messages=False)
            r = fastf1.get_session(year, rnd, "R"); r.load(laps=True, telemetry=False, weather=True, messages=False)
        except Exception as e:  # sprint-only weekends, cancelled races, missing data
            log.warning("skipping %s round %s: %s", year, rnd, e)
            continue
        loc = str(ev["Location"])
        ttype = TRACK_TYPE.get(loc, "balanced")
        wx = r.weather_data
        races.append(dict(
            race_id=f"{year}-{rnd:02d}", season=year, round=rnd, name=str(ev["EventName"]),
            track_type=ttype, laps=int(r.laps["LapNumber"].max()), pit_loss_s=PIT_LOSS[ttype],
            overtake_difficulty=OVERTAKE[ttype], rain=bool(wx["Rainfall"].any()),
            air_temp_c=float(wx["AirTemp"].mean())))
        gaps = quali_gaps(q.results)
        pace = pace_deltas(r.laps)
        deg = tyre_degradation(r.laps)
        stints = r.laps.groupby("Driver")["Stint"].max()
        for _, row in r.results.iterrows():
            abbr = row["Abbreviation"]
            finished = str(row["Status"]) == "Finished" or str(row["Status"]).startswith("+")
            results.append(dict(
                race_id=f"{year}-{rnd:02d}", driver=abbr, team=str(row["TeamName"]),
                grid=int(row["GridPosition"]) if row["GridPosition"] > 0 else 20,
                quali_gap_s=gaps.get(abbr, 5.0),
                finish_pos=int(row["Position"]) if finished else None, dnf=not finished,
                laps_completed=int(row["Laps"]),
                pace_delta_s=pace.get(abbr), tyre_deg_s_per_lap=deg.get(abbr),
                pit_stops=max(0, int(stints.get(abbr, 1)) - 1)))
    return races, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", type=int, default=2018)
    ap.add_argument("--to", dest="end", type=int, default=2025)
    ap.add_argument("--db", default="f1.sqlite")
    args = ap.parse_args()
    import fastf1
    Path("fastf1_cache").mkdir(exist_ok=True)
    fastf1.Cache.enable_cache("fastf1_cache")
    con = connect(args.db)
    for y in range(args.start, args.end + 1):
        races, results = load_season(y)
        print(y, write(con, races, results, reject_log="rejected_rows.jsonl"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
