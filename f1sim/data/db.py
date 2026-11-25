"""SQLite storage with validation on the way in."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

import pandas as pd
from pydantic import ValidationError

from .schema import SCHEMA_SQL, Race, Result


def connect(path: str | Path) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(SCHEMA_SQL)
    return con


def validate_rows(rows: Iterable[dict], model) -> tuple[list, list[dict]]:
    """Split rows into valid models and rejects (with the reason), never raising."""
    good, bad = [], []
    for r in rows:
        try:
            good.append(model(**r))
        except ValidationError as e:
            bad.append({"row": r, "errors": [err["msg"] for err in e.errors()]})
    return good, bad


def write(con: sqlite3.Connection, races: list[dict], results: list[dict],
          reject_log: str | Path | None = None) -> dict:
    """Validate and insert. A race and its results are written in one transaction."""
    good_races, bad_races = validate_rows(races, Race)
    good_results, bad_results = validate_rows(results, Result)
    race_ids = {r.race_id for r in good_races}
    orphans = [r for r in good_results if r.race_id not in race_ids]
    good_results = [r for r in good_results if r.race_id in race_ids]
    bad_results += [{"row": r.model_dump(), "errors": ["unknown race_id"]} for r in orphans]

    with con:  # one transaction: all or nothing
        con.executemany(
            "INSERT OR REPLACE INTO races VALUES (:race_id,:season,:round,:name,:track_type,"
            ":laps,:pit_loss_s,:overtake_difficulty,:rain,:air_temp_c)",
            [r.model_dump() for r in good_races])
        con.executemany(
            "INSERT OR REPLACE INTO results VALUES (:race_id,:driver,:team,:grid,:quali_gap_s,"
            ":finish_pos,:dnf,:laps_completed,:pace_delta_s,:tyre_deg_s_per_lap,:pit_stops)",
            [r.model_dump() for r in good_results])

    if reject_log and (bad_races or bad_results):
        with open(reject_log, "a") as f:
            for b in bad_races + bad_results:
                f.write(json.dumps(b, default=str) + "\n")
    return {"races": len(good_races), "results": len(good_results),
            "rejected": len(bad_races) + len(bad_results)}


def load_frame(con: sqlite3.Connection) -> pd.DataFrame:
    """One row per driver per race, joined with race information, in time order."""
    df = pd.read_sql_query(
        "SELECT r.*, ra.season, ra.round, ra.name, ra.track_type, ra.laps, ra.pit_loss_s, "
        "ra.overtake_difficulty, ra.rain, ra.air_temp_c "
        "FROM results r JOIN races ra USING (race_id) ORDER BY ra.season, ra.round, r.grid", con)
    df["dnf"] = df["dnf"].astype(bool)
    df["rain"] = df["rain"].astype(bool)
    return df
