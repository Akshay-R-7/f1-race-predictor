import sqlite3
from datetime import timedelta

import pandas as pd
import pytest

from f1sim.data.db import connect, load_frame, validate_rows, write
from f1sim.data.fastf1_loader import pace_deltas, quali_gaps, tyre_degradation
from f1sim.data.sample import generate
from f1sim.data.schema import Race, Result

RACE = dict(race_id="2024-01", season=2024, round=1, name="Bahrain", track_type="balanced",
            laps=57, pit_loss_s=22.0, overtake_difficulty=0.3, rain=False, air_temp_c=25.0)
RES = dict(race_id="2024-01", driver="VER", team="Redline", grid=1, quali_gap_s=0.0,
           finish_pos=1, dnf=False, laps_completed=57, pace_delta_s=-0.6,
           tyre_deg_s_per_lap=0.06, pit_stops=2)


def test_valid_rows_pass():
    Race(**RACE)
    Result(**RES)


@pytest.mark.parametrize("change", [
    {"grid": 0}, {"driver": "VERSTAPPEN"}, {"pace_delta_s": 40.0},
    {"dnf": True},                       # did not finish but has a finishing position
    {"finish_pos": None},                # finished but no position
])
def test_bad_results_rejected(change):
    _, bad = validate_rows([{**RES, **change}], Result)
    assert len(bad) == 1


def test_race_id_must_match_season_and_round():
    _, bad = validate_rows([{**RACE, "race_id": "2024-02"}], Race)
    assert bad and "race_id" in bad[0]["errors"][0]


def test_write_rejects_orphans_and_logs(tmp_path):
    con = connect(tmp_path / "t.sqlite")
    log = tmp_path / "rej.jsonl"
    stats = write(con, [RACE], [RES, {**RES, "race_id": "2024-09", "driver": "HAM"}], log)
    assert stats == {"races": 1, "results": 1, "rejected": 1}
    assert log.read_text().count("unknown race_id") == 1


def test_sample_data_is_all_valid_and_loads(tmp_path):
    races, results = generate(seasons=range(2023, 2025))
    con = connect(tmp_path / "s.sqlite")
    stats = write(con, races, results)
    assert stats["rejected"] == 0
    df = load_frame(con)
    assert len(df) == stats["results"]
    # every race has exactly one pole sitter and unique grid slots
    assert (df.groupby("race_id")["grid"].nunique() == df.groupby("race_id").size()).all()


def _laps(rows):
    return pd.DataFrame(rows)


def test_quali_gap_uses_best_session_time():
    q = pd.DataFrame({"Abbreviation": ["VER", "HAM"],
                      "Q1": [timedelta(seconds=90.5), timedelta(seconds=90.4)],
                      "Q2": [timedelta(seconds=90.0), pd.NaT],
                      "Q3": [timedelta(seconds=89.8), pd.NaT]})
    g = quali_gaps(q)
    assert g["VER"] == 0.0 and abs(g["HAM"] - 0.6) < 1e-9


def test_pace_and_degradation_from_laps():
    rows = []
    for drv, base, slope in [("VER", 90.0, 0.05), ("HAM", 90.5, 0.10)]:
        for age in range(1, 11):
            rows.append(dict(Driver=drv, LapTime=timedelta(seconds=base + slope * age),
                             IsAccurate=True, PitInTime=pd.NaT, PitOutTime=pd.NaT,
                             TrackStatus="1", Stint=1, TyreLife=age))
    laps = _laps(rows)
    p = pace_deltas(laps)
    assert p["VER"] < 0 < p["HAM"]
    d = tyre_degradation(laps)
    assert abs(d["VER"] - 0.05) < 1e-6 and abs(d["HAM"] - 0.10) < 1e-6
