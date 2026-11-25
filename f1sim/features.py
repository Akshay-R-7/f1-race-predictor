"""Turn race results into model inputs, using ONLY information available before the race.

The key rule: a feature for race N may only use races 1..N-1 (plus qualifying for
race N, which happens before the race). Everything rolling is shifted by one race,
so the model never sees the future.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data.schema import TRACK_TYPES

FEATURES = ["quali_gap_s", "grid", "form_pace", "form_finish", "team_pace",
            "dnf_rate", "team_deg", "rain", "overtake_difficulty", "races_seen"] + \
           [f"track_{t}" for t in TRACK_TYPES]
TARGETS = ["pace_delta_s", "dnf", "tyre_deg_s_per_lap"]
DEFAULT_DEG_S_PER_LAP = 0.08  # used when a data source has no tyre data at all (e.g. Jolpica)


def _past_mean(s: pd.Series, window: int) -> pd.Series:
    # mean of the previous `window` values, excluding the current row
    return s.shift(1).rolling(window, min_periods=1).mean()


def _median_or(s: pd.Series, default: float) -> float:
    m = s.median()
    return default if pd.isna(m) else float(m)


def build(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["season", "round", "grid"]).reset_index(drop=True).copy()
    # An all-empty column comes out of SQLite as Python None objects; make it numeric NaN.
    for c in ("pace_delta_s", "tyre_deg_s_per_lap", "finish_pos"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    df["race_idx"] = df.groupby(["season", "round"]).ngroup()

    by_drv = df.groupby("driver", group_keys=False)
    df["form_pace"] = by_drv["pace_delta_s"].apply(lambda s: _past_mean(s, 5))
    df["form_finish"] = by_drv["finish_pos"].apply(lambda s: _past_mean(s, 5))
    df["dnf_rate"] = by_drv["dnf"].apply(lambda s: _past_mean(s.astype(float), 10))
    df["races_seen"] = by_drv.cumcount()

    # Team form: average over the team's two cars per race, then past 5 races.
    team_race = (df.groupby(["team", "race_idx"])[["pace_delta_s", "tyre_deg_s_per_lap"]]
                 .mean().reset_index().sort_values("race_idx"))
    team_race["team_pace"] = team_race.groupby("team")["pace_delta_s"].transform(lambda s: _past_mean(s, 5))
    team_race["team_deg"] = team_race.groupby("team")["tyre_deg_s_per_lap"].transform(lambda s: _past_mean(s, 5))
    df = df.merge(team_race[["team", "race_idx", "team_pace", "team_deg"]], on=["team", "race_idx"], how="left")

    for t in TRACK_TYPES:
        df[f"track_{t}"] = (df["track_type"] == t).astype(float)
    df["rain"] = df["rain"].astype(float)

    # Sensible defaults for drivers or teams with no history yet (rookies, new teams).
    fill = {"form_pace": 0.0, "form_finish": 10.5, "dnf_rate": 0.08,
            "team_pace": 0.0, "team_deg": _median_or(df["tyre_deg_s_per_lap"], DEFAULT_DEG_S_PER_LAP)}
    df = df.fillna(value=fill)
    return df


def time_split(df: pd.DataFrame, test_from: int = 2024):
    """Train on seasons before `test_from`, test on `test_from` and later. Never mixed."""
    return df[df["season"] < test_from].copy(), df[df["season"] >= test_from].copy()
