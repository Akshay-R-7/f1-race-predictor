"""Row models and the database schema.

Every row is validated with Pydantic before it is written, so bad data from an
API (a negative grid slot, a lap delta of 40 seconds) is rejected, logged and
never reaches the model.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

TrackType = Literal["street", "high_speed", "technical", "balanced"]
TRACK_TYPES: tuple[str, ...] = ("street", "high_speed", "technical", "balanced")


class Race(BaseModel):
    race_id: str = Field(pattern=r"^\d{4}-\d{2}$")  # e.g. "2024-05"
    season: int = Field(ge=1950, le=2100)
    round: int = Field(ge=1, le=30)
    name: str = Field(min_length=1)
    track_type: TrackType
    laps: int = Field(ge=20, le=90)
    pit_loss_s: float = Field(ge=10, le=40)      # time lost driving through the pit lane
    overtake_difficulty: float = Field(ge=0, le=1)  # 0 = easy to pass, 1 = Monaco
    rain: bool
    air_temp_c: float = Field(ge=-5, le=50)

    @model_validator(mode="after")
    def id_matches(self):
        if self.race_id != f"{self.season}-{self.round:02d}":
            raise ValueError("race_id must be '<season>-<round:02d>'")
        return self


class Result(BaseModel):
    race_id: str
    driver: str = Field(min_length=3, max_length=3)  # three-letter code, e.g. VER
    team: str = Field(min_length=1)
    grid: int = Field(ge=1, le=30)
    quali_gap_s: float = Field(ge=0, le=10)          # gap to pole in qualifying
    finish_pos: Optional[int] = Field(default=None, ge=1, le=30)  # None = did not finish
    dnf: bool
    laps_completed: int = Field(ge=0, le=90)
    pace_delta_s: Optional[float] = Field(default=None, ge=-5, le=10)  # median clean lap vs field median
    tyre_deg_s_per_lap: Optional[float] = Field(default=None, ge=-0.2, le=1.0)
    pit_stops: int = Field(ge=0, le=8)

    @model_validator(mode="after")
    def finish_consistent(self):
        if self.dnf and self.finish_pos is not None:
            raise ValueError("a driver who did not finish cannot have a finishing position")
        if not self.dnf and self.finish_pos is None:
            raise ValueError("a finisher needs a finishing position")
        return self


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS races (
    race_id TEXT PRIMARY KEY,
    season INTEGER NOT NULL,
    round INTEGER NOT NULL,
    name TEXT NOT NULL,
    track_type TEXT NOT NULL,
    laps INTEGER NOT NULL,
    pit_loss_s REAL NOT NULL,
    overtake_difficulty REAL NOT NULL,
    rain INTEGER NOT NULL,
    air_temp_c REAL NOT NULL,
    UNIQUE (season, round)
);
CREATE TABLE IF NOT EXISTS results (
    race_id TEXT NOT NULL REFERENCES races(race_id),
    driver TEXT NOT NULL,
    team TEXT NOT NULL,
    grid INTEGER NOT NULL,
    quali_gap_s REAL NOT NULL,
    finish_pos INTEGER,
    dnf INTEGER NOT NULL,
    laps_completed INTEGER NOT NULL,
    pace_delta_s REAL,
    tyre_deg_s_per_lap REAL,
    pit_stops INTEGER NOT NULL,
    PRIMARY KEY (race_id, driver)
);
CREATE INDEX IF NOT EXISTS results_race_idx ON results(race_id);
"""
