"""FastAPI backend for the dashboard.

    uvicorn api.main:app --reload

Serves the predictions written by `python -m f1sim.pipeline`, and a /simulate
endpoint that runs the C++ simulator live for a "what if" race.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

from f1sim.simulate import _racesim, simulate

PRED_PATH = Path(os.environ.get("PREDICTIONS", "results/predictions.json"))

app = FastAPI(title="F1 Race Predictor")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])


def _data() -> dict:
    if not PRED_PATH.exists():
        raise HTTPException(503, "no predictions yet: run `python -m f1sim.pipeline` first")
    return json.loads(PRED_PATH.read_text())


@app.get("/api/health")
def health():
    return {"status": "ok", "simulator": "C++" if _racesim is not None else "Python"}


@app.get("/api/summary")
def summary():
    return _data()["summary"]


@app.get("/api/races")
def races():
    return [{k: r[k] for k in ("race_id", "season", "round", "name", "winner", "predicted_favourite")}
            for r in _data()["races"]]


@app.get("/api/races/{race_id}")
def race(race_id: str):
    for r in _data()["races"]:
        if r["race_id"] == race_id:
            return r
    raise HTTPException(404, f"race {race_id} not found")


class SimDriver(BaseModel):
    driver: str = Field(min_length=1, max_length=3)
    pace: float = Field(ge=-5, le=5)
    grid: int = Field(ge=1, le=30)
    dnf_prob: float = Field(default=0.05, ge=0, le=1)
    deg: float = Field(default=0.06, ge=0, le=0.5)


class SimRequest(BaseModel):
    drivers: list[SimDriver] = Field(min_length=2, max_length=26)
    laps: int = Field(default=57, ge=10, le=90)
    overtake_difficulty: float = Field(default=0.4, ge=0, le=1)
    n_sims: int = Field(default=20_000, ge=100, le=200_000)

    @model_validator(mode="after")
    def unique_grid(self):
        if len({d.grid for d in self.drivers}) != len(self.drivers):
            raise ValueError("grid positions must be unique")
        return self


@app.post("/api/simulate")
def simulate_race(req: SimRequest):
    d = req.drivers
    s = simulate([x.pace for x in d], [0.15] * len(d), [x.dnf_prob for x in d], [x.deg for x in d],
                 [x.grid for x in d], req.laps, overtake_difficulty=req.overtake_difficulty,
                 n_sims=req.n_sims if _racesim is not None else min(req.n_sims, 2000))
    return {"n_sims": s["n_sims"], "results": sorted(
        [{"driver": x.driver, "grid": x.grid, "win_prob": s["win_prob"][i],
          "podium_prob": s["podium_prob"][i], "mean_finish": s["mean_finish"][i]}
         for i, x in enumerate(d)], key=lambda r: -r["win_prob"])}
