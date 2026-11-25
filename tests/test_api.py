import json

import pytest
from fastapi.testclient import TestClient

import api.main as api

SAMPLE = {"summary": {"test_races": 1}, "races": [{
    "race_id": "2024-01", "season": 2024, "round": 1, "name": "Bahrain", "winner": "VER",
    "predicted_favourite": "VER",
    "drivers": [{"driver": "VER", "team": "Redline", "grid": 1, "win_prob": 0.6}]}]}


@pytest.fixture
def client(tmp_path, monkeypatch):
    p = tmp_path / "p.json"
    p.write_text(json.dumps(SAMPLE))
    monkeypatch.setattr(api, "PRED_PATH", p)
    return TestClient(api.app)


def test_races_and_detail(client):
    assert client.get("/api/races").json()[0]["race_id"] == "2024-01"
    assert client.get("/api/races/2024-01").json()["drivers"][0]["driver"] == "VER"
    assert client.get("/api/races/1999-01").status_code == 404


def test_missing_predictions_gives_503(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "PRED_PATH", tmp_path / "nope.json")
    assert TestClient(api.app).get("/api/races").status_code == 503


def test_simulate_validates_and_runs(client):
    body = {"drivers": [{"driver": "AAA", "pace": -1.0, "grid": 2},
                        {"driver": "BBB", "pace": 0.0, "grid": 1}], "laps": 30, "n_sims": 2000}
    r = client.post("/api/simulate", json=body).json()
    assert r["results"][0]["driver"] == "AAA" and abs(sum(x["win_prob"] for x in r["results"]) - 1) < 0.02
    body["drivers"][1]["grid"] = 2  # duplicate grid slot
    assert client.post("/api/simulate", json=body).status_code == 422
