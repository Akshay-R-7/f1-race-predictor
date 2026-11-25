"""Offline tests for the Jolpica loader, using tiny hand-made API responses."""
import pytest

from f1sim.data import jolpica_loader as jl
from f1sim.data.db import connect, write


def drv(i, code):
    return {"driverId": i, "code": code}


RESULTS = [
    {"position": "1", "grid": "1", "laps": "30", "status": "Finished",
     "Driver": drv("max_verstappen", "VER"), "Constructor": {"name": "Red Bull"}},
    {"position": "2", "grid": "3", "laps": "30", "status": "+1 Lap",
     "Driver": drv("norris", "NOR"), "Constructor": {"name": "McLaren"}},
    {"position": "3", "grid": "0", "laps": "12", "status": "Engine",
     "Driver": drv("hamilton", "HAM"), "Constructor": {"name": "Ferrari"}},
]
QUALI = [
    {"Driver": drv("max_verstappen", "VER"), "Q1": "1:30.000", "Q2": "1:29.500", "Q3": "1:29.000"},
    {"Driver": drv("norris", "NOR"), "Q1": "1:30.200", "Q2": "1:29.800", "Q3": ""},
    {"Driver": drv("hamilton", "HAM"), "Q1": "1:31.000"},
]


def make_laps(n=30, sc_lap=15):
    laps = []
    for k in range(1, n + 1):
        base = 90.0 if k != sc_lap else 120.0  # a safety-car lap
        timings = [{"driverId": "max_verstappen", "time": f"1:{base - 60 - 0.5:06.3f}"},
                   {"driverId": "norris", "time": f"1:{base - 60:06.3f}"}]
        if k <= 12:
            timings.append({"driverId": "hamilton", "time": f"1:{base - 60 + 0.3:06.3f}"})
        laps.append({"number": str(k), "Timings": timings})
    return laps


PITS = [{"driverId": "max_verstappen", "lap": "10", "stop": "1"},
        {"driverId": "norris", "lap": "11", "stop": "1"}]
RACE = {"season": "2024", "round": "3", "raceName": "Test GP",
        "Circuit": {"circuitId": "monaco"}, "Results": RESULTS}


def test_lap_seconds():
    assert jl.lap_seconds("1:33.523") == pytest.approx(93.523)
    assert jl.lap_seconds("58.1") == pytest.approx(58.1)
    assert jl.lap_seconds("") is None and jl.lap_seconds(None) is None
    assert jl.lap_seconds("garbage") is None


def test_quali_gaps_use_best_session_and_missing_times():
    gaps = jl.quali_gaps(QUALI, jl.driver_codes(RESULTS))
    assert gaps == {"VER": 0.0, "NOR": pytest.approx(0.8), "HAM": pytest.approx(2.0)}


def test_finished_status():
    assert jl.is_finished("Finished") and jl.is_finished("+2 Laps") and jl.is_finished("Lapped")
    assert not jl.is_finished("Engine") and not jl.is_finished("Retired")


def test_pace_ignores_safety_car_and_pit_laps():
    pace = jl.pace_deltas(make_laps(), jl.pit_laps_by_driver(PITS), jl.driver_codes(RESULTS))
    # Verstappen is 0.5 s/lap faster than Norris; the safety-car lap must not change that.
    assert pace["VER"] - pace["NOR"] == pytest.approx(-0.5)
    assert "HAM" in pace  # 10 clean laps before retiring


def test_build_race_rows_pass_validation(tmp_path):
    race_row, rows = jl.build_race(RACE, QUALI, make_laps(), PITS)
    assert race_row["race_id"] == "2024-03" and race_row["track_type"] == "street"
    ham = next(r for r in rows if r["driver"] == "HAM")
    assert ham["dnf"] and ham["finish_pos"] is None and ham["grid"] == 20  # pit-lane start
    assert all(r["tyre_deg_s_per_lap"] is None for r in rows)
    assert {r["driver"]: r["pit_stops"] for r in rows} == {"VER": 1, "NOR": 1, "HAM": 0}
    out = write(connect(tmp_path / "t.sqlite"), [race_row], rows)
    assert out == {"races": 1, "results": 3, "rejected": 0}


class FakeClient(jl.Client):
    """Serves two pages where one race's laps are split across the page boundary."""

    def __init__(self, tmp_path):
        super().__init__(cache_dir=tmp_path, delay_s=0)
        laps = make_laps(4, sc_lap=99)
        self.pages = {
            0: {"MRData": {"total": "150", "RaceTable": {"Races": [
                {"season": "2024", "round": "3", "Laps": laps[:2]}]}}},
            100: {"MRData": {"total": "150", "RaceTable": {"Races": [
                {"season": "2024", "round": "3", "Laps": laps[2:]}]}}},
        }

    def get(self, path, fresh=False, **params):
        return self.pages[params["offset"]]


def test_pagination_merges_pages(tmp_path):
    races = FakeClient(tmp_path).get_all("2024/3/laps.json", "RaceTable", "Laps")
    assert len(races) == 1
    assert [lap["number"] for lap in races[0]["Laps"]] == ["1", "2", "3", "4"]


def test_cache_means_no_second_request(tmp_path, monkeypatch):
    calls = []

    class Resp:
        def __init__(self): self.body = b'{"MRData": {"total": "0"}}'
        def read(self): return self.body
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(jl.urllib.request, "urlopen", lambda *a, **k: calls.append(1) or Resp())
    c = jl.Client(cache_dir=tmp_path, delay_s=0)
    assert c.get("2024.json") == c.get("2024.json")
    assert len(calls) == 1


def test_timeout_is_retried(tmp_path, monkeypatch):
    attempts = []

    class Resp:
        def read(self): return b'{"ok": 1}'
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def flaky(*a, **k):
        attempts.append(1)
        if len(attempts) == 1:
            raise TimeoutError("The read operation timed out")
        return Resp()

    monkeypatch.setattr(jl.urllib.request, "urlopen", flaky)
    monkeypatch.setattr(jl.time, "sleep", lambda s: None)
    assert jl.Client(cache_dir=tmp_path, delay_s=0).get("x.json") == {"ok": 1}
    assert len(attempts) == 2
