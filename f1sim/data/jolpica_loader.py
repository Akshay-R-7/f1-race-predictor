"""Load real F1 data from the free Jolpica API (the Ergast replacement). No account needed.

    python -m f1sim.data.jolpica_loader --from 2018 --to 2025 --db f1.sqlite

What you get: results, grid, qualifying times, lap times and pit stops.
What you lose compared with FastF1:
  * tyre wear: Jolpica has no tyre data, so tyre_deg_s_per_lap stays empty
    (the model already ignores missing values through sample weights);
  * weather: rain is set to False and air temperature to 25 C;
  * safety-car flags: slow laps are filtered out with a simple rule instead.

Rate limits: the API allows short bursts (4 requests/s) but only a few hundred
requests per hour. Every response is saved in ./jolpica_cache, so if the run stops
you can restart it and it skips what it already has. One race needs about 15
requests (lap times come 100 records per page), so 8 seasons take a few hours
the first time and a few seconds after that.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import http.client
import json
import logging
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .db import connect, write

log = logging.getLogger(__name__)

BASE = "https://api.jolpi.ca/ergast/f1/"
PAGE = 100  # the API's maximum page size

# Rough circuit classification by Jolpica circuitId; unknown circuits are "balanced".
TRACK_TYPE = {
    "monaco": "street", "baku": "street", "marina_bay": "street", "jeddah": "street",
    "albert_park": "street", "miami": "street", "vegas": "street",
    "monza": "high_speed", "spa": "high_speed", "silverstone": "high_speed",
    "red_bull_ring": "high_speed", "villeneuve": "high_speed",
    "hungaroring": "technical", "catalunya": "technical", "suzuka": "technical",
    "zandvoort": "technical", "imola": "technical", "madring": "street",
}
OVERTAKE = {"street": 0.6, "technical": 0.7, "high_speed": 0.25, "balanced": 0.4}
PIT_LOSS = {"street": 21.0, "technical": 22.0, "high_speed": 21.0, "balanced": 22.0}
DEFAULT_AIR_TEMP_C = 25.0


# --------------------------------------------------------------------------- HTTP

class Client:
    """Tiny HTTP client: disk cache, a fixed delay between real requests, retry on 429."""

    def __init__(self, cache_dir: str | Path = "jolpica_cache", delay_s: float = 8.0,
                 max_retries: int = 8):
        self.cache = Path(cache_dir)
        self.cache.mkdir(exist_ok=True)
        self.delay_s = delay_s
        self.max_retries = max_retries
        self._last = 0.0

    def _path(self, url: str) -> Path:
        return self.cache / (hashlib.sha1(url.encode()).hexdigest() + ".json")

    def get(self, path: str, fresh: bool = False, **params) -> dict:
        """fresh=True skips the cache (used for the current season, which keeps growing)."""
        url = BASE + path + ("?" + urllib.parse.urlencode(params) if params else "")
        cached = self._path(url)
        if cached.exists() and not fresh:
            return json.loads(cached.read_text(encoding="utf-8"))

        wait = 60.0
        for attempt in range(self.max_retries):
            pause = self.delay_s - (time.monotonic() - self._last)
            if pause > 0:
                time.sleep(pause)
            self._last = time.monotonic()
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "f1-race-predictor"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                cached.write_text(json.dumps(data), encoding="utf-8")
                return data
            except urllib.error.HTTPError as e:
                if e.code == 429 or e.code >= 500:
                    retry_after = e.headers.get("Retry-After")
                    sleep_s = float(retry_after) if retry_after and retry_after.isdigit() else wait
                    log.warning("HTTP %s on %s, waiting %.0f s (attempt %d)",
                                e.code, url, sleep_s, attempt + 1)
                    time.sleep(sleep_s)
                    wait = min(wait * 2, 900)
                    continue
                raise
            except (OSError, http.client.HTTPException, json.JSONDecodeError) as e:
                # OSError covers URLError, timeouts and dropped connections; the server
                # being slow for a moment shouldn't stop a run that takes hours.
                log.warning("network error on %s: %r, retrying in %.0f s", url, e, wait / 2)
                time.sleep(wait / 2)
                wait = min(wait * 2, 900)
        raise RuntimeError(f"gave up on {url} after {self.max_retries} attempts")

    def get_all(self, path: str, table: str, item: str, fresh: bool = False) -> list[dict]:
        """Follow offset pagination and return the races in one merged list.

        Lap and pit-stop pages split one race across many pages, so items with the
        same race are merged (laps from page 2 are appended to the race from page 1).
        """
        races: dict[tuple, dict] = {}
        offset = 0
        while True:
            mr = self.get(path, fresh=fresh, limit=PAGE, offset=offset)["MRData"]
            for race in mr[table]["Races"]:
                key = (race["season"], race["round"])
                if key not in races:
                    races[key] = {**race, item: []}
                races[key][item].extend(race.get(item, []))
            offset += PAGE
            if offset >= int(mr["total"]):
                return list(races.values())


# ------------------------------------------------------------------ pure parsing

def lap_seconds(s: str | None) -> float | None:
    """'1:33.523' -> 93.523, '58.1' -> 58.1, '' or None -> None."""
    if not s:
        return None
    try:
        if ":" in s:
            m, sec = s.split(":")
            return int(m) * 60 + float(sec)
        return float(s)
    except ValueError:
        return None


def driver_codes(results: list[dict]) -> dict[str, str]:
    """Map Jolpica driverId -> three-letter code (VER, HAM...)."""
    out = {}
    for r in results:
        d = r["Driver"]
        out[d["driverId"]] = (d.get("code") or d["driverId"][:3]).upper()
    return out


def quali_gaps(quali: list[dict], codes: dict[str, str]) -> dict[str, float]:
    """Gap to pole in seconds using each driver's best time across Q1-Q3."""
    best = {}
    for q in quali:
        times = [lap_seconds(q.get(k)) for k in ("Q1", "Q2", "Q3")]
        times = [t for t in times if t]
        if times:
            best[codes.get(q["Driver"]["driverId"], q["Driver"]["driverId"][:3].upper())] = min(times)
    if not best:
        return {}
    pole = min(best.values())
    return {d: round(min(10.0, t - pole), 3) for d, t in best.items()}


def is_finished(status: str) -> bool:
    """'Finished', '+1 Lap' and (newer data) 'Lapped' count as finished."""
    return status in ("Finished", "Lapped") or status.startswith("+")


def pace_deltas(laps: list[dict], pit_laps: dict[str, set[int]], codes: dict[str, str],
                min_laps: int = 5) -> dict[str, float]:
    """Median clean lap per driver minus the median of those medians.

    Jolpica has no track-status flags, so 'clean' is approximated:
      * drop lap 1 (standing start) and each driver's pit in-lap and out-lap;
      * drop whole laps where the field median is >10% slower than the race median
        (safety car / red flag);
      * drop a driver's laps more than 7% slower than their own median (spins, damage).
    """
    by_lap: dict[int, list[tuple[str, float]]] = {}
    for lap in laps:
        n = int(lap["number"])
        if n == 1:
            continue
        for t in lap["Timings"]:
            sec = lap_seconds(t.get("time"))
            drv = t["driverId"]
            if sec is None or n in pit_laps.get(drv, ()) or (n - 1) in pit_laps.get(drv, ()):
                continue
            by_lap.setdefault(n, []).append((drv, sec))
    if not by_lap:
        return {}

    lap_medians = {n: statistics.median(s for _, s in v) for n, v in by_lap.items()}
    race_median = statistics.median(lap_medians.values())
    per_driver: dict[str, list[float]] = {}
    for n, v in by_lap.items():
        if lap_medians[n] > 1.10 * race_median:
            continue
        for drv, sec in v:
            per_driver.setdefault(drv, []).append(sec)

    med = {}
    for drv, ts in per_driver.items():
        m = statistics.median(ts)
        ts = [t for t in ts if t <= 1.07 * m]
        if len(ts) >= min_laps:
            med[codes.get(drv, drv[:3].upper())] = statistics.median(ts)
    if not med:
        return {}
    field = statistics.median(med.values())
    return {d: round(m - field, 3) for d, m in med.items()}


def pit_laps_by_driver(pitstops: list[dict]) -> dict[str, set[int]]:
    out: dict[str, set[int]] = {}
    for p in pitstops:
        out.setdefault(p["driverId"], set()).add(int(p["lap"]))
    return out


def build_race(race: dict, quali: list[dict], laps: list[dict], pitstops: list[dict]):
    """Turn one race's API data into (race_row, result_rows) for db.write()."""
    season, rnd = int(race["season"]), int(race["round"])
    race_id = f"{season}-{rnd:02d}"
    results = race["Results"]
    codes = driver_codes(results)
    circuit = race["Circuit"]["circuitId"]
    ttype = TRACK_TYPE.get(circuit, "balanced")
    pits = pit_laps_by_driver(pitstops)
    gaps = quali_gaps(quali, codes)
    pace = pace_deltas(laps, pits, codes)

    race_row = dict(
        race_id=race_id, season=season, round=rnd, name=race["raceName"], track_type=ttype,
        laps=max(int(r["laps"]) for r in results), pit_loss_s=PIT_LOSS[ttype],
        overtake_difficulty=OVERTAKE[ttype], rain=False, air_temp_c=DEFAULT_AIR_TEMP_C)

    rows = []
    for r in results:
        drv_id = r["Driver"]["driverId"]
        code = codes[drv_id]
        finished = is_finished(r["status"])
        grid = int(r["grid"])
        rows.append(dict(
            race_id=race_id, driver=code, team=r["Constructor"]["name"],
            grid=grid if grid > 0 else 20,  # 0 means a pit-lane start
            quali_gap_s=gaps.get(code, 5.0),
            finish_pos=int(r["position"]) if finished else None, dnf=not finished,
            laps_completed=int(r["laps"]), pace_delta_s=pace.get(code),
            tyre_deg_s_per_lap=None, pit_stops=min(8, len(pits.get(drv_id, ())))))
    return race_row, rows


# ------------------------------------------------------------------------ loading

def load_season(client: Client, year: int):
    races, results = [], []
    # The current season gains a race every week or two, so its season-wide lists are
    # always fetched fresh. Per-race laps and pit stops never change once a race is done.
    fresh = year >= datetime.date.today().year
    by_round = {int(r["round"]): r for r in
                client.get_all(f"{year}/results.json", "RaceTable", "Results", fresh=fresh)}
    quali = {int(r["round"]): r["QualifyingResults"] for r in
             client.get_all(f"{year}/qualifying.json", "RaceTable", "QualifyingResults", fresh=fresh)}
    for rnd, race in sorted(by_round.items()):
        try:
            laps = client.get_all(f"{year}/{rnd}/laps.json", "RaceTable", "Laps")
            stops = client.get_all(f"{year}/{rnd}/pitstops.json", "RaceTable", "PitStops")
            race_row, rows = build_race(race, quali.get(rnd, []),
                                        laps[0]["Laps"] if laps else [],
                                        stops[0]["PitStops"] if stops else [])
        except Exception as e:  # keep going; one bad race shouldn't stop a season
            log.warning("skipping %s round %s: %s", year, rnd, e)
            continue
        races.append(race_row)
        results.extend(rows)
        log.info("%s round %d %s: %d drivers", year, rnd, race["raceName"], len(rows))
    return races, results


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--from", dest="start", type=int, default=2018)
    ap.add_argument("--to", dest="end", type=int, default=2025)
    ap.add_argument("--db", default="f1.sqlite")
    ap.add_argument("--delay", type=float, default=8.0,
                    help="seconds between real API requests (cached ones are instant)")
    args = ap.parse_args()
    client = Client(delay_s=args.delay)
    con = connect(args.db)
    for y in range(args.start, args.end + 1):
        races, results = load_season(client, y)
        print(y, write(con, races, results, reject_log="rejected_rows.jsonl"), flush=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
