"""Race simulation from Python.

`simulate()` uses the fast C++ module when it has been built (cpp/ -> f1sim/_racesim*.so)
and falls back to `simulate_python()`, a line-by-line Python port of the same
algorithm. The Python version exists for two reasons: to check that the C++ code
gives the same answers, and to measure how much faster C++ is.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

try:
    from . import _racesim  # type: ignore
except ImportError:  # not built yet
    _racesim = None


@dataclass
class RaceParams:
    laps: int
    pit_loss: float = 22.0
    sc_prob_per_lap: float = 0.004
    overtake_difficulty: float = 0.4
    grid_gap: float = 0.25
    lap_noise: float = 0.3


def _once(pace, pace_sd, dnf, deg, grid, r: RaceParams, rng: random.Random):
    n = len(pace)
    time = [(grid[i] - 1) * r.grid_gap for i in range(n)]
    today = [pace[i] + pace_sd[i] * rng.gauss(0, 1) for i in range(n)]
    hazard = [1 - (1 - dnf[i]) ** (1 / r.laps) for i in range(n)]
    stint = []
    for i in range(n):
        best = math.sqrt(2 * r.pit_loss / deg[i]) if deg[i] > 1e-6 else r.laps
        stint.append(min(max(int(best), 10), max(10, r.laps // 2)))
    age, laps_done, retired = [0] * n, [0] * n, [False] * n
    order = sorted(range(n), key=lambda i: grid[i])
    for lap in range(1, r.laps + 1):
        sc = rng.random() < r.sc_prob_per_lap
        if sc:
            lead, k = None, 0
            for i in order:
                if retired[i]:
                    continue
                lead = time[i] if lead is None else lead
                time[i] = lead + 0.8 * k
                k += 1
        pitted = [False] * n
        for i in range(n):
            if retired[i]:
                continue
            if rng.random() < hazard[i]:
                retired[i] = True
                continue
            lt = today[i] + deg[i] * age[i] + r.lap_noise * rng.gauss(0, 1)
            if (age[i] >= stint[i] or (sc and age[i] > stint[i] * 0.6)) and lap < r.laps - 3:
                lt += r.pit_loss * (0.5 if sc else 1.0)
                age[i], pitted[i] = 0, True
            else:
                age[i] += 1
            time[i] += lt
            laps_done[i] = lap
        running = [i for i in order if not retired[i]]
        out = [i for i in order if retired[i]]
        for k in range(1, len(running)):
            c, a = running[k], running[k - 1]
            if time[c] < time[a]:
                if pitted[a] or rng.random() > r.overtake_difficulty:
                    running[k], running[k - 1] = a, c
                else:
                    time[c] = time[a] + 0.2
        running.sort(key=lambda i: time[i])
        order = running + out
    fin = [i for i in order if not retired[i]]
    dnfs = sorted([i for i in order if retired[i]], key=lambda i: -laps_done[i])
    return fin + dnfs, retired


def simulate_python(pace, pace_sd, dnf_prob, deg, grid, laps, pit_loss=22.0,
                    sc_prob_per_lap=0.004, overtake_difficulty=0.4, grid_gap=0.25, lap_noise=0.3,
                    n_sims=10_000, seed=42, **_):
    n = len(pace)
    r = RaceParams(laps, pit_loss, sc_prob_per_lap, overtake_difficulty, grid_gap, lap_noise)
    rng = random.Random(seed)
    wins, pods, dnfs, fins, pos_sum = [0] * n, [0] * n, [0] * n, [0] * n, [0.0] * n
    for _ in range(n_sims):
        order, retired = _once(pace, pace_sd, dnf_prob, deg, grid, r, rng)
        if not retired[order[0]]:
            wins[order[0]] += 1
        for p, i in enumerate(order):
            if retired[i]:
                dnfs[i] += 1
                continue
            pods[i] += p < 3
            fins[i] += 1
            pos_sum[i] += p + 1
    return {"win_prob": [w / n_sims for w in wins], "podium_prob": [x / n_sims for x in pods],
            "mean_finish": [pos_sum[i] / fins[i] if fins[i] else 0.0 for i in range(n)],
            "dnf_rate": [x / n_sims for x in dnfs], "n_sims": n_sims}


def simulate(*args, n_sims=100_000, n_threads=0, seed=42, **kw):
    """Use C++ if available, otherwise Python (much slower)."""
    if _racesim is not None:
        return _racesim.simulate(*args, n_sims=n_sims, n_threads=n_threads, seed=seed, **kw)
    return simulate_python(*args, n_sims=n_sims, seed=seed, **kw)
