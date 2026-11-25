"""Realistic synthetic F1 seasons, used when the real FastF1 data is not available
(for tests, CI, and machines without internet access).

The generator has hidden "truth" (car speed per team and season, driver skill,
reliability), produces qualifying and race results from it with noise, and then
forgets the truth. The model only ever sees the same columns real data has.
"""
from __future__ import annotations

import numpy as np

TEAMS = ["Redline", "Silverstar", "Prancer", "Papaya", "Greenfield",
         "Alpine Blue", "Bullhorn", "Racing Bulls", "Williamsville", "Sauberg"]
DRIVERS = [("VER", 0), ("PER", 0), ("HAM", 1), ("RUS", 1), ("LEC", 2), ("SAI", 2),
           ("NOR", 3), ("PIA", 3), ("ALO", 4), ("STR", 4), ("GAS", 5), ("OCO", 5),
           ("TSU", 6), ("RIC", 6), ("HUL", 7), ("MAG", 7), ("ALB", 8), ("SAR", 8),
           ("BOT", 9), ("ZHO", 9)]
TRACKS = [  # name, type, laps, pit loss (s), overtaking difficulty (0 easy .. 1 very hard)
    ("Bahrain", "balanced", 57, 22.0, 0.25), ("Jeddah", "street", 50, 20.0, 0.4),
    ("Melbourne", "street", 58, 19.0, 0.5), ("Suzuka", "technical", 53, 22.5, 0.6),
    ("Shanghai", "balanced", 56, 23.0, 0.3), ("Miami", "street", 57, 20.0, 0.45),
    ("Imola", "technical", 63, 28.0, 0.75), ("Monaco", "street", 78, 20.0, 0.95),
    ("Montreal", "high_speed", 70, 19.0, 0.3), ("Barcelona", "technical", 66, 22.0, 0.6),
    ("Spielberg", "high_speed", 71, 21.0, 0.25), ("Silverstone", "high_speed", 52, 21.0, 0.3),
    ("Budapest", "technical", 70, 21.5, 0.8), ("Spa", "high_speed", 44, 20.0, 0.2),
    ("Zandvoort", "technical", 72, 21.0, 0.8), ("Monza", "high_speed", 53, 24.0, 0.2),
    ("Baku", "street", 51, 20.0, 0.25), ("Singapore", "street", 62, 28.0, 0.85),
    ("Austin", "balanced", 56, 20.0, 0.35), ("Mexico City", "balanced", 71, 22.0, 0.4),
    ("Interlagos", "balanced", 71, 20.0, 0.3), ("Las Vegas", "street", 50, 20.0, 0.3),
    ("Abu Dhabi", "balanced", 58, 22.0, 0.45),
]


def generate(seasons=range(2018, 2026), seed: int = 7):
    rng = np.random.default_rng(seed)
    skill = {d: rng.normal(0, 0.15) for d, _ in DRIVERS}   # s/lap, lower is faster
    car = rng.normal(0, 0.6, len(TEAMS))                    # s/lap
    track_fit = rng.normal(0, 0.15, (len(TEAMS), 4))        # team suits some track types
    reliability = rng.uniform(0.03, 0.09, len(TEAMS))       # DNF chance per race
    deg_team = rng.uniform(0.04, 0.10, len(TEAMS))          # tyre wear, s/lap of tyre age
    ttype_idx = {"street": 0, "high_speed": 1, "technical": 2, "balanced": 3}

    races, results = [], []
    for season in seasons:
        car = 0.6 * car + rng.normal(0, 0.45, len(TEAMS))   # cars change between seasons
        reliability = np.clip(reliability + rng.normal(0, 0.01, len(TEAMS)), 0.01, 0.15)
        n_rounds = min(len(TRACKS), 21 + (season >= 2022) + (season >= 2023))
        for rnd in range(1, n_rounds + 1):
            name, ttype, laps, pit_loss, ovt = TRACKS[rnd - 1]
            rain = bool(rng.random() < 0.12)
            race_id = f"{season}-{rnd:02d}"
            races.append(dict(race_id=race_id, season=season, round=rnd, name=name,
                              track_type=ttype, laps=laps, pit_loss_s=pit_loss,
                              overtake_difficulty=ovt, rain=rain,
                              air_temp_c=float(np.round(rng.uniform(14, 36), 1))))
            car_now = car + rng.normal(0, 0.12, len(TEAMS))  # upgrades during the season
            true_pace = {}
            for d, t in DRIVERS:
                p = car_now[t] + track_fit[t, ttype_idx[ttype]] + skill[d]
                if rain:
                    p += rng.normal(0, 0.35)                 # rain mixes up the order
                true_pace[d] = p
            quali = {d: true_pace[d] + rng.normal(0, 0.12) for d, _ in DRIVERS}
            order = sorted(quali, key=quali.get)
            pole_time = quali[order[0]]
            grid = {d: i + 1 for i, d in enumerate(order)}
            if rng.random() < 0.15:                          # occasional grid penalty
                d = order[rng.integers(0, 10)]
                grid = {k: v for k, v in grid.items()}
                pen = int(rng.integers(3, 6))
                others = sorted(grid, key=grid.get)
                others.remove(d)
                pos = min(len(others), grid[d] - 1 + pen)
                others.insert(pos, d)
                grid = {k: i + 1 for i, k in enumerate(others)}

            # Race: pace with noise, a start-position effect, and DNFs.
            race_score, dnf = {}, {}
            for d, t in DRIVERS:
                dnf[d] = bool(rng.random() < reliability[t] * (1.6 if rain else 1.0))
                race_pace = true_pace[d] + rng.normal(0, 0.15)
                stuck = ovt * 0.012 * grid[d] * laps / 10        # hard to pass = grid matters
                race_score[d] = race_pace * laps + stuck * laps + rng.normal(0, 3.0)
            finishers = sorted([d for d in race_score if not dnf[d]], key=race_score.get)
            med = np.median([true_pace[d] for d in true_pace])
            for d, t in DRIVERS:
                fin = None if dnf[d] else finishers.index(d) + 1
                laps_done = laps if not dnf[d] else int(rng.integers(1, laps))
                deg = deg_team[t] * (1.25 if ttype == "street" else 1.0) + rng.normal(0, 0.01)
                stops = 1 + int(deg * laps > 4.5) + int(rain and rng.random() < 0.5)
                results.append(dict(
                    race_id=race_id, driver=d, team=TEAMS[t], grid=grid[d],
                    quali_gap_s=float(round(max(0.0, quali[d] - pole_time), 3)),
                    finish_pos=fin, dnf=dnf[d], laps_completed=laps_done,
                    pace_delta_s=None if laps_done < 10 else float(round(true_pace[d] - med + rng.normal(0, 0.05), 3)),
                    tyre_deg_s_per_lap=None if laps_done < 10 else float(round(max(0.0, deg), 4)),
                    pit_stops=0 if laps_done < 10 else stops))
    return races, results
