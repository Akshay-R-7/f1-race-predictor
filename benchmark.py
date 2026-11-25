"""How much faster is C++ than Python for the same simulation?  python benchmark.py"""
import json
import os
import time

from f1sim.simulate import _racesim, simulate_python

FIELD = dict(pace=[i * 0.08 for i in range(20)], pace_sd=[0.15] * 20, dnf_prob=[0.06] * 20,
             deg=[0.06] * 20, grid=list(range(1, 21)), laps=57, pit_loss=22.0)

if __name__ == "__main__":
    n_py = 2_000
    t = time.perf_counter(); simulate_python(**FIELD, n_sims=n_py); py = time.perf_counter() - t
    py_per = py / n_py
    rows = {"python_s_per_race": py_per, "python_100k_estimate_s": py_per * 100_000}
    if _racesim:
        for threads in (1, os.cpu_count()):
            t = time.perf_counter(); _racesim.simulate(**FIELD, n_sims=100_000, n_threads=threads)
            rows[f"cpp_{threads}_threads_100k_s"] = time.perf_counter() - t
        rows["speedup_1_thread"] = rows["python_100k_estimate_s"] / rows["cpp_1_threads_100k_s"]
        rows[f"speedup_{os.cpu_count()}_threads"] = rows["python_100k_estimate_s"] / rows[f"cpp_{os.cpu_count()}_threads_100k_s"]
    print(json.dumps({k: round(v, 4) for k, v in rows.items()}, indent=2))
    os.makedirs("results", exist_ok=True)
    json.dump(rows, open("results/benchmark.json", "w"), indent=2)
