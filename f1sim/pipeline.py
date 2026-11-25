"""End to end: data -> features -> pace model -> race simulator -> evaluation.

    python -m f1sim.pipeline --db f1_sample.sqlite          # sample data
    python -m f1sim.pipeline --db f1.sqlite --sims 100000   # real data from fastf1_loader

Splits by time: fit on seasons up to `val_season - 1`, tune post-processing on
`val_season`, and test on `test_from` onwards. The test seasons are never touched
until the final evaluation.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data.db import connect, load_frame
from .features import build
from .metrics import calibration, race_scores
from .model import apply_post_processing, fit_post_processing, predict, train
from .simulate import _racesim, simulate


def grid_prior(train_df: pd.DataFrame, max_grid: int = 22) -> np.ndarray:
    """Baseline: P(win | starting slot), learned from training seasons (+1 smoothing)."""
    wins = train_df[train_df["finish_pos"] == 1]["grid"].value_counts()
    counts = np.array([wins.get(g, 0) for g in range(1, max_grid + 1)], float) + 1.0
    races = train_df["race_id"].nunique() + max_grid
    return counts / races


def _simulate_race(g: pd.DataFrame, pace_sd: float, sim: dict, sims: int, seed: int):
    return simulate(g["pred_pace"].tolist(), [pace_sd] * len(g), g["pred_dnf"].tolist(),
                    g["pred_deg"].tolist(), g["grid"].astype(int).tolist(), int(g["laps"].iloc[0]),
                    pit_loss=float(g["pit_loss_s"].iloc[0]),
                    overtake_difficulty=float(min(0.98, g["overtake_difficulty"].iloc[0] * sim["overtake_scale"])),
                    grid_gap=sim["grid_gap"], n_sims=sims, seed=seed)


def tune_simulator(val_pred: pd.DataFrame, pace_sd: float, sims: int = 4000) -> tuple[dict, list]:
    """Pick the simulator's track-position settings that best explain the validation season.

    Only two global knobs are tuned: how hard overtaking is (a multiplier on each
    track's value) and how much time separates grid slots at the start. Everything
    is judged by log loss on validation races; the test seasons are not used.
    """
    trials = []
    for scale in (1.0, 1.6):
        for gap in (0.5, 1.0, 2.0, 3.0):
            for sd_scale in (0.5, 1.0):
                sim = {"overtake_scale": scale, "grid_gap": gap, "sd_scale": sd_scale}
                losses = []
                for k, (_, g) in enumerate(val_pred.groupby("race_id", sort=True)):
                    g = g.reset_index(drop=True)
                    w = g.index[g["finish_pos"] == 1]
                    if len(w) != 1:
                        continue
                    p = np.array(_simulate_race(g, pace_sd * sd_scale, sim, sims, seed=1000 + k)["win_prob"])
                    losses.append(race_scores(p, int(w[0]))["log_loss"])
                trials.append({**sim, "val_log_loss": round(float(np.mean(losses)), 4)})
    best = min(trials, key=lambda t: t["val_log_loss"])
    return {k: best[k] for k in ("overtake_scale", "grid_gap", "sd_scale")}, trials


def run(db: str, val_season: int, test_from: int, sims: int, out_dir: str = "results", seed: int = 0,
        test_to: int | None = None,
        tune: bool = True):
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    df = build(load_frame(connect(db)))
    fit_df = df[df["season"] < val_season]
    val_df = df[df["season"] == val_season]
    test_df = df[(df["season"] >= test_from) & (df["season"] <= (test_to or 9999))]
    if len(test_df) == 0:
        raise SystemExit(f"no test data from {test_from}")

    t0 = time.time()
    model, hist = train(fit_df, val_df, seed=seed)
    post = fit_post_processing(model, val_df)
    # How uncertain is a pace prediction? Use the model's error spread on validation.
    vp = predict(model, val_df)
    resid = (vp["pred_pace"] - val_df["pace_delta_s"].to_numpy())
    post["pace_sd"] = float(np.nanstd(resid))
    post["train_seconds"] = round(time.time() - t0, 1)
    post["epochs"] = len(hist["loss"])
    model.save(out / "pace_model.keras")

    val_pred = apply_post_processing(vp, val_df, post).join(
        val_df[["laps", "pit_loss_s", "overtake_difficulty", "finish_pos"]])
    default = {"overtake_scale": 1.0, "grid_gap": 0.25, "sd_scale": 1.0}
    sim_params, trials = tune_simulator(val_pred, post["pace_sd"]) if tune else (default, [])
    post["simulator"] = sim_params
    post["simulator_tuning_trials"] = trials

    # Blend: final = w * simulator + (1 - w) * grid prior, with w chosen on validation.
    prior = grid_prior(fit_df)
    val_sets = []
    for k, (_, g) in enumerate(val_pred.groupby("race_id", sort=True)):
        g = g.reset_index(drop=True)
        w_idx = g.index[g["finish_pos"] == 1]
        if len(w_idx) != 1:
            continue
        p_sim = np.array(_simulate_race(g, post["pace_sd"] * sim_params["sd_scale"], sim_params, 4000,
                                        seed=2000 + k)["win_prob"])
        p_pri = prior[np.clip(g["grid"].to_numpy() - 1, 0, len(prior) - 1)]
        val_sets.append((p_sim, p_pri / p_pri.sum(), int(w_idx[0])))
    blend_trials = []
    for wgt in (0.0, 0.25, 0.5, 0.75, 1.0):
        ll = np.mean([race_scores(wgt * a + (1 - wgt) * b, w)["log_loss"] for a, b, w in val_sets])
        blend_trials.append({"sim_weight": wgt, "val_log_loss": round(float(ll), 4)})
    post["blend_weight"] = min(blend_trials, key=lambda t: t["val_log_loss"])["sim_weight"]
    post["blend_trials"] = blend_trials

    prior = grid_prior(pd.concat([fit_df, val_df]))
    pred = apply_post_processing(predict(model, test_df), test_df, post)
    pred = pred.join(test_df[["season", "round", "name", "laps", "pit_loss_s",
                              "overtake_difficulty", "finish_pos", "dnf"]])

    races, scores_model, scores_blend, scores_pole, scores_prior = [], [], [], [], []
    all_p, all_y = [], []
    sim_seconds = 0.0
    for race_id, g in pred.groupby("race_id", sort=True):
        g = g.reset_index(drop=True)
        winners = g.index[g["finish_pos"] == 1]
        if len(winners) != 1:
            continue
        w = int(winners[0])
        t = time.time()
        s = _simulate_race(g, post["pace_sd"] * sim_params["sd_scale"], sim_params, sims, seed=seed + len(races))
        sim_seconds += time.time() - t
        p = np.array(s["win_prob"])
        pole = (g["grid"] == 1).to_numpy().astype(float)
        pri = prior[np.clip(g["grid"].to_numpy() - 1, 0, len(prior) - 1)]
        pri = pri / pri.sum()
        scores_model.append(race_scores(p, w))
        bw = post["blend_weight"]
        p_blend = bw * p + (1 - bw) * pri
        scores_blend.append(race_scores(p_blend, w))
        scores_pole.append(race_scores(pole, w))
        scores_prior.append(race_scores(pri, w))
        all_p.append(p_blend)
        all_y.append((np.arange(len(g)) == w).astype(float))
        drivers = []
        for i in np.argsort(-p_blend):
            drivers.append({
                "driver": g.at[i, "driver"], "team": g.at[i, "team"], "grid": int(g.at[i, "grid"]),
                "win_prob": round(float(p_blend[i]), 4), "sim_win_prob": round(float(p[i]), 4),
                "podium_prob": round(float(s["podium_prob"][i]), 4),
                "mean_finish": round(float(s["mean_finish"][i]), 2),
                "dnf_prob": round(float(s["dnf_rate"][i]), 4),
                "pred_pace": round(float(g.at[i, "pred_pace"]), 3),
                "actual_finish": None if g.at[i, "dnf"] else int(g.at[i, "finish_pos"])})
        races.append({"race_id": race_id, "season": int(g["season"].iloc[0]),
                      "round": int(g["round"].iloc[0]), "name": g["name"].iloc[0],
                      "winner": g.at[w, "driver"], "predicted_favourite": drivers[0]["driver"],
                      "drivers": drivers})

    def agg(rows):
        return {k: round(float(np.mean([r[k] for r in rows])), 4) for k in ("top1", "log_loss", "brier")}

    cal_rows, ece = calibration(np.concatenate(all_p), np.concatenate(all_y))
    summary = {
        "data": db, "fit_seasons": f"{fit_df.season.min()}-{fit_df.season.max()}",
        "validation_season": val_season, "test_seasons": f"{test_df.season.min()}-{test_df.season.max()}",
        "test_races": len(races), "sims_per_race": sims,
        "simulator": "C++" if _racesim is not None else "Python",
        "simulation_seconds_total": round(sim_seconds, 1),
        "simulator_only": agg(scores_model), "model": agg(scores_blend),
        "baseline_pole_wins": agg(scores_pole),
        "baseline_grid_prior": agg(scores_prior),
        "calibration": cal_rows, "calibration_error": round(ece, 4),
        "post_processing": post,
    }
    json.dump(summary, open(out / "metrics.json", "w"), indent=2)
    json.dump({"summary": {k: summary[k] for k in ("test_seasons", "test_races", "model", "simulator_only",
                                                     "baseline_pole_wins", "baseline_grid_prior")},
               "races": races}, open(out / "predictions.json", "w"), indent=2)
    _plot_calibration(cal_rows, out / "calibration.png")
    return summary


def _plot_calibration(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = [r["mean_predicted"] for r in rows]
    y = [r["actual_rate"] for r in rows]
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfectly calibrated")
    ax.plot(x, y, "o-", label="model")
    for r in rows:
        ax.annotate(str(r["n"]), (r["mean_predicted"], r["actual_rate"]), fontsize=7,
                    textcoords="offset points", xytext=(4, -10))
    ax.set_xlabel("predicted win probability")
    ax.set_ylabel("actual win rate")
    ax.set_title("Calibration on test seasons")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="f1_sample.sqlite")
    ap.add_argument("--val-season", type=int, default=2023)
    ap.add_argument("--test-from", type=int, default=2024)
    ap.add_argument("--test-to", type=int, default=None, help="last test season (default: newest)")
    ap.add_argument("--sims", type=int, default=20_000)
    ap.add_argument("--out", default="results")
    ap.add_argument("--no-tune", action="store_true", help="skip simulator tuning on validation")
    a = ap.parse_args()
    s = run(a.db, a.val_season, a.test_from, a.sims, a.out, tune=not a.no_tune, test_to=a.test_to)
    print(json.dumps({k: s[k] for k in ("test_races", "simulator_only", "model", "baseline_pole_wins",
                                          "baseline_grid_prior", "calibration_error",
                                          "simulation_seconds_total")}, indent=2))
    print("simulator settings chosen on validation:", s["post_processing"]["simulator"])
    print("blend weight on simulator chosen on validation:", s["post_processing"]["blend_weight"])


if __name__ == "__main__":
    main()
