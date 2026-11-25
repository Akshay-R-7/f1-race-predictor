"""Metrics for "who wins this race?" predictions.

Each race gives a probability per driver. We judge them with:
- top-1 accuracy: did the most likely driver actually win?
- log loss: -log(probability given to the real winner). Punishes confident mistakes hard.
- Brier score: mean squared error between probabilities and what happened (1 for the
  winner, 0 for everyone else), summed over drivers. Lower is better.
- calibration: of all the times we said "about 30%", did it happen about 30% of the time?
"""
from __future__ import annotations

import numpy as np

EPS = 1e-4  # never give the real winner exactly zero, or log loss becomes infinite


def race_scores(probs: np.ndarray, winner: int) -> dict:
    p = np.clip(probs, EPS, 1.0)
    p = p / p.sum()
    y = np.zeros_like(p)
    y[winner] = 1.0
    return {"top1": float(np.argmax(p) == winner),
            "log_loss": float(-np.log(p[winner])),
            "brier": float(((p - y) ** 2).sum())}


def calibration(probs: np.ndarray, outcomes: np.ndarray, bins=(0, .05, .1, .2, .3, .5, .7, 1.0001)):
    """Group all driver-race probabilities into bins; compare mean prediction vs. win rate."""
    rows, ece, n = [], 0.0, len(probs)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (probs >= lo) & (probs < hi)
        if m.sum() == 0:
            continue
        pred, actual = float(probs[m].mean()), float(outcomes[m].mean())
        rows.append({"bin": f"{lo:.2f}-{min(hi, 1):.2f}", "n": int(m.sum()),
                     "mean_predicted": pred, "actual_rate": actual})
        ece += m.sum() / n * abs(pred - actual)  # expected calibration error
    return rows, float(ece)
