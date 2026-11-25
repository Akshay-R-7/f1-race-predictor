import numpy as np

from f1sim.metrics import calibration, race_scores


def test_perfect_prediction():
    s = race_scores(np.array([1.0, 0.0, 0.0]), 0)
    assert s["top1"] == 1.0 and s["log_loss"] < 0.01 and s["brier"] < 0.01


def test_confident_mistake_is_punished_more_than_uncertainty():
    confident_wrong = race_scores(np.array([0.95, 0.05]), 1)
    unsure = race_scores(np.array([0.5, 0.5]), 1)
    assert confident_wrong["log_loss"] > unsure["log_loss"]
    assert confident_wrong["brier"] > unsure["brier"]


def test_calibration_of_a_calibrated_forecaster_is_near_zero():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 200_000)
    y = (rng.uniform(0, 1, p.size) < p).astype(float)  # events happen at exactly rate p
    _, ece = calibration(p, y)
    assert ece < 0.01
