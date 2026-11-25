import pytest

from f1sim.simulate import _racesim, simulate_python

FIELD = dict(pace=[i * 0.08 for i in range(10)], pace_sd=[0.15] * 10, dnf_prob=[0.05] * 10,
             deg=[0.06] * 10, grid=list(range(1, 11)), laps=50, pit_loss=22.0,
             sc_prob_per_lap=0.004, overtake_difficulty=0.4)


def test_python_version_sensible():
    s = simulate_python(**FIELD, n_sims=2000, seed=1)
    assert 0.98 < sum(s["win_prob"]) <= 1.0
    assert s["win_prob"][0] > s["win_prob"][-1]  # pole + fastest beats slowest


@pytest.mark.skipif(_racesim is None, reason="C++ module not built")
def test_cpp_and_python_agree():
    """Same algorithm, different random numbers: win chances must match within noise."""
    py = simulate_python(**FIELD, n_sims=6000, seed=3)
    cpp = _racesim.simulate(**FIELD, n_sims=60000, n_threads=2, seed=3)
    for a, b in zip(py["win_prob"], cpp["win_prob"]):
        assert abs(a - b) < 0.03
    for a, b in zip(py["dnf_rate"], cpp["dnf_rate"]):
        assert abs(a - b) < 0.015
