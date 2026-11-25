import numpy as np

from f1sim.data.sample import generate
from f1sim.features import FEATURES, build, time_split
import pandas as pd


def _frame():
    races, results = generate(seasons=range(2022, 2024), seed=1)
    r = pd.DataFrame(races)
    return pd.DataFrame(results).merge(r, on="race_id")


def test_no_missing_features():
    f = build(_frame())
    assert not f[FEATURES].isna().any().any()


def test_features_do_not_use_the_current_race():
    """Change the results of one race; that race's features must not move."""
    df = _frame()
    target = "2023-05"
    before = build(df).set_index(["race_id", "driver"]).loc[target, FEATURES]
    df2 = df.copy()
    m = df2["race_id"] == target
    df2.loc[m, "pace_delta_s"] += 3.0          # wildly different result...
    df2.loc[m, "dnf"] = True
    df2.loc[m, "finish_pos"] = None
    after = build(df2).set_index(["race_id", "driver"]).loc[target, FEATURES]
    pd.testing.assert_frame_equal(before, after)  # ...same features: no leakage


def test_time_split_never_mixes_seasons():
    tr, te = time_split(build(_frame()), test_from=2023)
    assert tr["season"].max() < te["season"].min()
