"""Pace model: one Keras network with three outputs per driver and race.

  pace_delta_s        expected lap time vs. the field median (regression)
  dnf                 probability of not finishing (binary)
  tyre_deg_s_per_lap  expected tyre wear (regression)

A shared body learns what makes a driver fast at a given race; small heads turn
that into each prediction. Targets that are missing (a car retired on lap 3 has no
pace measurement) are masked out of the loss with sample weights.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from .features import FEATURES


def make_model(n_features: int, mean: np.ndarray, var: np.ndarray, seed: int = 0) -> tf.keras.Model:
    tf.keras.utils.set_random_seed(seed)
    inp = tf.keras.Input(shape=(n_features,), name="features")
    x = tf.keras.layers.Normalization(mean=mean, variance=var)(inp)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    pace = tf.keras.layers.Dense(1, name="pace")(x)
    dnf = tf.keras.layers.Dense(1, activation="sigmoid", name="dnf")(x)
    deg = tf.keras.layers.Dense(1, name="deg")(x)
    model = tf.keras.Model(inp, {"pace": pace, "dnf": dnf, "deg": deg})
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss={"pace": "huber", "dnf": "binary_crossentropy", "deg": "mse"},
                  loss_weights={"pace": 1.0, "dnf": 0.5, "deg": 20.0},
                  weighted_metrics={"pace": ["mae"], "dnf": ["AUC"], "deg": ["mae"]})
    return model


def arrays(df: pd.DataFrame):
    x = df[FEATURES].to_numpy(np.float32)
    pace = df["pace_delta_s"].to_numpy(np.float32)
    deg = df["tyre_deg_s_per_lap"].to_numpy(np.float32)
    y = {"pace": np.nan_to_num(pace), "dnf": df["dnf"].to_numpy(np.float32), "deg": np.nan_to_num(deg)}
    w = {"pace": (~np.isnan(pace)).astype(np.float32), "dnf": np.ones(len(df), np.float32),
         "deg": (~np.isnan(deg)).astype(np.float32)}
    return x, y, w


def train(train_df: pd.DataFrame, val_df: pd.DataFrame, epochs: int = 200, seed: int = 0):
    x, y, w = arrays(train_df)
    xv, yv, wv = arrays(val_df)
    model = make_model(x.shape[1], x.mean(0), x.var(0) + 1e-6, seed)
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
    hist = model.fit(x, y, sample_weight=w, validation_data=(xv, yv, wv),
                     epochs=epochs, batch_size=64, verbose=0, callbacks=[stop])
    return model, hist.history


def predict(model: tf.keras.Model, df: pd.DataFrame) -> pd.DataFrame:
    x = df[FEATURES].to_numpy(np.float32)
    pred = model.predict(x, verbose=0)
    out = df[["race_id", "driver", "team", "grid"]].copy()
    out["pred_pace"] = pred["pace"][:, 0]
    out["pred_dnf"] = np.clip(pred["dnf"][:, 0], 0.005, 0.5)
    out["pred_deg"] = np.clip(pred["deg"][:, 0], 0.0, 0.5)
    return out


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def fit_post_processing(model: tf.keras.Model, val_df: pd.DataFrame) -> dict:
    """Decide, on the validation season only, how to post-process the raw outputs.

    1. DNF calibration: shift the predicted log-odds so the average predicted DNF
       rate matches what actually happened in validation.
    2. Tyre wear source: use the model if it beats the simple 'team's recent
       average' on validation, otherwise use that average.
    """
    raw = predict(model, val_df)
    target = float(val_df["dnf"].mean())
    lo, hi = -5.0, 5.0
    for _ in range(60):                      # bisection on the log-odds shift
        mid = (lo + hi) / 2
        mean = float((1 / (1 + np.exp(-(_logit(raw["pred_dnf"].to_numpy()) + mid)))).mean())
        lo, hi = (mid, hi) if mean < target else (lo, mid)
    y = val_df["tyre_deg_s_per_lap"].to_numpy(dtype=float)
    ok = ~np.isnan(y)
    if not ok.any():  # no tyre data in this source: nothing to compare, use the team average
        return {"dnf_logit_shift": (lo + hi) / 2, "dnf_val_rate": target,
                "deg_source": "team_average", "deg_val_mae_model": None,
                "deg_val_mae_team_average": None}
    mae_model = float(np.abs(raw["pred_deg"].to_numpy()[ok] - y[ok]).mean())
    mae_team = float(np.abs(val_df["team_deg"].to_numpy()[ok] - y[ok]).mean())
    return {"dnf_logit_shift": (lo + hi) / 2, "dnf_val_rate": target,
            "deg_source": "model" if mae_model < mae_team else "team_average",
            "deg_val_mae_model": mae_model, "deg_val_mae_team_average": mae_team}


def apply_post_processing(pred: pd.DataFrame, feats: pd.DataFrame, post: dict) -> pd.DataFrame:
    out = pred.copy()
    out["pred_dnf"] = 1 / (1 + np.exp(-(_logit(out["pred_dnf"].to_numpy()) + post["dnf_logit_shift"])))
    if post["deg_source"] == "team_average":
        out["pred_deg"] = feats["team_deg"].to_numpy()
    return out
