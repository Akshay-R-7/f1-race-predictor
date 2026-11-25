"""Quantise the pace model to int8 with TensorFlow Lite and measure what it costs.

    python quantize.py

int8 stores weights (and activations) as 8-bit integers instead of 32-bit floats:
about 4x smaller and often faster on CPUs and edge chips, at some accuracy cost.
A small "representative dataset" of real inputs tells the converter which value
ranges to expect, so it can choose the integer scales.
"""
import json
import os
import time

import numpy as np
import tensorflow as tf

from f1sim.data.db import connect, load_frame
from f1sim.features import FEATURES, build


def tflite_run(model_bytes: bytes, x: np.ndarray) -> tuple[dict, float]:
    """Run a TFLite model through its signature, which keeps the output names."""
    interp = tf.lite.Interpreter(model_content=model_bytes)
    runner = interp.get_signature_runner()
    in_name = list(runner.get_input_details())[0]
    outs, total = {}, 0.0
    for i in range(0, len(x), 256):                 # signature runners take fixed batches
        batch = x[i:i + 256].astype(np.float32)
        t = time.perf_counter()
        res = runner(**{in_name: batch})
        total += time.perf_counter() - t
        for k, v in res.items():
            outs.setdefault(k, []).append(v[:, 0])
    return {k: np.concatenate(v) for k, v in outs.items()}, total


if __name__ == "__main__":
    db = os.environ.get("DB", "f1_sample.sqlite")
    df = build(load_frame(connect(db)))
    test = df[df["season"] >= 2024]
    x_test = test[FEATURES].to_numpy(np.float32)
    x_rep = df[df["season"] < 2023][FEATURES].to_numpy(np.float32)
    model = tf.keras.models.load_model("results/pace_model.keras")

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    float_bytes = conv.convert()

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = lambda: ([x_rep[i:i + 1]] for i in range(0, len(x_rep), 10))
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    int8_bytes = conv.convert()

    ref = model.predict(x_test, verbose=0)
    y = test["pace_delta_s"].to_numpy()
    ok = ~np.isnan(y)
    report = {}
    for name, b in (("float32", float_bytes), ("int8", int8_bytes)):
        out, _ = tflite_run(b, x_test)
        times = [tflite_run(b, x_test)[1] for _ in range(20)]
        report[name] = {
            "size_kb": round(len(b) / 1024, 1),
            "pace_mae_vs_truth": round(float(np.abs(out["pace"][ok] - y[ok]).mean()), 4),
            "max_pace_diff_vs_keras": round(float(np.abs(out["pace"] - ref["pace"][:, 0]).max()), 4),
            "batch_latency_ms": round(1000 * float(np.median(times)), 3),
        }
    print(json.dumps(report, indent=2))
    json.dump(report, open("results/quantization.json", "w"), indent=2)
