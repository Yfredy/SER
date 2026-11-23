"""Tests for the TIM-Net (TF) benchmark runner.

The runner consumes the unified manifest + splits and trains the official
TIM-Net model.  These tests use synthetic random features so they run in
seconds without needing the full feature extraction pipeline.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, "/home/bytedance/project/yaoyue/SER/TIM-Net_SER/Code")

from benchmark_ser import labels, manifest, run_tf_tim, splits


def test_model_shape_and_evaluate():
    m = run_tf_tim.build_model(6)
    assert m.input_shape == (None, 188, 39)
    assert m.output_shape == (None, 6)

    s = manifest.by_dataset()["emodb"][:90]
    y = np.array([labels.CLASSES.index(x.label) for x in s])
    rng = np.random.default_rng(0)
    feat = rng.normal(size=(len(s), 188, 39)).astype(np.float32)
    feat[np.arange(len(s)), :, y] += 2.0

    sp = splits.build(s, "random", 0)
    rep = run_tf_tim.evaluate(s, sp, feat, seed=0)

    for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
        assert 0.0 <= rep[k] <= 1.0
    assert rep["params"] > 100000
