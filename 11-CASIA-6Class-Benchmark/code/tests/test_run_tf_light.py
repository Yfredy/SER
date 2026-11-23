"""Tests for the Light-SERNet (TF) benchmark runner.

The runner consumes the unified manifest + splits and trains the official
Light_SERNet_V1 model.  These tests use synthetic random features so they run
in seconds without needing the full feature extraction pipeline.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, "/home/bytedance/project/yaoyue/SER/LIGHT-SERNET")

from benchmark_ser import labels, manifest, run_tf_light, splits


def test_model_shape_and_evaluate():
    m = run_tf_light.build_model(6)
    assert m.input_shape == (None, 184, 40, 1)
    assert m.output_shape == (None, 6)

    s = manifest.by_dataset()["emodb"][:90]
    y = np.array([labels.CLASSES.index(x.label) for x in s])
    rng = np.random.default_rng(0)
    feat = rng.normal(size=(len(s), 184, 40, 1)).astype(np.float32)
    feat[np.arange(len(s)), :, y, 0] += 2.0

    sp = splits.build(s, "random", 0)
    rep = run_tf_light.evaluate(s, sp, feat, seed=0)

    for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
        assert 0.0 <= rep[k] <= 1.0
    assert rep["params"] > 100000