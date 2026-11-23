"""Tests for the ERT-WFCCL (PyTorch) benchmark runner.

The runner consumes the unified manifest + splits + Whisper features and
trains the official ERT-WFCCL pipeline (Whisper 512-dim -> MLP 64-dim ->
ERTWFCCL classifier).  These tests use synthetic random features so they do
not need the Whisper weights and run in seconds.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from benchmark_ser import labels, manifest, run_torch, splits


def _separable_synthetic(s):
    """Class-centered synthetic features: separable through the 64-dim MLP.

    Mirrors the official repo's ``tests/test_synthetic.py`` construction
    (centers ~ N(0, 3) + noise ~ N(0, 1)) so a macro_f1 > 0.5 assertion is
    meaningful at the tiny (n ~= 85 train) sizes these corpora yield.  A
    single-dimension +3.0 bump is NOT separable through the MLP at this
    sample count and can never satisfy the brief's macro_f1 > 0.5 contract.
    """
    y = np.array([labels.CLASSES.index(x.label) for x in s])
    rng = np.random.default_rng(0)
    centers = rng.normal(0, 3, size=(6, 512))
    return np.array(
        [centers[c] + rng.normal(0, 1.0, size=512) for c in y], dtype=np.float32
    ), y


def test_evaluate_returns_all_metrics():
    s = manifest.by_dataset()["emodb"][:120]
    feat, _ = _separable_synthetic(s)
    sp = splits.build(s, "random", 0)
    rep = run_torch.evaluate(s, sp, feat, seed=0)
    for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
        assert 0.0 <= rep[k] <= 1.0
    assert rep["macro_f1"] > 0.5
    assert rep["params"] > 0


def test_evaluate_reports_n_classes_present():
    """The report must disclose how many classes were present in y_true."""
    s = manifest.by_dataset()["emodb"][:120]
    feat, _ = _separable_synthetic(s)
    sp = splits.build(s, "random", 0)
    rep = run_torch.evaluate(s, sp, feat, seed=0)
    assert 1 <= rep["n_classes_present"] <= 6
