import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from benchmark_ser import reproduce_paper as RP


def test_raw_classes_match_corpus_dirs():
    """Every declared class must exist as a directory in the corpus root."""
    for ds, classes in RP.RAW_CLASSES.items():
        root = RP.CORPUS_ROOTS[ds]
        for cls in classes:
            assert os.path.isdir(os.path.join(root, cls)), f"{ds}/{cls} missing"


def test_scan_counts_match_raw_dirs():
    """_scan must return one entry per wav under the declared class dirs."""
    paths, y = RP._scan("emodb")
    assert len(paths) == len(y) == 535
    assert set(np.unique(y).tolist()) == set(range(7))


def test_uar_war_definitions():
    """UAR is macro recall over present classes; WAR is accuracy."""
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 0, 2, 2])
    uar, war = RP._uar_war(y_true, y_pred, 3)
    # recalls: class0 2/2=1.0, class1 1/2=0.5, class2 2/2=1.0 -> UAR 0.8333
    assert abs(uar - (1.0 + 0.5 + 1.0) / 3) < 1e-9
    assert abs(war - 5 / 6) < 1e-9


def test_uar_skips_absent_classes():
    """A class missing from y_true must not be counted (no divide-by-zero)."""
    y_true = np.array([0, 0, 2, 2])
    y_pred = np.array([0, 0, 2, 2])
    uar, war = RP._uar_war(y_true, y_pred, 3)
    assert abs(uar - 1.0) < 1e-9
    assert abs(war - 1.0) < 1e-9


def test_render_has_reference_columns():
    rows = [{
        "dataset": "emodb", "model": "tim", "split": "random",
        "n_classes": 7, "n_samples": 535,
        "folds": 10, "epochs": 500, "uar_mean": 95.0, "uar_std": 0.5,
        "war_mean": 95.5, "war_std": 0.4, "train_s_total": 1.0,
        "per_fold_uar": [95.0], "per_fold_war": [95.5],
    }]
    md = RP.render(rows)
    assert "UAR" in md and "WAR" in md
    assert "95.17" in md  # paper reference must appear
    assert "+" in md or "-" in md  # delta column populated
