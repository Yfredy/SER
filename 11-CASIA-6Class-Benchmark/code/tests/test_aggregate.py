import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from benchmark_ser import aggregate, manifest, render


def _rows():
    out = []
    for model, base in (("ERT-WFCCL", 0.80), ("TIM-Net", 0.75), ("Light-SERNet", 0.70)):
        for seed, d in ((0, 0.00), (1, 0.02), (2, -0.02)):
            out.append({"dataset": "emodb", "protocol": "random", "seed": seed,
                        "model": model, "hash": "abc",
                        "accuracy": base + d, "macro_precision": base + d,
                        "macro_recall": base + d, "macro_f1": base + d,
                        "params": 1000, "train_s": 1.0})
    return out


def test_summary_mean_and_std():
    s = aggregate.summary(_rows())
    mean, std, n = s[("emodb", "random", "ERT-WFCCL")]["macro_f1"]
    assert n == 3
    assert abs(mean - 0.80) < 1e-9   # summary maintains 0-1; only render multiplies by 100
    assert std > 0


def test_markdown_has_all_models_and_caveat():
    md = render.build_markdown(_rows(), manifest.audit(), {"gpu": "RTX 4080 SUPER"})
    for m in ("ERT-WFCCL", "TIM-Net", "Light-SERNet"):
        assert m in md
    assert "surprise" in md  # EMO-DB missing class must appear in doc
    assert "random" in md and "speaker" in md