"""ERT-WFCCL (PyTorch) benchmark runner over the unified 24 partitions.

Pipeline per (corpus, protocol, seed):
  Whisper-Base 512-dim global features -> StandardScaler (fit on train only)
  -> MLP 512->256->128->64 (trained with val-based early stopping)
  -> ERTWFCCL(eps=0.5, n_estimators=200) -> test metrics.

Consumes ``manifest`` / ``splits`` / ``features`` from the earlier tasks and
delegates training to the official ``ert_wfccl`` implementation (read-only).
"""

import argparse
import json
import os
import sys
import time

import numpy as np

from benchmark_ser import features, labels, manifest, splits

_ERT_WFCCL_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "ERT-WFCCL", "projects", "ERT-WFCCL")
)

_WHISPER_MODEL = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "_local", "ERT-WFCCL-data", "whisper-base"
    )
)

_CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "results", "cache"))
_SPLITS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "results", "splits.json"))

MODEL_NAME = "ERT-WFCCL"


def _ert_modules():
    """Import the official ERT-WFCCL modules (inserting their root on sys.path)."""
    if _ERT_WFCCL_ROOT not in sys.path:
        sys.path.insert(0, _ERT_WFCCL_ROOT)
    from ert_wfccl import features as ert_features
    from ert_wfccl import model as ert_model

    return ert_features, ert_model


def evaluate(samples, split: dict, feat: np.ndarray, seed: int = 0) -> dict:
    """Train + evaluate ERT-WFCCL on one split.

    ``feat`` is the full (N, 512) feature matrix for *samples*, in sample
    order.  ``split["train"/"val"/"test"]`` hold indices relative to *samples*
    (and therefore to ``feat``).  Returns 0-1 floats for all metrics.
    """
    from sklearn.preprocessing import StandardScaler

    ert_features, ert_model = _ert_modules()

    y = np.array([labels.CLASSES.index(s.label) for s in samples], dtype=np.int64)
    train_idx = np.asarray(split["train"], dtype=np.int64)
    val_idx = np.asarray(split["val"], dtype=np.int64)
    test_idx = np.asarray(split["test"], dtype=np.int64)

    X = np.asarray(feat, dtype=np.float32)

    # Scaler fits on train only; val/test are only transformed.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_val = scaler.transform(X[val_idx])
    X_test = scaler.transform(X[test_idx])
    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    # train_s = total training wall time. The window covers MLP training +
    # train-side embedding extraction (produces the representation clf.fit
    # consumes) + ERT ensemble fit. It excludes feature extraction
    # (Whisper/cache), test-side embedding, prediction, and metric
    # computation. Tasks 5 and 6 (TensorFlow) time their whole model.fit(...)
    # to match this.
    t0 = time.time()
    # train_mlp returns (model, device); the model still carries the temporary
    # classification head, so its param count is the full 173,638 at 6 classes.
    model, device = ert_features.train_mlp(
        X_train,
        y_train,
        n_classes=None,
        epochs=50,
        lr=5e-4,
        batch_size=64,
        seed=seed,
        X_val=X_val,
        y_val=y_val,
        device=None,
    )
    X64_train = ert_features.extract_mlp_embedding(model, X_train, device=device)

    clf = ert_model.ERTWFCCL(eps=0.5, n_estimators=200, random_state=seed)
    clf.fit(X64_train, y_train)
    train_s = time.time() - t0

    X64_test = ert_features.extract_mlp_embedding(model, X_test, device=device)
    y_pred = clf.predict(X64_test)

    report = ert_model.classification_report(y_test, y_pred)

    return {
        "accuracy": float(report["accuracy"]),
        "macro_precision": float(report["precision"]),
        "macro_recall": float(report["recall"]),
        "macro_f1": float(report["macro_f1"]),
        "params": int(sum(p.numel() for p in model.parameters())),
        "train_s": float(train_s),
        "n_concepts": int(sum(len(v) for v in clf.subspaces_.values())),
        "n_classes_present": int(len(np.unique(y_test))),
    }


def _extract_corpus_features(samples, cache_dir: str, whisper_model: str) -> np.ndarray:
    """Whisper 512-dim features for one corpus, cached by content hash."""
    paths = [s.path for s in samples]
    ds = samples[0].dataset

    def fn(path_list):
        return features.whisper_512(path_list, whisper_model, batch_size=16)

    return features.cache_array(cache_dir, f"{ds}_whisper512", paths, fn)


def build_cli(argv=None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ERT-WFCCL on the 24 shared partitions.")
    parser.add_argument("--out", required=True, help="Output JSON path (24 rows).")
    parser.add_argument("--splits", default=_SPLITS_PATH, help="splits.json path.")
    parser.add_argument(
        "--whisper-model", default=_WHISPER_MODEL, help="Whisper-Base weights directory."
    )
    parser.add_argument("--cache-dir", default=_CACHE_DIR, help="Feature cache directory.")
    return parser


def run(out: str, splits_path: str = _SPLITS_PATH,
        whisper_model: str = _WHISPER_MODEL, cache_dir: str = _CACHE_DIR) -> list[dict]:
    """Run all 24 partitions and return the row list."""
    by_ds = manifest.by_dataset()
    all_splits = splits.load_all(splits_path)

    rows: list[dict] = []
    for ds in sorted(by_ds):
        samples = by_ds[ds]
        print(f"[ert-wfccl] extracting Whisper features for {ds} ({len(samples)} samples)",
              flush=True)
        feat = _extract_corpus_features(samples, cache_dir, whisper_model)

        for protocol in ("random", "speaker"):
            for seed in (0, 1, 2):
                key = f"{ds}_{protocol}_seed{seed}"
                sp = all_splits[key]
                assert sp["dataset"] == ds, (key, sp["dataset"])
                print(f"[ert-wfccl] running {key}", flush=True)
                rep = evaluate(samples, sp, feat, seed=seed)
                row = {
                    "dataset": ds,
                    "protocol": protocol,
                    "seed": seed,
                    "hash": sp["hash"],
                    "model": MODEL_NAME,
                    **rep,
                }
                rows.append(row)
                print(
                    f"[ert-wfccl] {key}: acc={rep['accuracy']:.4f} "
                    f"macro_f1={rep['macro_f1']:.4f}",
                    flush=True,
                )
    return rows


def main(argv=None) -> int:
    args = build_cli(argv).parse_args(argv)
    rows = run(args.out, args.splits, args.whisper_model, args.cache_dir)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"[ert-wfccl] wrote {len(rows)} rows to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
