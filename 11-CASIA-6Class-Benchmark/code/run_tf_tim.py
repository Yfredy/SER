"""TIM-Net (official TensorFlow implementation) benchmark runner.

Pipeline per (corpus, protocol, seed):
  librosa MFCC ((188, 39) float32, mirrors the official extract_feature.py
  frontend already ported to ``benchmark_ser.features.timnet_mfcc``) ->
  official ``TIMNET`` layer + ``WeightLayer`` + softmax Dense (106,952 params)
  -> Adam(1e-3, beta_1=0.93, beta_2=0.98) + categorical crossentropy with
  0.1 label smoothing + val-based early stopping -> test metrics.

Consumes ``manifest`` / ``splits`` / ``features`` from the earlier tasks and
delegates model construction to the official ``TIM-Net_SER/Code``
implementation (read-only).  Requires ``TF_USE_LEGACY_KERAS=1`` because the
official ``TIMNET`` layer uses ``K.reverse`` on a symbolic tensor, which Keras 3
rejects.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

from benchmark_ser import features, labels, manifest, splits

_TIM_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "TIM-Net_SER", "Code")
)

_CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "results", "cache"))
_SPLITS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "results", "splits.json"))

MODEL_NAME = "TIM-Net"

# Fixed hyperparameters (matching the official implementation; no per-dataset tuning).
_N_CLASSES = 6
_INPUT_SHAPE = (188, 39)
_NB_FILTERS = 39
_KERNEL_SIZE = 2
_NB_STACKS = 1
_DILATIONS = 8
_DROPOUT = 0.1
_ACTIVATION = "relu"
_LR = 1e-3
_BETA_1 = 0.93
_BETA_2 = 0.98
_EPSILON = 1e-8
_LABEL_SMOOTHING = 0.1
_BATCH_SIZE = 64
_EPOCHS = 200
_PATIENCE = 40


def _ensure_tim_path() -> None:
    if _TIM_ROOT not in sys.path:
        sys.path.insert(0, _TIM_ROOT)


def build_model(n_classes: int):
    """Official TIM-Net model: input ``(188, 39)``, output ``(n_classes,)``.

    Replicates ``Model.TIMNET_Model.create_model`` exactly:
    ``Input((188, 39))`` -> ``TIMNET`` -> ``WeightLayer`` -> softmax ``Dense``.
    """
    _ensure_tim_path()
    import tensorflow as tf
    from tensorflow.keras.layers import Dense, Input
    from tensorflow.keras.models import Model

    from Model import WeightLayer
    from TIMNET import TIMNET

    inputs = Input(shape=_INPUT_SHAPE)
    multi_decision = TIMNET(
        nb_filters=_NB_FILTERS,
        kernel_size=_KERNEL_SIZE,
        nb_stacks=_NB_STACKS,
        dilations=_DILATIONS,
        dropout_rate=_DROPOUT,
        activation=_ACTIVATION,
        return_sequences=True,
        name="TIMNET",
    )(inputs)
    decision = WeightLayer()(multi_decision)
    predictions = Dense(n_classes, activation="softmax")(decision)
    return Model(inputs=inputs, outputs=predictions)


def _smooth_labels(labels: np.ndarray, factor: float = _LABEL_SMOOTHING) -> np.ndarray:
    """Replicate ``Model.smooth_labels`` (in-place ``*= (1-f); += f/n``)."""
    labels = labels * (1.0 - factor) + factor / labels.shape[1]
    return labels


def _classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Macro metrics averaged over classes present in y_true (matches Tasks 4/5).

    Mirrors the metric definition used by the other two runners so Task 7's
    merge can compare all three on identical semantics.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    present = np.unique(y_true)
    precs, recs, f1s = [], [], []
    for c in present:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        precs.append(prec)
        recs.append(rec)
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return {
        "accuracy": float(np.mean(y_pred == y_true)),
        "precision": float(np.mean(precs)),
        "recall": float(np.mean(recs)),
        "macro_f1": float(np.mean(f1s)),
    }


def evaluate(samples, split: dict, feat: np.ndarray, seed: int = 0) -> dict:
    """Train + evaluate TIM-Net on one split.

    ``feat`` is the full ``(N, 188, 39)`` feature matrix for *samples*, in
    sample order.  ``split["train"/"val"/"test"]`` hold indices relative to
    *samples* (and therefore to ``feat``).  Returns 0-1 floats for all metrics.
    """
    _ensure_tim_path()
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)

    y = np.array([labels.CLASSES.index(s.label) for s in samples], dtype=np.int64)
    train_idx = np.asarray(split["train"], dtype=np.int64)
    val_idx = np.asarray(split["val"], dtype=np.int64)
    test_idx = np.asarray(split["test"], dtype=np.int64)

    X = np.asarray(feat, dtype=np.float32)
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    model = build_model(_N_CLASSES)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=_LR, beta_1=_BETA_1, beta_2=_BETA_2, epsilon=_EPSILON
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=_PATIENCE,
            restore_best_weights=True,
            mode="max",
        )
    ]

    # Official pipeline smooths one-hot training labels by 0.1.
    y_train_oh = tf.keras.utils.to_categorical(y_train, num_classes=_N_CLASSES).astype(
        np.float32
    )
    y_train_smooth = _smooth_labels(y_train_oh, _LABEL_SMOOTHING)

    # train_s = total training wall time = the whole model.fit(...) call.
    # Excludes feature extraction and evaluation/prediction (matches Tasks 4/5).
    t0 = time.time()
    model.fit(
        X_train,
        y_train_smooth,
        validation_data=(X_val, tf.keras.utils.to_categorical(y_val, num_classes=_N_CLASSES)),
        batch_size=_BATCH_SIZE,
        epochs=_EPOCHS,
        callbacks=callbacks,
        verbose=0,
    )
    train_s = time.time() - t0

    # Test evaluated once, after training, using the best-val weights.
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    report = _classification_report(y_test, y_pred)

    return {
        "accuracy": float(report["accuracy"]),
        "macro_precision": float(report["precision"]),
        "macro_recall": float(report["recall"]),
        "macro_f1": float(report["macro_f1"]),
        "params": int(model.count_params()),
        "train_s": float(train_s),
        "n_classes_present": int(len(np.unique(y_test))),
    }


def make_features(paths: list[str]) -> np.ndarray:
    """TIM-Net MFCC features ``(N, 188, 39)`` for a list of audio paths."""
    return np.stack([features.timnet_mfcc(p) for p in paths], axis=0).astype(np.float32)


def _extract_corpus_features(samples, cache_dir: str) -> np.ndarray:
    """TIM-Net MFCC features for one corpus, cached by content hash."""
    paths = [s.path for s in samples]
    ds = samples[0].dataset
    return features.cache_array(cache_dir, f"{ds}_timnet_mfcc", paths, make_features)


def build_cli(argv=None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TIM-Net on the 24 shared partitions.")
    parser.add_argument("--out", required=True, help="Output JSON path (24 rows).")
    parser.add_argument("--splits", default=_SPLITS_PATH, help="splits.json path.")
    parser.add_argument("--cache-dir", default=_CACHE_DIR, help="Feature cache directory.")
    return parser


def run(out: str, splits_path: str = _SPLITS_PATH, cache_dir: str = _CACHE_DIR) -> list[dict]:
    """Run all 24 partitions and return the row list."""
    by_ds = manifest.by_dataset()
    all_splits = splits.load_all(splits_path)

    rows: list[dict] = []
    for ds in sorted(by_ds):
        samples = by_ds[ds]
        print(f"[tim-net] extracting MFCC features for {ds} ({len(samples)} samples)",
              flush=True)
        feat = _extract_corpus_features(samples, cache_dir)

        for protocol in ("random", "speaker"):
            for seed in (0, 1, 2):
                key = f"{ds}_{protocol}_seed{seed}"
                sp = all_splits[key]
                assert sp["dataset"] == ds, (key, sp["dataset"])
                print(f"[tim-net] running {key}", flush=True)
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
                    f"[tim-net] {key}: acc={rep['accuracy']:.4f} "
                    f"macro_f1={rep['macro_f1']:.4f}",
                    flush=True,
                )
    return rows


def main(argv=None) -> int:
    args = build_cli(argv).parse_args(argv)
    rows = run(args.out, args.splits, args.cache_dir)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"[tim-net] wrote {len(rows)} rows to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
