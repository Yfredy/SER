"""Light-SERNet (official TensorFlow implementation) benchmark runner.

Pipeline per (corpus, protocol, seed):
  16 kHz mono audio (center pad/crop to 3.0 s) -> official dataio frontend
  (stft -> log-mel -> mfcc, (184, 40, 1)) -> Light_SERNet_V1 (463,110 params)
  -> Adam(1e-4) + SparseCategoricalFocalLoss(gamma=2) with val-based early
  stopping -> test metrics.

Consumes ``manifest`` / ``splits`` / ``features`` from the earlier tasks and
delegates model construction / feature frontend to the official
``LIGHT-SERNET`` implementation (read-only).
"""

import argparse
import json
import os
import sys
import time

import numpy as np

from benchmark_ser import features, labels, manifest, splits

_LIGHT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "LIGHT-SERNET")
)

_CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "results", "cache"))
_SPLITS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "results", "splits.json"))

MODEL_NAME = "Light-SERNet"

# Fixed hyperparameters (matching the official implementation; no per-dataset tuning).
_N_CLASSES = 6
_INPUT_DURATION = 3.0
_INPUT_TYPE = "mfcc"
_TARGET_SAMPLES = 48000  # 3.0 s at 16 kHz
_SAMPLE_RATE = 16000
_LR = 1e-4
_GAMMA = 2
_BATCH_SIZE = 32
_EPOCHS = 200
_PATIENCE = 40


def _ensure_light_path() -> None:
    if _LIGHT_ROOT not in sys.path:
        sys.path.insert(0, _LIGHT_ROOT)


def build_model(n_classes: int):
    """Return the official ``Light_SERNet_V1`` model: input (184, 40, 1)."""
    _ensure_light_path()
    import models

    return models.Light_SERNet_V1(
        output_class=n_classes,
        input_duration=_INPUT_DURATION,
        input_type=_INPUT_TYPE,
    )


def make_features(paths: list[str]) -> np.ndarray:
    """Official Light-SERNet MFCC frontend: ``(N, 184, 40, 1)`` float32.

    Reads each file as mono, resamples to 16 kHz, center pads/crops to 48000
    samples, then applies ``dataio.get_spectrogram -> get_mel_spectrogram ->
    get_mfcc`` exactly as the official training pipeline does, finally adding
    the trailing channel axis.
    """
    _ensure_light_path()
    import librosa
    import soundfile as sf
    import tensorflow as tf

    from dataio import get_mel_spectrogram, get_mfcc, get_spectrogram

    waves = []
    for p in paths:
        audio, sr = sf.read(p, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != _SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=_SAMPLE_RATE)
        n = len(audio)
        if n < _TARGET_SAMPLES:
            pad = _TARGET_SAMPLES - n
            left = pad // 2
            audio = np.pad(audio, (left, pad - left), "constant")
        elif n > _TARGET_SAMPLES:
            start = (n - _TARGET_SAMPLES) // 2
            audio = audio[start : start + _TARGET_SAMPLES]
        waves.append(audio.astype(np.float32))

    waveform = tf.convert_to_tensor(np.stack(waves, axis=0))  # (N, 48000)
    spectrogram = get_spectrogram(waveform)
    log_mel = get_mel_spectrogram(spectrogram)
    mfcc = get_mfcc(log_mel)
    mfcc = tf.expand_dims(mfcc, -1)
    return mfcc.numpy()


def _classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Macro metrics averaged over classes present in y_true (matches Task 4).

    Mirrors ``ert_wfccl.model.classification_report`` semantics so Task 7's
    merge can compare the three runners on identical metric definitions.
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
    """Train + evaluate Light-SERNet on one split.

    ``feat`` is the full (N, 184, 40, 1) feature matrix for *samples*, in
    sample order.  ``split["train"/"val"/"test"]`` hold indices relative to
    *samples* (and therefore to ``feat``).  Returns 0-1 floats for all metrics.
    """
    _ensure_light_path()
    import tensorflow as tf

    from loss import SparseCategoricalFocalLoss

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
        optimizer=tf.keras.optimizers.Adam(_LR),
        loss=SparseCategoricalFocalLoss(gamma=_GAMMA),
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

    # train_s = total training wall time = the whole model.fit(...) call.
    # Excludes feature extraction and evaluation/prediction (matches Task 4).
    t0 = time.time()
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
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


def _extract_corpus_features(samples, cache_dir: str) -> np.ndarray:
    """Light-SERNet MFCC features for one corpus, cached by content hash."""
    paths = [s.path for s in samples]
    ds = samples[0].dataset
    return features.cache_array(cache_dir, f"{ds}_lightsernet_mfcc", paths, make_features)


def build_cli(argv=None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Light-SERNet on the 24 shared partitions.")
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
        print(f"[light-sernet] extracting MFCC features for {ds} ({len(samples)} samples)",
              flush=True)
        feat = _extract_corpus_features(samples, cache_dir)

        for protocol in ("random", "speaker"):
            for seed in (0, 1, 2):
                key = f"{ds}_{protocol}_seed{seed}"
                sp = all_splits[key]
                assert sp["dataset"] == ds, (key, sp["dataset"])
                print(f"[light-sernet] running {key}", flush=True)
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
                    f"[light-sernet] {key}: acc={rep['accuracy']:.4f} "
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
    print(f"[light-sernet] wrote {len(rows)} rows to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
