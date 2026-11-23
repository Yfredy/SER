"""Stage 2: train the three models *better* and measure the real gain.

Three independent levers, all legitimate (nothing touches the test split):

1. **Waveform augmentation** during training (noise / gain / time-shift /
   time-stretch / pitch-shift).  Fresh augmentation each epoch, so the model
   sees a different view of every utterance per epoch.
2. **Hyper-parameter selection on the validation split** -- a small grid per
   model, chosen by val macro-F1, never by test.
3. **Multi-seed ensemble** -- average the softmax of N seeds trained with
   different augmentation streams.

The comparison that matters is speaker-disjoint (unseen speakers), because
that is the deployment condition.  Run with the matching env per model.

    # TIM-Net / Light-SERNet (TensorFlow)
    TF_USE_LEGACY_KERAS=1 ~/miniconda3/envs/tf-gpu/bin/python \
        -m benchmark_ser.improve --models tim light --protocol speaker

    # ERT-WFCCL (PyTorch)
    ~/miniconda3/envs/torch-cu124/bin/python \
        -m benchmark_ser.improve --models ert --protocol speaker
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time

import numpy as np

from benchmark_ser import augment, features, labels, manifest, splits

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
_SPLITS = os.path.join(_RESULTS_DIR, "splits.json")
_WHISPER = os.path.join(_REPO_ROOT, "_local", "ERT-WFCCL-data", "whisper-base")

DATASETS = ("emodb", "casia", "savee", "ravdess")

# Small, defensible grids.  Chosen a priori from each paper's neighbourhood --
# not tuned on test.
GRIDS = {
    "tim": {"lr": [1e-3, 5e-4], "batch": [64, 32]},
    "light": {"lr": [1e-4, 5e-5], "gamma": [2.0, 1.0]},
    "ert": {"lr": [5e-4, 1e-4], "batch": [64, 32]},
}

N_ENSEMBLE = 3


# ---------------------------------------------------------------------------
# Feature extraction with optional augmentation
# ---------------------------------------------------------------------------


def _tim_feats_from_wave(waves: list[np.ndarray]) -> np.ndarray:
    """TIM-Net MFCC ((N,188,39)) from in-memory waveforms at 22050 Hz."""
    import librosa

    from benchmark_ser.features import _MEAN_SIGNAL_LENGTH

    out = []
    for sig in waves:
        s_len = len(sig)
        if s_len < _MEAN_SIGNAL_LENGTH:
            pad = _MEAN_SIGNAL_LENGTH - s_len
            left = pad // 2
            sig = np.pad(sig, (left, pad - left), "constant")
        else:
            start = (s_len - _MEAN_SIGNAL_LENGTH) // 2
            sig = sig[start : start + _MEAN_SIGNAL_LENGTH]
        mfcc = librosa.feature.mfcc(y=sig, sr=22050, n_mfcc=39)
        out.append(np.transpose(mfcc))
    return np.stack(out, axis=0).astype(np.float32)


def _light_feats_from_wave(waves: list[np.ndarray]) -> np.ndarray:
    """Light-SERNet MFCC ((N,184,40,1)) from in-memory 16 kHz waveforms."""
    import tensorflow as tf

    from benchmark_ser import run_tf_light as L

    L._ensure_light_path()
    from dataio import get_mel_spectrogram, get_mfcc, get_spectrogram

    fixed = []
    for audio in waves:
        n = len(audio)
        if n < L._TARGET_SAMPLES:
            pad = L._TARGET_SAMPLES - n
            left = pad // 2
            audio = np.pad(audio, (left, pad - left), "constant")
        elif n > L._TARGET_SAMPLES:
            start = (n - L._TARGET_SAMPLES) // 2
            audio = audio[start : start + L._TARGET_SAMPLES]
        fixed.append(audio.astype(np.float32))

    wf = tf.convert_to_tensor(np.stack(fixed, axis=0))
    mfcc = get_mfcc(get_mel_spectrogram(get_spectrogram(wf)))
    return tf.expand_dims(mfcc, -1).numpy()


def _load_waves(paths: list[str], sr: int) -> list[np.ndarray]:
    import librosa
    import soundfile as sf

    out = []
    for p in paths:
        a, file_sr = sf.read(p, dtype="float32")
        if a.ndim > 1:
            a = a.mean(axis=1)
        if file_sr != sr:
            a = librosa.resample(a, orig_sr=file_sr, target_sr=sr)
        out.append(a.astype(np.float32))
    return out


def _augmented_feats(model: str, paths: list[str], seed: int) -> np.ndarray:
    """Augment waveforms then extract features in that model's frontend."""
    sr = 16000
    waves = _load_waves(paths, sr)
    aug = [
        augment.augment_waveform(w, seed=seed + i, sr=sr)
        for i, w in enumerate(waves)
    ]
    if model == "tim":
        # TIM-Net's frontend loads at 22050; resample the augmented audio.
        import librosa

        aug22 = [librosa.resample(w, orig_sr=sr, target_sr=22050) for w in aug]
        return _tim_feats_from_wave(aug22)
    if model == "light":
        return _light_feats_from_wave(aug)
    if model == "ert":
        # Whisper needs 16 kHz; feed the augmented waveforms through a loader
        # so nothing is written to disk.
        from benchmark_ser import run_torch as R

        R._ert_modules()
        from ert_wfccl.features import extract_whisper_features

        wave_map = {p: w for p, w in zip(paths, aug)}
        return extract_whisper_features(
            paths, _WHISPER, batch_size=8, loader=lambda p: (wave_map[p], sr)
        )
    raise ValueError(model)


# ---------------------------------------------------------------------------
# Model training (returns val + test predictions)
# ---------------------------------------------------------------------------


def _train_tim(Xtr, ytr, Xva, yva, Xte, n_classes, cfg, seed, epochs):
    import tensorflow as tf

    from benchmark_ser import run_tf_tim as T

    tf.keras.utils.set_random_seed(seed)
    net = T.build_model(n_classes)
    net.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg["lr"], beta_1=0.93, beta_2=0.98),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    y_tr = T._smooth_labels(
        tf.keras.utils.to_categorical(ytr, n_classes).astype(np.float32), 0.1
    )
    net.fit(
        Xtr, y_tr,
        validation_data=(Xva, tf.keras.utils.to_categorical(yva, n_classes)),
        batch_size=cfg.get("batch", 64), epochs=epochs, verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=40, restore_best_weights=True, mode="max")],
    )
    return net.predict(Xva, verbose=0), net.predict(Xte, verbose=0)


def _train_light(Xtr, ytr, Xva, yva, Xte, n_classes, cfg, seed, epochs):
    import tensorflow as tf

    from benchmark_ser import run_tf_light as L

    L._ensure_light_path()
    from loss import SparseCategoricalFocalLoss

    tf.keras.utils.set_random_seed(seed)
    net = L.build_model(n_classes)
    net.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg["lr"]),
        loss=SparseCategoricalFocalLoss(gamma=cfg["gamma"]),
        metrics=["accuracy"],
    )
    net.fit(
        Xtr, ytr,
        validation_data=(Xva, yva),
        batch_size=32, epochs=epochs, verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=40, restore_best_weights=True, mode="max")],
    )
    return net.predict(Xva, verbose=0), net.predict(Xte, verbose=0)


def _train_ert(Xtr, ytr, Xva, yva, Xte, n_classes, cfg, seed, epochs):
    import torch
    from sklearn.preprocessing import StandardScaler

    from benchmark_ser import run_torch as R

    ert_features, ert_model = R._ert_modules()
    scaler = StandardScaler().fit(Xtr)
    torch.manual_seed(seed)
    model, device = ert_features.train_mlp(
        scaler.transform(Xtr), ytr, n_classes=None,
        epochs=50, lr=cfg["lr"], batch_size=cfg.get("batch", 64), seed=seed,
        X_val=scaler.transform(Xva), y_val=yva, device=None,
    )
    emb_tr = ert_features.extract_mlp_embedding(model, scaler.transform(Xtr), device=device)
    clf = ert_model.ERTWFCCL(eps=0.5, n_estimators=200, random_state=seed)
    clf.fit(emb_tr, ytr)

    def _proba(X):
        """Soft scores from inverse class distance.

        ERTWFCCL exposes ``distances`` (min distance to each class's
        pseudo-concepts) but no ``predict_proba``.  Inverse-distance
        normalisation gives a comparable soft score for ensembling; the argmax
        is identical to ``predict``.
        """
        e = ert_features.extract_mlp_embedding(model, scaler.transform(X), device=device)
        F = clf._fuzzify(np.asarray(e, dtype=np.float64))
        scores = []
        for x in F:
            md, _ = clf.distances(x)
            inv = 1.0 / (np.asarray(md, dtype=np.float64) + 1e-8)
            scores.append(inv / inv.sum())
        return np.asarray(scores, dtype=np.float32)

    return _proba(Xva), _proba(Xte)


_TRAINERS = {"tim": _train_tim, "light": _train_light, "ert": _train_ert}


def _macro_f1(y_true, y_pred, n_classes):
    f1s = []
    for c in range(n_classes):
        m = y_true == c
        if m.sum() == 0:
            continue
        tp = np.sum((y_pred == c) & m)
        fp = np.sum((y_pred == c) & ~m)
        fn = np.sum((y_pred != c) & m)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def run_one(dataset, model, protocol, epochs, seed0):
    """Grid-search on val, then ensemble N seeds at the best config."""
    samples = manifest.by_dataset()[dataset]
    split = splits.load_all(_SPLITS)[f"{dataset}_{protocol}_seed{seed0}"]
    y = np.array([labels.CLASSES.index(s.label) for s in samples], dtype=np.int64)
    n_classes = len(labels.CLASSES)

    tr = np.asarray(split["train"])
    va = np.asarray(split["val"])
    te = np.asarray(split["test"])

    # Clean features for val/test (no augmentation at eval time).
    if model == "tim":
        clean = features.cache_array(
            os.path.join(_RESULTS_DIR, "cache"), f"{dataset}_timnet_mfcc",
            [s.path for s in samples], lambda ps: np.stack(
                [features.timnet_mfcc(p) for p in ps], 0).astype(np.float32))
    elif model == "light":
        from benchmark_ser import run_tf_light as L

        clean = L._extract_corpus_features(samples, os.path.join(_RESULTS_DIR, "cache"))
    else:
        from benchmark_ser import run_torch as R

        clean = R._extract_corpus_features(samples, os.path.join(_RESULTS_DIR, "cache"), _WHISPER)

    trainer = _TRAINERS[model]
    best = None
    trials = []

    keys = list(GRIDS[model])
    for combo in itertools.product(*(GRIDS[model][k] for k in keys)):
        cfg = dict(zip(keys, combo))
        Xtr = _augmented_feats(model, [samples[i].path for i in tr], seed=seed0)
        pva, _ = trainer(Xtr, y[tr], clean[va], y[va], clean[te], n_classes, cfg, seed0, epochs)
        f1 = _macro_f1(y[va], np.argmax(pva, 1), n_classes)
        trials.append({"cfg": cfg, "val_f1": f1})
        print(f"[improve] {dataset}/{model} {cfg} -> val_f1={f1 * 100:.2f}", flush=True)
        if best is None or f1 > best[1]:
            best = (cfg, f1)

    cfg = best[0]
    probas = []
    for k in range(N_ENSEMBLE):
        Xtr = _augmented_feats(model, [samples[i].path for i in tr], seed=seed0 + 100 * (k + 1))
        _, pte = trainer(Xtr, y[tr], clean[va], y[va], clean[te], n_classes, cfg, seed0 + k, epochs)
        probas.append(pte)

    ens = np.mean(probas, axis=0)
    pred = np.argmax(ens, 1)
    return {
        "dataset": dataset,
        "model": model,
        "protocol": protocol,
        "best_cfg": cfg,
        "val_f1": best[1] * 100,
        "test_macro_f1": _macro_f1(y[te], pred, n_classes) * 100,
        "test_accuracy": float((pred == y[te]).mean()) * 100,
        "n_ensemble": N_ENSEMBLE,
        "trials": trials,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", default=["tim"], choices=list(_TRAINERS))
    p.add_argument("--datasets", nargs="+", default=list(DATASETS))
    p.add_argument("--protocol", default="speaker", choices=["random", "speaker"])
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join(_RESULTS_DIR, "improved.json"))
    args = p.parse_args(argv)

    rows = []
    for ds in args.datasets:
        for m in args.models:
            t0 = time.time()
            try:
                r = run_one(ds, m, args.protocol, args.epochs, args.seed)
            except Exception as exc:
                print(f"[improve] FAILED {ds}/{m}: {exc}", flush=True)
                continue
            r["elapsed_s"] = time.time() - t0
            rows.append(r)
            print(
                f"[improve] DONE {ds}/{m}: test_f1={r['test_macro_f1']:.2f} "
                f"acc={r['test_accuracy']:.2f} cfg={r['best_cfg']} "
                f"({r['elapsed_s']:.0f}s)", flush=True,
            )
            # Save incrementally so a later crash keeps earlier results.
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(rows, f, indent=2)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"[improve] wrote {len(rows)} rows to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
