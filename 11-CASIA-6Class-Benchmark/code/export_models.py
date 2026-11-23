"""Export one deployable model per (model, corpus) under the CASIA-style 6 classes.

Twelve files total: 3 models x 4 corpora, trained on the speaker-disjoint
seed-0 split (the deployment-relevant protocol: unseen speakers).  Formats are
chosen for edge deployment:

  ERT-WFCCL   -> .pt    state_dict of the trained MLP tower + scaler stats
  Light-SERNet-> .tflite int8-quantised (~0.4 MB)
  TIM-Net     -> .hdf5  Keras weights (~0.4 MB fp32)

Run with the TF env (it also needs torch for the ERT export):

  TF_USE_LEGACY_KERAS=1 ~/miniconda3/envs/tf-gpu/bin/python \
      -m benchmark_ser.export_models
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from benchmark_ser import features, labels, manifest, splits

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
_DEFAULT_OUT = os.path.join(_REPO_ROOT, "exported_models")

DATASETS = ("emodb", "casia", "savee", "ravdess")
_PROTOCOL = "speaker"
_SEED = 0


def _fit_light(samples, split, feat):
    """Train Light-SERNet on one split; return the fitted Keras model."""
    import tensorflow as tf

    from benchmark_ser import run_tf_light as L

    L._ensure_light_path()
    from loss import SparseCategoricalFocalLoss

    tf.keras.utils.set_random_seed(_SEED)
    y = np.array([labels.CLASSES.index(s.label) for s in samples], dtype=np.int64)
    X = np.asarray(feat, dtype=np.float32)

    tr = np.asarray(split["train"], dtype=np.int64)
    va = np.asarray(split["val"], dtype=np.int64)

    model = L.build_model(len(labels.CLASSES))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=L._LR),
        loss=SparseCategoricalFocalLoss(gamma=L._GAMMA),
        metrics=["accuracy"],
    )
    model.fit(
        X[tr], y[tr],
        validation_data=(X[va], y[va]),
        batch_size=L._BATCH_SIZE,
        epochs=L._EPOCHS,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=L._PATIENCE,
                restore_best_weights=True, mode="max",
            )
        ],
        verbose=0,
    )
    return model


def _fit_tim(samples, split, feat):
    """Train TIM-Net on one split; return the fitted Keras model."""
    import tensorflow as tf

    from benchmark_ser import run_tf_tim as T

    tf.keras.utils.set_random_seed(_SEED)
    y = np.array([labels.CLASSES.index(s.label) for s in samples], dtype=np.int64)
    X = np.asarray(feat, dtype=np.float32)
    tr = np.asarray(split["train"], dtype=np.int64)
    va = np.asarray(split["val"], dtype=np.int64)

    model = T.build_model(len(labels.CLASSES))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=T._LR, beta_1=T._BETA_1, beta_2=T._BETA_2, epsilon=T._EPSILON
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    y_tr = T._smooth_labels(
        tf.keras.utils.to_categorical(y[tr], len(labels.CLASSES)).astype(np.float32),
        T._LABEL_SMOOTHING,
    )
    model.fit(
        X[tr], y_tr,
        validation_data=(
            X[va], tf.keras.utils.to_categorical(y[va], len(labels.CLASSES))
        ),
        batch_size=T._BATCH_SIZE,
        epochs=T._EPOCHS,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=T._PATIENCE,
                restore_best_weights=True, mode="max",
            )
        ],
        verbose=0,
    )
    return model


def _export_tflite_int8(model, out_path: str, n_samples: int = 200) -> str:
    """Quantise *model* to int8 TFLite using a representative dataset."""
    import tensorflow as tf

    shape = model.input_shape[1:]
    rng = np.random.default_rng(0)
    rep = [rng.normal(size=(1, *shape)).astype(np.float32) for _ in range(n_samples)]

    def _rep():
        for x in rep:
            yield [x]

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = _rep
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.float32
    conv.inference_output_type = tf.float32
    blob = conv.convert()
    with open(out_path, "wb") as f:
        f.write(blob)
    return out_path


def export_ert(out_dir: str) -> list[dict]:
    """Train ERT-WFCCL per corpus and save its MLP state_dict + scaler."""
    import torch
    from sklearn.preprocessing import StandardScaler

    from benchmark_ser import run_torch as R

    ert_features, _ = R._ert_modules()
    by_ds = manifest.by_dataset()
    all_splits = splits.load_all(os.path.join(_RESULTS_DIR, "splits.json"))
    out = []

    for ds in DATASETS:
        samples = by_ds[ds]
        split = all_splits[f"{ds}_{_PROTOCOL}_seed{_SEED}"]
        feat = R._extract_corpus_features(
            samples, os.path.join(_RESULTS_DIR, "cache"),
            os.path.join(_REPO_ROOT, "_local", "ERT-WFCCL-data", "whisper-base"),
        )
        y = np.array([labels.CLASSES.index(s.label) for s in samples], dtype=np.int64)
        X = np.asarray(feat, dtype=np.float32)
        tr = np.asarray(split["train"], dtype=np.int64)
        va = np.asarray(split["val"], dtype=np.int64)

        scaler = StandardScaler().fit(X[tr])
        torch.manual_seed(_SEED)
        model, device = ert_features.train_mlp(
            scaler.transform(X[tr]), y[tr], n_classes=None,
            epochs=50, lr=5e-4, batch_size=64, seed=_SEED,
            X_val=scaler.transform(X[va]), y_val=y[va], device=None,
        )

        path = os.path.join(out_dir, f"ERT-WFCCL_{ds}.pt")
        torch.save(
            {
                "state_dict": model.state_dict(),
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(),
                "classes": list(labels.CLASSES),
                "protocol": _PROTOCOL,
                "seed": _SEED,
                "corpus": ds,
                "input_dim": 512,
                "embed_dim": 64,
                "note": "frozen Whisper-Base encoder NOT included; features are 512-d",
            },
            path,
        )
        out.append({
            "model": "ERT-WFCCL", "corpus": ds, "path": os.path.basename(path),
            "bytes": os.path.getsize(path), "format": "torch state_dict",
        })
        print(f"[export] ERT-WFCCL {ds}: {os.path.getsize(path)/1024:.0f} KB", flush=True)
    return out


def export_light(out_dir: str) -> list[dict]:
    from benchmark_ser import run_tf_light as L

    by_ds = manifest.by_dataset()
    all_splits = splits.load_all(os.path.join(_RESULTS_DIR, "splits.json"))
    out = []
    for ds in DATASETS:
        samples = by_ds[ds]
        split = all_splits[f"{ds}_{_PROTOCOL}_seed{_SEED}"]
        feat = L._extract_corpus_features(samples, os.path.join(_RESULTS_DIR, "cache"))
        model = _fit_light(samples, split, feat)

        # int8 TFLite is the deployment artefact; the intermediate .h5 is only
        # written to disk transiently because the converter needs a saved model.
        h5 = os.path.join(out_dir, f".tmp_Light-SERNet_{ds}.h5")
        model.save(h5)
        tfl = os.path.join(out_dir, f"Light-SERNet_{ds}_int8.tflite")
        try:
            _export_tflite_int8(model, tfl)
            os.remove(h5)
        except Exception as exc:  # quantisation can fail on some TF builds
            print(f"[export] WARN int8 export failed for {ds}: {exc}", flush=True)
            os.replace(h5, os.path.join(out_dir, f"Light-SERNet_{ds}.h5"))
            tfl = os.path.join(out_dir, f"Light-SERNet_{ds}.h5")

        out.append({
            "model": "Light-SERNet", "corpus": ds,
            "path": os.path.basename(tfl),
            "bytes": os.path.getsize(tfl),
            "format": "tflite int8" if tfl.endswith(".tflite") else "keras h5 fp32",
        })
        print(f"[export] Light-SERNet {ds}: {tfl} "
              f"{os.path.getsize(tfl)/1024:.0f} KB", flush=True)
    return out


def export_tim(out_dir: str) -> list[dict]:
    from benchmark_ser import run_tf_tim as T

    by_ds = manifest.by_dataset()
    all_splits = splits.load_all(os.path.join(_RESULTS_DIR, "splits.json"))
    out = []
    for ds in DATASETS:
        samples = by_ds[ds]
        split = all_splits[f"{ds}_{_PROTOCOL}_seed{_SEED}"]
        feat = T.make_features([s.path for s in samples])
        model = _fit_tim(samples, split, feat)

        path = os.path.join(out_dir, f"TIM-Net_{ds}.hdf5")
        model.save_weights(path)
        out.append({
            "model": "TIM-Net", "corpus": ds, "path": os.path.basename(path),
            "bytes": os.path.getsize(path), "format": "keras hdf5 weights",
        })
        print(f"[export] TIM-Net {ds}: {os.path.getsize(path)/1024:.0f} KB", flush=True)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=_DEFAULT_OUT)
    parser.add_argument("--models", nargs="+", default=["ert", "light", "tim"],
                        choices=["ert", "light", "tim"])
    args = parser.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    # Each model family lives in a different conda env (torch vs tf), so a
    # single process cannot import all three.  Merge with an existing manifest.
    man_path = os.path.join(args.out, "MANIFEST.json")
    rows: list[dict] = []
    if os.path.exists(man_path):
        try:
            with open(man_path, encoding="utf-8") as f:
                rows = json.load(f).get("files", [])
        except Exception:
            rows = []
    new_rows: list[dict] = []
    if "ert" in args.models:
        new_rows += export_ert(args.out)
    if "light" in args.models:
        new_rows += export_light(args.out)
    if "tim" in args.models:
        new_rows += export_tim(args.out)

    # Drop stale entries for the families being rewritten, then append fresh.
    touched = {
        "ert": "ERT-WFCCL", "light": "Light-SERNet", "tim": "TIM-Net"
    }
    rewritten = {touched[m] for m in args.models}
    rows = [r for r in rows if r["model"] not in rewritten] + new_rows
    index = {
        "classes": list(labels.CLASSES),
        "protocol": _PROTOCOL,
        "seed": _SEED,
        "note": "one model per (model, corpus), trained on speaker-disjoint seed-0 split",
        "files": rows,
        "total_bytes": sum(r["bytes"] for r in rows),
    }
    with open(os.path.join(args.out, "MANIFEST.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    print(f"[export] {len(rows)} files, {index['total_bytes']/1024/1024:.1f} MB total", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
