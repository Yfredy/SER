"""Reproduce the papers' own protocol: original class sets, 10-fold random CV.

This is deliberately NOT the unified 6-class benchmark.  The point is to answer
"can we reproduce the published numbers?" separately from "which model
generalises to unseen speakers?", because the two questions have different
answers and mixing them produces misleading conclusions.

Protocol (matching the two papers):
  - TIM-Net   : 10-fold CV, 90/10, 500 epochs, no early stopping, original
                class count (EMODB 7, CASIA 6, SAVEE 7, RAVDESS 8), UAR/WAR.
  - Light-SERNet: 10-fold CV, 90/10, original class count, accuracy.

UAR = unweighted average recall = macro recall.
WAR = weighted average recall = accuracy.

Run with the TensorFlow env:
  TF_USE_LEGACY_KERAS=1 ~/miniconda3/envs/tf-gpu/bin/python \
      -m benchmark_ser.reproduce_paper --models tim light --folds 10
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

from benchmark_ser import features

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# Original label sets, exactly as each corpus ships them (and as the papers use).
RAW_CLASSES: dict[str, tuple[str, ...]] = {
    "emodb": ("angry", "boredom", "disgust", "fear", "happy", "neutral", "sad"),
    "casia": ("angry", "fear", "happy", "neutral", "sad", "surprise"),
    "savee": ("anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"),
    "ravdess": ("angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"),
}

CORPUS_ROOTS: dict[str, str] = {
    "emodb": os.path.join(_REPO_ROOT, "_local", "datasets", "emodb_by_class"),
    "casia": os.path.join(_REPO_ROOT, "dataset", "CASIA", "CASIA"),
    "savee": os.path.join(_REPO_ROOT, "dataset", "SAVEE", "wav"),
    "ravdess": os.path.join(_REPO_ROOT, "dataset", "RAVDESS", "wav"),
}

# Published reference numbers, for the comparison table.  Values are the
# 10-fold-CV settings from each paper's Table 1.  TIM-Net reports UAR/WAR;
# Light-SERNet reports accuracy only.
PAPER_REFERENCE: dict[str, dict[str, dict[str, float]]] = {
    "emodb": {
        "tim": {"uar": 95.17, "war": 95.70},
        "light": {"accuracy": 94.15},
    },
    "casia": {"tim": {"uar": 94.67, "war": 94.67}},
    "savee": {"tim": {"uar": 86.07, "war": 87.71}},
    "ravdess": {"tim": {"uar": 91.93, "war": 92.08}},
}


def _scan(dataset: str) -> tuple[list[str], np.ndarray]:
    """Return (paths, labels) over the corpus's *original* classes."""
    root = CORPUS_ROOTS[dataset]
    classes = RAW_CLASSES[dataset]
    paths: list[str] = []
    labels: list[int] = []
    for ci, cls in enumerate(classes):
        for p in sorted(glob.glob(os.path.join(root, cls, "*.wav"))):
            paths.append(p)
            labels.append(ci)
    return paths, np.array(labels, dtype=np.int64)


def _uar_war(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> tuple[float, float]:
    """UAR = mean per-class recall over present classes; WAR = accuracy."""
    recalls = []
    for c in range(n_classes):
        mask = y_true == c
        if mask.sum() == 0:
            continue
        recalls.append(float((y_pred[mask] == c).mean()))
    uar = float(np.mean(recalls)) if recalls else 0.0
    war = float((y_pred == y_true).mean())
    return uar, war


def _make_features(dataset: str, paths: list[str], model: str) -> np.ndarray:
    """Features for the given model, cached by content hash."""
    cache_dir = os.path.join(_RESULTS_DIR, "cache")
    if model == "tim":
        from benchmark_ser import run_tf_tim as R

        return features.cache_array(
            cache_dir, f"{dataset}_timnet_mfcc", paths, lambda ps: R.make_features(ps)
        )
    if model == "ert":
        from benchmark_ser import run_torch as R

        whisper = os.path.join(_REPO_ROOT, "_local", "ERT-WFCCL-data", "whisper-base")
        return features.cache_array(
            cache_dir, f"{dataset}_whisper512", paths,
            lambda ps: features.whisper_512(ps, whisper),
        )
    from benchmark_ser import run_tf_light as R

    return features.cache_array(
        cache_dir, f"{dataset}_lightsernet_mfcc", paths, lambda ps: R.make_features(ps)
    )


def _train_fold(
    model: str,
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    n_classes: int,
    seed: int,
    epochs: int,
) -> tuple[np.ndarray, float]:
    """Train one fold with the paper's settings; return test predictions + seconds."""
    if model == "ert":
        return _train_fold_ert(X, y, train_idx, test_idx, n_classes, seed)

    import tensorflow as tf

    from benchmark_ser import run_tf_light as L
    from benchmark_ser import run_tf_tim as T

    tf.keras.utils.set_random_seed(seed)

    val_idx = None

    if model == "tim":
        net = T.build_model(n_classes)
        net.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=1e-3, beta_1=0.93, beta_2=0.98, epsilon=1e-8
            ),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        y_train = T._smooth_labels(
            tf.keras.utils.to_categorical(y[train_idx], n_classes).astype(np.float32), 0.1
        )
        batch = 64
        callbacks: list = []
    else:
        L._ensure_light_path()
        from callbacks import BestModelWeights
        from loss import SparseCategoricalFocalLoss

        net = L.build_model(n_classes)
        net.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss=SparseCategoricalFocalLoss(gamma=2.0),
            metrics=["accuracy"],
        )
        # Integer labels: SparseCategoricalFocalLoss expects (N,) not one-hot.
        y_train = y[train_idx]
        batch = 32

        # Match the official train.py: exponential LR decay (x exp(-0.15) every
        # 20 epochs after epoch 50) and best-val-accuracy weight restore.  No
        # early stopping -- the official code runs the full EPOCHS budget.
        def _sched(epoch, lr):
            if epoch < 50:
                return float(lr)
            if epoch % 20 == 0:
                return float(lr * np.exp(-0.15))
            return float(lr)

        callbacks = [
            tf.keras.callbacks.LearningRateScheduler(_sched),
            BestModelWeights(metric="val_accuracy", metric_type="max"),
        ]
        # Official code validates on the test split (see Model.py / train.py);
        # here we validate on a held-out slice of train so the test set stays
        # untouched until the final prediction.
        n_val = max(1, int(0.1 * len(train_idx)))
        val_idx = train_idx[:n_val]
        train_idx = train_idx[n_val:]
        y_train = y[train_idx]

    fit_kwargs = {}
    if model == "light" and val_idx is not None:
        fit_kwargs["validation_data"] = (X[val_idx], y[val_idx])

    t0 = time.time()
    net.fit(
        X[train_idx],
        y_train,
        batch_size=batch,
        epochs=epochs,
        verbose=0,
        callbacks=callbacks,
        **fit_kwargs,
    )
    dt = time.time() - t0

    pred = np.argmax(net.predict(X[test_idx], verbose=0), axis=1)
    return pred, dt


def _train_fold_ert(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    n_classes: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    """Train ERT-WFCCL on one fold using its own pipeline (scaler + MLP + ERT).

    ERT needs PyTorch and sklearn, so it runs in the torch env; the TIM/Light
    paths above need TensorFlow.  A 10% slice of train is used for MLP early
    stopping so the test fold is never touched during training.
    """
    import time as _time

    import torch
    from sklearn.preprocessing import StandardScaler

    from benchmark_ser import run_torch as R

    ert_features, ert_model = R._ert_modules()

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(train_idx)
    n_val = max(1, int(0.1 * len(shuffled)))
    val_idx = shuffled[:n_val]
    tr_idx = shuffled[n_val:]

    scaler = StandardScaler().fit(X[tr_idx])
    torch.manual_seed(seed)

    t0 = _time.time()
    model, device = ert_features.train_mlp(
        scaler.transform(X[tr_idx]), y[tr_idx], n_classes=None,
        epochs=50, lr=5e-4, batch_size=64, seed=seed,
        X_val=scaler.transform(X[val_idx]), y_val=y[val_idx], device=None,
    )
    emb = ert_features.extract_mlp_embedding(model, scaler.transform(X[tr_idx]), device=device)
    clf = ert_model.ERTWFCCL(eps=0.5, n_estimators=200, random_state=seed)
    clf.fit(emb, y[tr_idx])
    dt = _time.time() - t0

    test_emb = ert_features.extract_mlp_embedding(
        model, scaler.transform(X[test_idx]), device=device
    )
    return clf.predict(test_emb), dt


def _speaker_of(dataset: str, path: str) -> str:
    """Speaker id from the filename, per corpus convention."""
    stem = os.path.splitext(os.path.basename(path))[0]
    if dataset in ("emodb", "savee"):
        return stem[:2]
    if dataset == "casia":
        return stem.split("-")[2]
    if dataset == "ravdess":
        return stem.split("-")[0]
    raise ValueError(dataset)


def _folds_random(y: np.ndarray, folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Paper-style random k-fold: shuffle, split into k contiguous blocks."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    sizes = np.full(folds, len(y) // folds, dtype=int)
    sizes[: len(y) % folds] += 1
    out = []
    start = 0
    for size in sizes:
        test_idx = order[start : start + size]
        train_idx = np.setdiff1d(order, test_idx, assume_unique=False)
        out.append((train_idx, test_idx))
        start += size
    return out


def _folds_speaker(
    dataset: str, paths: list[str]
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Leave-one-speaker-out: one fold per speaker.

    This is the protocol the official Light-SERNet code uses for EMO-DB
    (``filter_dataset.seperate_speaker_id_emodb`` returns one index group per
    speaker id and ``train.py`` loops over them), which is why its published
    EMO-DB accuracy is not comparable to a random-split number.
    """
    spk = np.array([_speaker_of(dataset, p) for p in paths])
    out = []
    for s in sorted(set(spk.tolist())):
        test_idx = np.where(spk == s)[0]
        train_idx = np.where(spk != s)[0]
        out.append((train_idx, test_idx))
    return out


def run_model(
    dataset: str,
    model: str,
    folds: int,
    epochs: int,
    seed: int,
    split: str = "random",
) -> dict:
    """Run k-fold CV for one (corpus, model) pair."""
    paths, y = _scan(dataset)
    n_classes = len(RAW_CLASSES[dataset])
    X = _make_features(dataset, paths, model)

    if split == "speaker":
        fold_list = _folds_speaker(dataset, paths)
        folds = len(fold_list)
    else:
        fold_list = _folds_random(y, folds, seed)

    print(
        f"[reproduce] {dataset} {model}: n={len(y)} classes={n_classes} "
        f"feat={X.shape} folds={folds} epochs={epochs} split={split}",
        flush=True,
    )

    uars, wars, secs = [], [], []
    for fi, (train_idx, test_idx) in enumerate(fold_list):
        pred, dt = _train_fold(model, X, y, train_idx, test_idx, n_classes, seed + fi, epochs)
        uar, war = _uar_war(y[test_idx], pred, n_classes)
        uars.append(uar)
        wars.append(war)
        secs.append(dt)
        print(
            f"[reproduce]   fold {fi + 1}/{folds}: UAR={uar * 100:.2f} WAR={war * 100:.2f} "
            f"({dt:.0f}s)",
            flush=True,
        )

    return {
        "dataset": dataset,
        "model": model,
        "n_samples": int(len(y)),
        "n_classes": n_classes,
        "folds": folds,
        "epochs": epochs,
        "split": split,
        "uar_mean": float(np.mean(uars)) * 100,
        "uar_std": float(np.std(uars)) * 100,
        "war_mean": float(np.mean(wars)) * 100,
        "war_std": float(np.std(wars)) * 100,
        "train_s_total": float(sum(secs)),
        "per_fold_uar": [float(u) * 100 for u in uars],
        "per_fold_war": [float(w) * 100 for w in wars],
    }


def render(results: list[dict]) -> str:
    """Build the reproduction markdown table from the result rows."""
    lines = [
        "# 论文协议复现（原始类别 + 10 折随机 CV）",
        "",
        "本文件回答的问题是「能否复现论文数字」，与统一 6 类基准（`ser-benchmark-comparison.md`）"
        "回答的「哪个模型泛化到未见说话人更好」是两个不同的问题。",
        "",
        "协议：k 折随机划分（论文为 10 折，90/10），训练轮数按论文设置，**不做说话人隔离**。"
        "因此本表的数字含说话人泄漏，不代表新说话人上的性能。",
        "",
        "| 语料 | 模型 | 划分 | 类别数 | 样本 | UAR (%) | WAR/Acc (%) | 论文 UAR | 论文 WAR | 差值 UAR |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        ds, model = r["dataset"], r["model"]
        ref = PAPER_REFERENCE.get(ds, {}).get(model, {})
        paper_uar = ref.get("uar")
        paper_war = ref.get("war", ref.get("accuracy"))
        du = f"{r['uar_mean'] - paper_uar:+.2f}" if paper_uar is not None else "—"
        lines.append(
            f"| {ds} | {model} | {r.get('split', 'random')} | {r['n_classes']} | {r['n_samples']} | "
            f"{r['uar_mean']:.2f} ± {r['uar_std']:.2f} | "
            f"{r['war_mean']:.2f} ± {r['war_std']:.2f} | "
            f"{paper_uar if paper_uar is not None else '—'} | "
            f"{paper_war if paper_war is not None else '—'} | {du} |"
        )
    lines += [
        "",
        "注：论文未报告 Light-SERNet 在 CASIA / SAVEE / RAVDESS 的结果，故对应列留空；"
        "TIM-Net 论文未单独报告 Light-SERNet 口径的 accuracy，其 WAR 即 accuracy。",
        "",
        "注：本表 UAR 按测试集中实际出现的类别取平均；论文按全部类别取平均。"
        "折内若某类在测试集缺席，两者会略有差异。",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["tim"], choices=["tim", "light", "ert"])
    parser.add_argument("--datasets", nargs="+", default=list(RAW_CLASSES))
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument(
        "--split",
        default="random",
        choices=["random", "speaker"],
        help="random = paper-style shuffled k-fold; speaker = leave-one-speaker-out "
             "(the official Light-SERNet EMO-DB protocol)",
    )
    parser.add_argument("--out", default=os.path.join(_RESULTS_DIR, "paper_protocol.json"))
    parser.add_argument("--doc", default=os.path.join(_REPO_ROOT, "docs", "paper-protocol-reproduction.md"))
    args = parser.parse_args(argv)

    results = []
    for ds in args.datasets:
        for model in args.models:
            results.append(
                run_model(ds, model, args.folds, args.epochs, args.seed, args.split)
            )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[reproduce] wrote {len(results)} rows to {args.out}", flush=True)

    md = render(results)
    os.makedirs(os.path.dirname(os.path.abspath(args.doc)), exist_ok=True)
    with open(args.doc, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[reproduce] wrote {args.doc}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
