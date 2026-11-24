"""Cross-corpus transfer: pretrain on RAVDESS (24 speakers), fine-tune on a small corpus.

Motivation.  Stage 2 showed that augmentation + val grid-search + ensembling
fail on CASIA/SAVEE because those corpora have only 4 speakers: the validation
speaker and the test speaker are different people, so any decision made on
validation does not transfer.  The hypothesis here is that the real bottleneck
is *speaker diversity*, not the training recipe -- so borrowing 24 speakers from
RAVDESS should help even when the target corpus has only 2 training speakers.

Design:
  * Pretrain on ALL of RAVDESS (24 speakers, 6 classes).
  * Fine-tune on the target corpus's speaker-disjoint TRAIN split only.
  * Compare against the from-scratch baseline on the same split.
  * Validation is used only for early stopping / epoch choice, never test.

Both corpora share the CASIA-style 6-class space, so the classifier head is
directly reusable.

    # ERT-WFCCL (PyTorch)
    ~/miniconda3/envs/torch-cu124/bin/python -m benchmark_ser.transfer --models ert

    # TIM-Net / Light-SERNet (TensorFlow)
    TF_USE_LEGACY_KERAS=1 ~/miniconda3/envs/tf-gpu/bin/python \
        -m benchmark_ser.transfer --models tim light
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from benchmark_ser import features, labels, manifest, splits

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
_SPLITS = os.path.join(_RESULTS_DIR, "splits.json")
_WHISPER = os.path.join(_REPO_ROOT, "_local", "ERT-WFCCL-data", "whisper-base")
_CACHE = os.path.join(_RESULTS_DIR, "cache")

SOURCE = "ravdess"
TARGETS = ("casia", "savee")
PROTOCOL = "speaker"
SEED = 0


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


def _feats(dataset: str, model: str) -> np.ndarray:
    """Cached features for one corpus in the given model's frontend."""
    samples = manifest.by_dataset()[dataset]
    paths = [s.path for s in samples]
    if model == "tim":
        from benchmark_ser import run_tf_tim as R

        return features.cache_array(
            _CACHE, f"{dataset}_timnet_mfcc", paths,
            lambda ps: np.stack([features.timnet_mfcc(p) for p in ps], 0).astype(np.float32),
        )
    if model == "light":
        from benchmark_ser import run_tf_light as R

        return R._extract_corpus_features(samples, _CACHE)
    from benchmark_ser import run_torch as R

    return R._extract_corpus_features(samples, _CACHE, _WHISPER)


def _labels(dataset: str) -> np.ndarray:
    samples = manifest.by_dataset()[dataset]
    return np.array([labels.CLASSES.index(s.label) for s in samples], dtype=np.int64)


# ---------------------------------------------------------------------------
# ERT-WFCCL
# ---------------------------------------------------------------------------


def transfer_ert(target: str, epochs: int = 50) -> dict:
    """Pretrain MLP+ERT on RAVDESS, then fine-tune on *target*."""
    import torch
    from sklearn.preprocessing import StandardScaler

    from benchmark_ser import run_torch as R

    ert_features, ert_model = R._ert_modules()
    n_classes = len(labels.CLASSES)

    # --- source: RAVDESS, all samples ---
    Xs = _feats(SOURCE, "ert")
    ys = _labels(SOURCE)
    scaler = StandardScaler().fit(Xs)
    torch.manual_seed(SEED)
    src_model, device = ert_features.train_mlp(
        scaler.transform(Xs), ys, n_classes=None,
        epochs=epochs, lr=5e-4, batch_size=64, seed=SEED,
        X_val=scaler.transform(Xs[:200]), y_val=ys[:200], device=None,
    )

    # --- target: fine-tune the same MLP on the speaker-disjoint train split ---
    Xt = _feats(target, "ert")
    yt = _labels(target)
    sp = splits.load_all(_SPLITS)[f"{target}_{PROTOCOL}_seed{SEED}"]
    tr = np.asarray(sp["train"])
    va = np.asarray(sp["val"])
    te = np.asarray(sp["test"])

    # Re-fit the scaler on the target train split: the two corpora have
    # different recording conditions, and keeping the source scaler measured
    # worse (CASIA 29.4 vs 31.0).  The pretrained *weights* are what transfer.
    scaler = StandardScaler().fit(Xt[tr])

    # Warm-start: copy the pretrained tower, then continue training on the
    # target train split.  ``train_mlp`` has no init hook, so build the module
    # here and fine-tune it with the same recipe at a lower LR.
    import torch.nn as nn

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)
    ft_model = ert_features.MLPWithClassifier(n_classes=n_classes).to(device)
    src_sd = src_model.state_dict()
    own_sd = ft_model.state_dict()
    # Load every tensor whose shape matches (head may differ in class count).
    ft_model.load_state_dict(
        {k: v for k, v in src_sd.items() if k in own_sd and own_sd[k].shape == v.shape},
        strict=False,
    )

    Xtr_t = torch.tensor(scaler.transform(Xt[tr]), dtype=torch.float32)
    ytr_t = torch.tensor(yt[tr], dtype=torch.long)
    Xva_t = torch.tensor(scaler.transform(Xt[va]), dtype=torch.float32)
    yva_t = torch.tensor(yt[va], dtype=torch.long)
    opt = torch.optim.Adam(ft_model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    rng = np.random.default_rng(SEED)
    best_val, best_state = -1.0, None
    ft_model.train()
    for _ in range(epochs):
        perm = rng.permutation(len(ytr_t))
        for lo in range(0, len(perm), 64):
            idx = perm[lo : lo + 64]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            logits = ft_model(Xtr_t[idx].to(device))
            loss = loss_fn(logits, ytr_t[idx].to(device))
            loss.backward()
            opt.step()
        ft_model.eval()
        with torch.no_grad():
            vlog = ft_model(Xva_t.to(device))
            acc = float((vlog.argmax(1).cpu() == yva_t).float().mean())
        ft_model.train()
        if acc > best_val:
            best_val = acc
            best_state = {k: v.detach().clone() for k, v in ft_model.state_dict().items()}
    if best_state is not None:
        ft_model.load_state_dict(best_state)

    emb_tr = ert_features.extract_mlp_embedding(
        ft_model, scaler.transform(Xt[tr]), device=device
    )
    clf = ert_model.ERTWFCCL(eps=0.5, n_estimators=200, random_state=SEED)
    clf.fit(emb_tr, yt[tr])

    emb_te = ert_features.extract_mlp_embedding(
        ft_model, scaler.transform(Xt[te]), device=device
    )
    pred = clf.predict(emb_te)
    return {
        "dataset": target, "model": "ert", "protocol": PROTOCOL,
        "source": SOURCE, "pretrain_epochs": epochs,
        "test_macro_f1": _macro_f1(yt[te], pred, n_classes) * 100,
        "test_accuracy": float((pred == yt[te]).mean()) * 100,
        "n_train": int(len(tr)), "n_test": int(len(te)),
    }


# ---------------------------------------------------------------------------
# TIM-Net / Light-SERNet
# ---------------------------------------------------------------------------


def _fit_tf(model: str, Xtr, ytr, Xva, yva, epochs: int, init_weights: str | None):
    """Build, optionally warm-start, and fit a TF model; return it."""
    import tensorflow as tf

    from benchmark_ser import run_tf_light as L
    from benchmark_ser import run_tf_tim as T

    n_classes = len(labels.CLASSES)
    tf.keras.utils.set_random_seed(SEED)

    if model == "tim":
        net = T.build_model(n_classes)
        net.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4, beta_1=0.93, beta_2=0.98),
            loss="categorical_crossentropy", metrics=["accuracy"],
        )
        y_tr = T._smooth_labels(
            tf.keras.utils.to_categorical(ytr, n_classes).astype(np.float32), 0.1
        )
        val = (Xva, tf.keras.utils.to_categorical(yva, n_classes))
        batch = 64
    else:
        L._ensure_light_path()
        from loss import SparseCategoricalFocalLoss

        net = L.build_model(n_classes)
        net.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss=SparseCategoricalFocalLoss(gamma=2.0), metrics=["accuracy"],
        )
        y_tr = ytr
        val = (Xva, yva)
        batch = 32

    if init_weights and os.path.exists(init_weights):
        net.load_weights(init_weights)

    net.fit(
        Xtr, y_tr, validation_data=val, batch_size=batch, epochs=epochs, verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=30, restore_best_weights=True, mode="max")],
    )
    return net


def transfer_tf(model: str, target: str, epochs: int = 200) -> dict:
    """Pretrain on RAVDESS, save weights, then fine-tune on *target*."""
    import tensorflow as tf

    n_classes = len(labels.CLASSES)
    ckpt = os.path.join(_RESULTS_DIR, f"pretrain_{model}_{SOURCE}.weights.h5")

    # --- source: pretrain on all of RAVDESS ---
    Xs, ys = _feats(SOURCE, model), _labels(SOURCE)
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(ys))
    n_val = max(1, int(0.1 * len(ys)))
    s_va, s_tr = order[:n_val], order[n_val:]
    src = _fit_tf(model, Xs[s_tr], ys[s_tr], Xs[s_va], ys[s_va], epochs, None)
    src.save_weights(ckpt)

    # --- target: warm-start from the pretrained weights ---
    Xt, yt = _feats(target, model), _labels(target)
    sp = splits.load_all(_SPLITS)[f"{target}_{PROTOCOL}_seed{SEED}"]
    tr, va, te = (np.asarray(sp[k]) for k in ("train", "val", "test"))

    ft = _fit_tf(model, Xt[tr], yt[tr], Xt[va], yt[va], epochs, ckpt)
    pred = np.argmax(ft.predict(Xt[te], verbose=0), axis=1)

    return {
        "dataset": target, "model": model, "protocol": PROTOCOL,
        "source": SOURCE, "pretrain_epochs": epochs,
        "test_macro_f1": _macro_f1(yt[te], pred, n_classes) * 100,
        "test_accuracy": float((pred == yt[te]).mean()) * 100,
        "n_train": int(len(tr)), "n_test": int(len(te)),
    }


def ablation_ert(target: str, epochs: int = 50) -> dict:
    """Controlled ablation: identical fine-tuning code, only toggle pretraining.

    The main transfer path re-fits the scaler and uses a different LR for the
    pretrained case, which confounds the comparison.  This function holds
    everything fixed so the *effect of the pretrained weights* is isolated.
    """
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import StandardScaler

    from benchmark_ser import run_torch as R

    ert_features, ert_model = R._ert_modules()
    n_classes = len(labels.CLASSES)

    Xs, ys = _feats(SOURCE, "ert"), _labels(SOURCE)
    Xt, yt = _feats(target, "ert"), _labels(target)
    sp = splits.load_all(_SPLITS)[f"{target}_{PROTOCOL}_seed{SEED}"]
    tr, va, te = (np.asarray(sp[k]) for k in ("train", "val", "test"))

    sc = StandardScaler().fit(Xt[tr])
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def _finetune(pretrained: bool) -> float:
        torch.manual_seed(SEED)
        model = ert_features.MLPWithClassifier(n_classes=n_classes).to(dev)
        if pretrained:
            src, _ = ert_features.train_mlp(
                sc.transform(Xs), ys, n_classes=None, epochs=epochs, lr=5e-4,
                batch_size=64, seed=SEED,
                X_val=sc.transform(Xs[:300]), y_val=ys[:300], device=None,
            )
            sd, own = src.state_dict(), model.state_dict()
            model.load_state_dict(
                {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape},
                strict=False,
            )
        Xtr = torch.tensor(sc.transform(Xt[tr]), dtype=torch.float32)
        ytr = torch.tensor(yt[tr], dtype=torch.long)
        Xva = torch.tensor(sc.transform(Xt[va]), dtype=torch.float32)
        yva = torch.tensor(yt[va], dtype=torch.long)
        opt = torch.optim.Adam(model.parameters(), lr=5e-4)
        loss_fn = nn.CrossEntropyLoss()
        rng = np.random.default_rng(SEED)
        best = (-1.0, None)
        for _ in range(epochs):
            model.train()
            perm = rng.permutation(len(ytr))
            for lo in range(0, len(perm), 64):
                idx = perm[lo : lo + 64]
                if len(idx) < 2:
                    continue
                opt.zero_grad()
                loss_fn(model(Xtr[idx].to(dev)), ytr[idx].to(dev)).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                acc = float((model(Xva.to(dev)).argmax(1).cpu() == yva).float().mean())
            if acc > best[0]:
                best = (acc, {k: v.clone() for k, v in model.state_dict().items()})
        if best[1]:
            model.load_state_dict(best[1])
        model.eval()

        emb = ert_features.extract_mlp_embedding(model, sc.transform(Xt[tr]), device=dev)
        clf = ert_model.ERTWFCCL(eps=0.5, n_estimators=200, random_state=SEED)
        clf.fit(emb, yt[tr])
        ete = ert_features.extract_mlp_embedding(model, sc.transform(Xt[te]), device=dev)
        return _macro_f1(yt[te], clf.predict(ete), n_classes) * 100

    scratch = _finetune(False)
    transferred = _finetune(True)
    return {
        "dataset": target, "model": "ert", "protocol": PROTOCOL,
        "ablation": True, "epochs": epochs,
        "scratch_macro_f1": scratch,
        "transfer_macro_f1": transferred,
        "delta": transferred - scratch,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", default=["ert"], choices=["ert", "tim", "light"])
    p.add_argument("--targets", nargs="+", default=list(TARGETS))
    p.add_argument("--epochs", type=int, default=0, help="0 = use per-model default")
    p.add_argument("--out", default=os.path.join(_RESULTS_DIR, "transfer.json"))
    p.add_argument(
        "--ablation", action="store_true",
        help="run the controlled scratch-vs-transfer ablation (ERT only)",
    )
    args = p.parse_args(argv)

    if args.ablation:
        out_rows = []
        for tgt in args.targets:
            r = ablation_ert(tgt, args.epochs or 50)
            out_rows.append(r)
            print(
                f"[transfer] ABLATION {tgt}: scratch={r['scratch_macro_f1']:.1f} "
                f"transfer={r['transfer_macro_f1']:.1f} delta={r['delta']:+.1f}",
                flush=True,
            )
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out_rows, f, indent=2)
        print(f"[transfer] wrote {len(out_rows)} ablation rows to {args.out}", flush=True)
        return 0

    rows = []
    for m in args.models:
        for tgt in args.targets:
            t0 = time.time()
            ep = args.epochs or (50 if m == "ert" else 200)
            try:
                r = transfer_ert(tgt, ep) if m == "ert" else transfer_tf(m, tgt, ep)
            except Exception as exc:
                print(f"[transfer] FAILED {tgt}/{m}: {exc}", flush=True)
                continue
            r["elapsed_s"] = time.time() - t0
            rows.append(r)
            print(
                f"[transfer] {tgt}/{m}: macro_f1={r['test_macro_f1']:.2f} "
                f"acc={r['test_accuracy']:.2f} ({r['elapsed_s']:.0f}s)", flush=True,
            )
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(rows, f, indent=2)

    print(f"[transfer] wrote {len(rows)} rows to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
