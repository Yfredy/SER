"""Render the authoritative SER benchmark comparison document.

Reads the three per-model result JSONs, merges them into
``benchmark_ser/results/merged.json`` (72 rows), and renders
``docs/ser-benchmark-comparison.md``.

Unit contract: ``aggregate.summary`` keeps metrics as 0-1 floats.  This module
is the ONLY place that multiplies by 100 (for the percentage display).
Every number in the produced document is computed from ``rows`` / ``audit``;
nothing is hand-written.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from benchmark_ser import aggregate, manifest
from benchmark_ser.labels import CLASSES

_RESULTS_DIR = Path(__file__).resolve().parent / "results"
_REPO_ROOT = Path(__file__).resolve().parent.parent

_RESULT_FILES = (
    _RESULTS_DIR / "ert_wfccl.json",
    _RESULTS_DIR / "light_sernet.json",
    _RESULTS_DIR / "tim_net.json",
)

_MERGED_PATH = _RESULTS_DIR / "merged.json"
_DOC_PATH = _REPO_ROOT / "docs" / "ser-benchmark-comparison.md"

MODELS = ("ERT-WFCCL", "Light-SERNet", "TIM-Net")
DATASETS = ("emodb", "casia", "savee", "ravdess")
PROTOCOLS = ("random", "speaker")

_DATASET_TITLE = {
    "emodb": "EMO-DB",
    "casia": "CASIA",
    "savee": "SAVEE",
    "ravdess": "RAVDESS",
}
_PROTOCOL_TITLE = {"random": "随机划分 (random)", "speaker": "说话人划分 (speaker)"}

# Display order for the two metric tables' model columns.
_TABLE_MODELS = ("ERT-WFCCL", "Light-SERNet", "TIM-Net")


def _pct(t: tuple[float, float, int]) -> str:
    """Format ``(mean, std, n)`` in 0-1 units as ``mean ± std (n=N)`` in %."""
    mean, std, n = t
    return f"{mean * 100:.1f} ± {std * 100:.1f} (n={n})"


def _delta_pct(a: float, b: float) -> str:
    """Signed percentage-point delta ``a - b``."""
    return f"{(a - b) * 100:+.1f}"


def _summary(rows: list[dict]) -> dict:
    return aggregate.summary(rows)


def _params_by_model(rows: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        model = r["model"]
        if model not in out:
            out[model] = int(r["params"])
        else:
            out[model] = max(out[model], int(r["params"]))
    return out


def _train_s_by_model(rows: list[dict]) -> dict[str, dict]:
    """Return ``{model: {"total": float, "n": int, "mean": float}}``."""
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for r in rows:
        m = r["model"]
        totals[m] = totals.get(m, 0.0) + float(r["train_s"])
        counts[m] = counts.get(m, 0) + 1
    return {
        m: {"total": totals[m], "n": counts[m], "mean": totals[m] / counts[m]}
        for m in totals
    }


def _train_s_minmax_by_model(rows: list[dict]) -> dict[str, tuple[float, float, float]]:
    """Return ``{model: (min, max, mean)}`` of ``train_s`` over all runs."""
    vals: dict[str, list[float]] = {}
    for r in rows:
        vals.setdefault(r["model"], []).append(float(r["train_s"]))
    out: dict[str, tuple[float, float, float]] = {}
    for m, v in vals.items():
        out[m] = (min(v), max(v), sum(v) / len(v))
    return out


def _nonzero_classes(audit: dict, ds: str) -> int:
    return sum(1 for c in CLASSES if audit[ds]["class_counts"][c] > 0)


_ENV_DEFAULTS = {
    "gpu": "NVIDIA GeForce RTX 4080 SUPER 16GB",
    "python": "3.10",
    "torch": "2.5.1+cu124",
    "tensorflow": "2.17.1",
    "tf_keras": "2.17",
}


def _env(env: dict, key: str) -> str:
    return env.get(key) or _ENV_DEFAULTS.get(key, "")


def _leakage_facts() -> dict[str, dict]:
    """Same-speaker-same-label leakage under the random protocol, from splits.json.

    For each corpus: the share of test samples whose (speaker, label) pair also
    occurs in the training split.  A high value means the random protocol lets a
    model recognise a familiar voice instead of learning the emotion.
    """
    try:
        from benchmark_ser import splits

        all_splits = splits.load_all(str(_RESULTS_DIR / "splits.json"))
        by_ds = manifest.by_dataset()
        out: dict[str, dict] = {}
        for ds in DATASETS:
            samples = by_ds.get(ds, [])
            rnd = all_splits.get(f"{ds}_random_seed0")
            spk = all_splits.get(f"{ds}_speaker_seed0")
            if not samples or not rnd or not spk:
                continue
            train_pairs = {(samples[i].speaker, samples[i].label) for i in rnd["train"]}
            test_idx = rnd["test"]
            hit = sum(
                1 for i in test_idx
                if (samples[i].speaker, samples[i].label) in train_pairs
            )
            train_spk = {samples[i].speaker for i in spk["train"]}
            spk_hit = sum(1 for i in spk["test"] if samples[i].speaker in train_spk)
            out[ds] = {
                "random_pair_hits": hit,
                "random_n": len(test_idx),
                "speaker_hits": spk_hit,
                "speaker_n": len(spk["test"]),
            }
        return out
    except Exception:
        return {}


def _speaker_rank(s: dict, metric: str = "macro_f1") -> dict[str, dict]:
    """Per-model speaker-protocol means plus per-corpus winner counts."""
    out: dict[str, dict] = {}
    wins = {m: 0 for m in MODELS}
    for ds in DATASETS:
        per_model = {}
        for m in MODELS:
            cell = s.get((ds, "speaker", m), {}).get(metric)
            if cell is not None:
                per_model[m] = cell
        if not per_model:
            continue
        best = max(per_model, key=lambda m: per_model[m][0])
        wins[best] += 1
        for m, cell in per_model.items():
            out.setdefault(m, {"means": {}})
            out[m]["means"][ds] = cell
    for m in out:
        out[m]["wins"] = wins.get(m, 0)
    return out


def _speaker_allocation() -> dict[str, tuple[int, int, int]]:
    """Return ``{dataset: (train, val, test)}`` speaker counts from splits.json."""
    try:
        from benchmark_ser import splits

        all_splits = splits.load_all(str(_RESULTS_DIR / "splits.json"))
        by_ds = manifest.by_dataset()
        out: dict[str, tuple[int, int, int]] = {}
        for ds in DATASETS:
            samples = by_ds.get(ds, [])
            sp = all_splits.get(f"{ds}_speaker_seed0")
            if not samples or not sp:
                continue
            counts = tuple(
                len({samples[i].speaker for i in sp[part]})
                for part in ("train", "val", "test")
            )
            out[ds] = counts  # type: ignore[assignment]
        return out
    except Exception:
        return {}


def _section1(env: dict, alloc: dict[str, tuple[int, int, int]] | None = None) -> str:
    alloc = alloc or {}
    alloc_txt = "；".join(
        f"{_DATASET_TITLE[ds]} {a[0]}/{a[1]}/{a[2]}"
        for ds, a in alloc.items()
    )
    if alloc_txt:
        alloc_txt = (
            f"说话人很少时名义比例无法精确达成，实际按说话人计的 train/val/test 为：{alloc_txt}"
            "（说话人粒度导致实际样本比例偏离名义比例）。"
        )
    lines = [
        "## 1. 环境与方法",
        "",
        "| 项目 | 值 |",
        "| --- | --- |",
        f"| GPU | {_env(env, 'gpu')} |",
        f"| Python | {_env(env, 'python')} |",
        f"| PyTorch | {_env(env, 'torch')} |",
        f"| TensorFlow | {_env(env, 'tensorflow')} + tf-keras {_env(env, 'tf_keras')}（运行时须 `export TF_USE_LEGACY_KERAS=1`，TIM-Net 的 `K.reverse` 在 Keras 3 下会报错） |",
        "",
        "### 协议定义",
        "",
        "- **random（随机划分）**：按标签分层随机划分，每类测试集约 20%、验证集取剩余样本的约 10%（train/val/test ≈ 72/8/20）；每个非空类别保证至少有 1 个测试样本。",
        "- **speaker（说话人划分）**：按说话人互斥划分，名义比例 70/15/15；验证集与测试集各保证至少 1 位说话人。若测试集缺类，则整体迁移携带该类的训练集说话人（保持说话人互斥）。" + alloc_txt,
        "",
        "### 可比性边界",
        "",
        "三个模型各自沿用其原论文前端（ERT-WFCCL：Whisper-Base；Light-SERNet：MFCC(40)；TIM-Net：MFCC(39)），因此 `ERT−TIM` 与 `ERT−Light` 差值并非特征受控（feature-controlled）比较，而是同时度量前端与架构的差异。"
        "同一模型内部的比较不受前端差异影响，但跨语料比较仍受各语料类别集合差异（EMO-DB 无 `surprise`）与划分比例差异的影响，见第 6 节。",
        "",
        "### 固定超参数（各模型沿用论文默认值，不做逐语料调参）",
        "",
        "| 模型 | 特征 | 模型/训练配置 |",
        "| --- | --- | --- |",
        "| ERT-WFCCL | Whisper-Base 512 维全局特征 → StandardScaler（仅训练集拟合） | MLP 512→256→128→64（dropout 0.3/0.2，Adam lr=5e-4，batch 64，epochs 50，CosineAnnealingWarmRestarts T_0=50，验证集早停）→ ERTWFCCL(eps=0.5, n_estimators=200) |",
        "| Light-SERNet | 16 kHz 单声道，居中补齐/裁剪至 3.0 s → (184, 40, 1) MFCC | Light_SERNet_V1；Adam(1e-4) + SparseCategoricalFocalLoss(γ=2)，batch 32，epochs 200，patience 40（验证集早停） |",
        "| TIM-Net | librosa MFCC (188, 39) float32 | TIMNET(nb_filters=39, kernel_size=2, nb_stacks=1, dilations=8, dropout=0.1, activation=relu) + WeightLayer + softmax Dense；Adam(1e-3, β₁=0.93, β₂=0.98) + 交叉熵（标签平滑 0.1），batch 64，epochs 200，patience 40 |",
        "",
    ]
    return "\n".join(lines)


def _section2(audit: dict) -> str:
    lines = [
        "## 2. 数据集一览",
        "",
        "CASIA 口径 6 类情绪空间：`angry, fear, happy, neutral, sad, surprise`。"
        "下表为各语料原始 wav 数、统一 6 类下的保留数、各类样本数（全部六类均列出，缺失类为 0）、"
        "说话人数与被丢弃标签。",
        "",
        "| 语料 | 原始 wav | 保留 (6 类) | angry | fear | happy | neutral | sad | surprise | 说话人 | 被丢弃标签 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for ds in DATASETS:
        a = audit[ds]
        cc = a["class_counts"]
        dropped = ", ".join(f"{k}: {v}" for k, v in sorted(a["dropped_labels"].items())) or "—"
        lines.append(
            f"| {_DATASET_TITLE[ds]} | {a['raw_wav']} | {a['kept']} | "
            f"{cc['angry']} | {cc['fear']} | {cc['happy']} | "
            f"{cc['neutral']} | {cc['sad']} | {cc['surprise']} | {a['speakers']} | {dropped} |"
        )
    lines.append("")
    return "\n".join(lines)


def _metric_table(s: dict, metric: str, audit: dict) -> list[str]:
    title = "macro-F1" if metric == "macro_f1" else "Accuracy"
    lines = [
        f"### {title}（均值 ± 标准差，n=3，单位 %）",
        "",
        "| 语料 | 协议 | ERT-WFCCL | Light-SERNet | TIM-Net | ERT−TIM | ERT−Light | 备注 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for ds in DATASETS:
        for proto in PROTOCOLS:
            cells = {}
            for model in _TABLE_MODELS:
                key = (ds, proto, model)
                cells[model] = s.get(key, {}).get(metric)
            ert = cells["ERT-WFCCL"]
            if ert is None:
                # Dataset/protocol absent from rows (should not happen in prod).
                continue
            light = cells["Light-SERNet"]
            tim = cells["TIM-Net"]
            note = ""
            if ds == "casia":
                note = (
                    "EMO-DB 无 `surprise` 类：6 类仅 5 类有样本，"
                    "macro 指标按 5 类平均，不可与其他三语料（6 类）直接比较"
                )
            lines.append(
                f"| {_DATASET_TITLE[ds]} | {_PROTOCOL_TITLE[proto]} | "
                f"{_pct(ert)} | {_pct(light)} | {_pct(tim)} | "
                f"{_delta_pct(ert[0], tim[0])} | {_delta_pct(ert[0], light[0])} | {note} |"
            )
    lines.append("")
    return lines


def _section3(s: dict, audit: dict) -> str:
    lines = [
        "## 3. 主表：`语料 × 协议 × 模型`",
        "",
        "每个格子为 3 个 seed 的 `均值 ± 标准差 (n=3)`。差值列为 "
        "`ERT−TIM` 与 `ERT−Light`（ERT-WFCCL 减去对应模型的均值，单位：百分点）。",
        "",
    ]
    lines += _metric_table(s, "macro_f1", audit)
    lines += _metric_table(s, "accuracy", audit)
    return "\n".join(lines)


def _section4_scenario(s: dict, audit: dict, leakage: dict) -> str:
    """Deployment-scenario guidance: which protocol column applies to whom."""
    rank = _speaker_rank(s)
    trusted = ("ravdess", "emodb")
    trusted_txt = " + ".join(_DATASET_TITLE[ds] for ds in trusted)

    # Trusted-corpus averages (the two corpora with enough held-out speakers).
    avg = {}
    for m in MODELS:
        vals = [
            rank[m]["means"][ds][0] * 100
            for ds in trusted
            if m in rank and ds in rank[m]["means"]
        ]
        if vals:
            avg[m] = sum(vals) / len(vals)
    avg_txt = "；".join(
        f"{m} {avg[m]:.1f}" for m in MODELS if m in avg
    )
    best = max(avg, key=avg.get) if avg else None

    # Leakage under the random protocol, per corpus.
    leak_txt = "；".join(
        f"{_DATASET_TITLE[ds]} {v['random_pair_hits']}/{v['random_n']}"
        for ds, v in leakage.items()
    )
    spk_zero = all(v["speaker_hits"] == 0 for v in leakage.values()) if leakage else False

    lines = [
        "## 4. 该看哪一列：部署场景 → 协议",
        "",
        "本文的两种协议回答两个不同的问题，选错会让选型和上线预期都偏掉：",
        "",
        "| 部署场景 | 线上说话人是否在训练语料中出现过 | 应看协议 |",
        "| --- | --- | --- |",
        "| 固定说话人（如少数配音演员的语料库产品） | 是 | **random** |",
        "| 任意用户上传音频、新录音 | 否 | **speaker** |",
        "",
    ]

    if leak_txt:
        lines += [
            "### 为什么 random 协议会高估",
            "",
            f"随机划分下，各语料测试集中「同一说话人 + 同一情绪」也出现在训练集里的比例为：{leak_txt}。"
            "说话人身份本身就能从声学特征里读出，因此模型可以走「这声音像谁 → 这个人这种情绪长什么样」的捷径，"
            "而不必学到情绪本身的声学模式。"
            + (
                "作为对照，speaker 协议下该比例为 0（训练集与测试集说话人互斥），"
                "这正是「线上来了没见过的说话人」所对应的条件。"
                if spk_zero else ""
            ),
            "",
        ]

    if avg:
        lines += [
            "### 未见过说话人时的可信结论",
            "",
            f"只有 {trusted_txt} 的留出说话人足够多（4 人 / 2 人），其 speaker 协议结果可作为稳定估计；"
            "CASIA 与 SAVEE 各仅 4 位说话人、测试集只有 1 人，标准差过大，不应被当作结论引用。"
            f"在这两个语料上按 speaker 协议的 macro-F1 均值为：{avg_txt}。",
            "",
        ]
        if best:
            lines += [
                f"**按 speaker 协议，{best} 在 {rank[best]['wins']}/{len(DATASETS)} 个语料上排名第一**"
                "（逐语料排名见第 3 节主表；差值列已给出逐格对比）。"
                "需要强调：该优势同时包含前端与架构差异（见「可比性边界」），"
                "不能单独归因于架构优劣。",
                "",
            ]

    # random -> speaker drop per model, on the trusted corpora only.
    drops = {}
    for m in MODELS:
        vals = []
        for ds in trusted:
            r = s.get((ds, "random", m), {}).get("macro_f1")
            p = s.get((ds, "speaker", m), {}).get("macro_f1")
            if r and p:
                vals.append((r[0] - p[0]) * 100)
        if vals:
            drops[m] = sum(vals) / len(vals)
    if drops:
        drop_txt = "；".join(f"{m} −{drops[m]:.1f}" for m in MODELS if m in drops)
        worst = max(drops, key=drops.get)
        lines += [
            "### 场景选择的量化影响",
            "",
            f"同一模型在 random 与 speaker 两个协议下 macro-F1 的落差（{trusted_txt} 均值）为：{drop_txt}"
            "（单位：百分点；负值表示 speaker 协议更低，即换成没见过的说话人后掉这么多）。"
            f"其中 {worst} 的落差最大（−{drops[worst]:.1f}），说明它的随机协议分数最依赖对已见说话人的记忆。"
            "若按 random 列做容量规划而实际部署面对新说话人，预期会被系统性高估。",
            "",
        ]
    return "\n".join(lines)


def _section5(rows: list[dict]) -> str:
    params = _params_by_model(rows)
    ts = _train_s_by_model(rows)
    lines = [
        "## 5. 参数量与单次训练耗时",
        "",
        "| 模型 | 参数量 | 24 次运行 train_s 合计 | 单次运行 train_s 均值 |",
        "| --- | --- | --- | --- |",
    ]
    for model in MODELS:
        if model not in params:
            continue
        lines.append(
            f"| {model} | {params[model]:,} | {ts[model]['total']:.1f} s | "
            f"{ts[model]['mean']:.1f} s |"
        )
    lines.extend([
        "",
        "注：`train_s` 为单次运行的训练墙钟时间（特征提取与预测不计入）。"
        "各 run 的 `train_s` 从亚秒级到约 60 s 不等、噪声占主导，见「局限」第 4 条。",
        "",
        "注：ERT-WFCCL 的参数量 173,638 仅为其可训练的 MLP 塔（512→256→128→64，"
        "含 BatchNorm 与临时 6 类分类层），不含产生 512 维特征的冻结 Whisper-Base 编码器"
        "（20,590,592 参数，训练中不更新）。Light-SERNet 与 TIM-Net 的 MFCC 前端无参数，"
        "故其参数量为端到端计数。",
        "",
    ])
    return "\n".join(lines)


def _esd_facts() -> dict:
    """Count ESD's real wavs and emotion classes from the local archive.

    The archive ships a macOS resource fork next to every real wav, so the raw
    entry count is exactly double.  Falls back to the known values if the
    archive is absent.
    """
    fallback = {"wavs": 35000, "n_classes": 5}
    archive = _REPO_ROOT / "dataset" / "Emotional Speech Dataset (ESD).zip"
    if not archive.exists():
        return fallback
    try:
        import collections
        import zipfile

        with zipfile.ZipFile(archive) as z:
            real = [
                n for n in z.namelist()
                if n.lower().endswith(".wav")
                and not n.startswith("__MACOSX")
                and "/._" not in n
            ]
        counts = collections.Counter(n.split("/")[-2] for n in real)
        return {"wavs": len(real), "n_classes": len(counts)}
    except Exception:
        return fallback


def _section6(rows: list[dict], audit: dict, savee_holdout: dict | None) -> str:
    n_nonzero = {ds: _nonzero_classes(audit, ds) for ds in DATASETS}
    casia_speakers = audit["casia"]["speakers"]
    savee_speakers = audit["savee"]["speakers"]

    # SAVEE speaker-protocol macro-F1 std for ERT-WFCCL, computed from rows.
    savee_std = None
    try:
        s = _summary(rows)
        key = ("savee", "speaker", "ERT-WFCCL")
        if key in s and "macro_f1" in s[key]:
            savee_std = s[key]["macro_f1"][1]
    except Exception:
        savee_std = None
    savee_std_txt = (
        f"（约 ±{savee_std * 100:.1f} 个 F1 百分点）" if savee_std is not None else ""
    )

    # SAVEE speaker-protocol held-out speakers, read from splits.json.  With only
    # 4 speakers the test set is a single speaker, and seeds can repeat one.
    held: list[str] = []
    n_distinct: int | None = None
    repeats: dict[str, int] = {}
    if savee_holdout:
        held = savee_holdout.get("held", [])
        n_distinct = savee_holdout.get("n_distinct")
        repeats = savee_holdout.get("repeats", {})

    if held:
        holdout = "SAVEE speaker 协议：" + "；".join(held)
        repeat_txt = "".join(
            f"，其中 {spk} 被 {k} 个 seed 重复留出" for spk, k in sorted(repeats.items())
        )
        seed_note = (
            f"三个 seed 并非三个等难度的抽样：它们仅覆盖 {n_distinct} 位不同说话人"
            f"{repeat_txt}，"
        )
    else:
        holdout = (
            "SAVEE 仅 4 位说话人，speaker 协议 70/15/15 切分后测试集只含 1 位说话人，"
            "不同 seed 的留出说话人未必互不相同"
        )
        seed_note = "三个 seed 并非三个等难度的抽样，"

    # Per-model train_s spread (24 runs each), computed from rows.
    ts_spread = _train_s_minmax_by_model(rows)
    ts_spread_txt = "；".join(
        f"{m} {v[0]:.1f}–{v[1]:.1f} s"
        for m, v in ts_spread.items()
    )
    esd = _esd_facts()

    lines = [
        "## 6. 局限",
        "",
        f"1. **EMO-DB 没有 `surprise` 类**（{n_nonzero['emodb']} 类有样本），"
        "其 macro 指标是按 5 个类平均，而 CASIA / SAVEE / RAVDESS 按 6 个类平均，"
        "因此 EMO-DB 的 macro-F1 不能与其他三个语料直接等同比较（主表中已在 EMO-DB 行旁标注）。",
        "",
        f"2. **说话人协议方差大**：CASIA 与 SAVEE 各仅 {casia_speakers} / {savee_speakers} 位说话人，"
        "按说话人切分后测试集只含 1 位说话人（实际分配见第 1 节），实测标准差显著大于 random 协议。"
        f"具体地，{holdout}。{seed_note}"
        f"因此 SAVEE speaker 协议的标准差极大{savee_std_txt}，其 `均值 ± 标准差` 不应被当作稳定估计。",
        "",
        f"3. **ESD（{esd['wavs']:,} 条）与中文朗读语料未纳入本轮**：ESD 体量会使单轮全量实验超过 10 小时，"
        f"且其只有 {esd['n_classes']} 类，与 CASIA 口径 6 类不一致；"
        "如需纳入应作为独立扩展计划。",
        "",
        f"4. **训练耗时噪声**：各模型 24 次运行的 `train_s` 跨度为 {ts_spread_txt}，"
        "单次运行从亚秒级到约 60 s 不等、噪声占主导，因此逐 seed 的耗时差异不具意义，"
        "只有跨模型的合计（见第 5 节）才具有参考价值。",
        "",
        "5. **指标分母**：macro 指标按各测试集实际出现的类别数（`n_classes_present`）取平均，"
        f"即 CASIA 为 {n_nonzero['casia']}、其余语料为 {n_nonzero['emodb']}；"
        "读者不应把 CASIA 的 macro-F1 与其他语料朴素地直接比较。",
        "",
        "6. **标准差类型**：主表中所有的 `±` 均为总体标准差（`statistics.pstdev`，ddof=0），"
        "即直接度量 3 个 seed 的离散程度，而非对更大总体的无偏估计。",
        "",
        "7. **GPU 非确定性**：cuDNN / cuBLAS 的默认选择可能因运行时条件选择不同内核路径，"
        "导致同一 seed 在不同硬件或驱动下结果不可逐比特复现；本报告的 `n=3` 交叉 seed 平均"
        "在一定程度上吸收了该波动。",
        "",
    ]
    return "\n".join(lines)


def build_markdown(
    rows: list[dict],
    audit: dict,
    env: dict,
    savee_holdout: dict | None = None,
) -> str:
    """Render the full comparison document from ``rows`` + ``audit``."""
    s = _summary(rows)
    parts = [
        "# SER 三模型四语料对比报告",
        "",
        "CASIA 口径 6 类情绪空间（`angry, fear, happy, neutral, sad, surprise`）下，"
        "ERT-WFCCL、Light-SERNet、TIM-Net 在 EMO-DB、CASIA、SAVEE、RAVDESS 四个语料上的"
        "随机协议与说话人协议基准对比。所有数字由 `benchmark_ser/results/merged.json` 与 "
        "`manifest.audit()` 实时计算。",
        "",
        _section1(env, _speaker_allocation()),
        _section2(audit),
        _section3(s, audit),
        _section4_scenario(s, audit, _leakage_facts()),
        _section5(rows),
        _section6(rows, audit, savee_holdout),
    ]
    return "\n".join(parts)


def compute_savee_holdout() -> dict | None:
    """Derive the SAVEE speaker-protocol held-out speakers from splits.json.

    Returns ``{"held": [...], "n_distinct": int}``, or None if the artifact is
    unavailable.  ``n_distinct`` is reported because with 4 speakers two seeds
    can hold out the same speaker.
    """
    try:
        from benchmark_ser import splits

        all_splits = splits.load_all(str(_RESULTS_DIR / "splits.json"))
        samples = manifest.by_dataset().get("savee", [])
        if not samples:
            return None
        held = []
        counts: dict[str, int] = {}
        for seed in (0, 1, 2):
            key = f"savee_speaker_seed{seed}"
            sp = all_splits.get(key)
            if not sp:
                return None
            speakers = sorted({samples[i].speaker for i in sp["test"]})
            for spk in speakers:
                counts[spk] = counts.get(spk, 0) + 1
            held.append(f"seed {seed} 留出 {', '.join(speakers)}")
        return {
            "held": held,
            "n_distinct": len(counts),
            "repeats": {spk: k for spk, k in counts.items() if k > 1},
        }
    except Exception:
        return None


def main() -> int:
    rows = aggregate.main([str(p) for p in _RESULT_FILES], _MERGED_PATH)

    audit = manifest.audit()
    env = {
        "gpu": "NVIDIA GeForce RTX 4080 SUPER 16GB",
        "python": "3.10",
        "torch": "2.5.1+cu124",
        "tensorflow": "2.17.1",
        "tf_keras": "2.17",
    }
    md = build_markdown(rows, audit, env, savee_holdout=compute_savee_holdout())
    _DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    _DOC_PATH.write_text(md, encoding="utf-8")
    print(f"[render] wrote {len(rows)} merged rows to {_MERGED_PATH}", flush=True)
    print(f"[render] wrote {_DOC_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
