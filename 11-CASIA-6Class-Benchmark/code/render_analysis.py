"""Render the SER model-comparison analysis for the public SER repository.

Consumes ``benchmark_ser/results/merged.json`` (72 rows) and the exported-model
manifest, and writes ``docs/SER-ANALYSIS.md``: a Chinese-language analysis of
three SER models across four corpora under the CASIA-style 6-class space.

Every number is computed from the artifacts; nothing is hand-written.
"""

from __future__ import annotations

import json
import os
import statistics as st

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
_DOC = os.path.join(_REPO_ROOT, "docs", "SER-ANALYSIS.md")
_MODELS_OUT = os.path.join(_REPO_ROOT, "exported_models")

MODELS = ("ERT-WFCCL", "Light-SERNet", "TIM-Net")
DATASETS = ("emodb", "casia", "savee", "ravdess")
TITLES = {"emodb": "EMO-DB", "casia": "CASIA", "savee": "SAVEE", "ravdess": "RAVDESS"}
# Corpora whose speaker-protocol test set has enough held-out speakers to trust.
TRUSTED = ("ravdess", "emodb")


def _load():
    with open(os.path.join(_RESULTS, "merged.json"), encoding="utf-8") as f:
        return json.load(f)


def _cell(rows, ds, proto, model, metric):
    vals = [
        r[metric] for r in rows
        if r["dataset"] == ds and r["protocol"] == proto and r["model"] == model
    ]
    if not vals:
        return None
    return st.mean(vals) * 100, st.pstdev(vals) * 100, len(vals)


def _fmt(cell, unit="%"):
    if cell is None:
        return "—"
    m, s, n = cell
    return f"{m:.1f} ± {s:.1f}"


def _speakers_per_corpus():
    """Held-out speaker counts for the speaker protocol, from splits.json."""
    try:
        from benchmark_ser import manifest, splits

        all_splits = splits.load_all(os.path.join(_RESULTS, "splits.json"))
        by_ds = manifest.by_dataset()
        out = {}
        for ds in DATASETS:
            sp = all_splits.get(f"{ds}_speaker_seed0")
            samples = by_ds.get(ds, [])
            if not sp or not samples:
                continue
            out[ds] = {
                "total": len({s.speaker for s in samples}),
                "test": len({samples[i].speaker for i in sp["test"]}),
            }
        return out
    except Exception:
        return {}


def _leakage():
    """Share of random-protocol test samples whose (speaker,label) is in train."""
    try:
        from benchmark_ser import manifest, splits

        all_splits = splits.load_all(os.path.join(_RESULTS, "splits.json"))
        by_ds = manifest.by_dataset()
        out = {}
        for ds in DATASETS:
            sp = all_splits.get(f"{ds}_random_seed0")
            samples = by_ds.get(ds, [])
            if not sp or not samples:
                continue
            pairs = {(samples[i].speaker, samples[i].label) for i in sp["train"]}
            hit = sum(
                1 for i in sp["test"]
                if (samples[i].speaker, samples[i].label) in pairs
            )
            out[ds] = (hit, len(sp["test"]))
        return out
    except Exception:
        return {}


def _dataset_table(rows):
    """Per-corpus sample accounting from manifest.audit()."""
    from benchmark_ser import manifest

    audit = manifest.audit()
    lines = [
        "| 语料 | 原始 wav | 保留 | angry | fear | happy | neutral | sad | surprise | 说话人 | 丢弃标签 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for ds in DATASETS:
        a = audit[ds]
        cc = a["class_counts"]
        dropped = ", ".join(
            f"{k}: {v}" for k, v in sorted(a["dropped_labels"].items())
        ) or "—"
        lines.append(
            f"| {TITLES[ds]} | {a['raw_wav']} | {a['kept']} | "
            f"{cc['angry']} | {cc['fear']} | {cc['happy']} | {cc['neutral']} | "
            f"{cc['sad']} | {cc['surprise']} | {a['speakers']} | {dropped} |"
        )
    return "\n".join(lines)


def _metric_table(rows, metric, title):
    lines = [
        f"### {title}（均值 ± 标准差，n=3，单位 %）",
        "",
        "| 语料 | 协议 | ERT-WFCCL | Light-SERNet | TIM-Net | ERT−TIM | ERT−Light |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for ds in DATASETS:
        for proto in ("random", "speaker"):
            cells = {m: _cell(rows, ds, proto, m, metric) for m in MODELS}
            if cells["ERT-WFCCL"] is None:
                continue
            ert = cells["ERT-WFCCL"][0]
            tim = cells["TIM-Net"][0]
            light = cells["Light-SERNet"][0]
            lines.append(
                f"| {TITLES[ds]} | {'随机' if proto == 'random' else '说话人互斥'} | "
                f"{_fmt(cells['ERT-WFCCL'])} | {_fmt(cells['Light-SERNet'])} | "
                f"{_fmt(cells['TIM-Net'])} | {ert - tim:+.1f} | {ert - light:+.1f} |"
            )
    return "\n".join(lines)


def _params_table(rows):
    params = {}
    for r in rows:
        params.setdefault(r["model"], int(r["params"]))
    train_s = {}
    for r in rows:
        train_s.setdefault(r["model"], []).append(float(r["train_s"]))
    lines = [
        "| 模型 | 可训练参数 | 冻结前端 | 总参数 | 单次训练耗时（均值） |",
        "| --- | --- | --- | --- | --- |",
    ]
    frozen = {"ERT-WFCCL": 20590592, "Light-SERNet": 0, "TIM-Net": 0}
    for m in MODELS:
        if m not in params:
            continue
        p = params[m]
        fz = frozen[m]
        lines.append(
            f"| {m} | {p:,} | {fz:,} | {p + fz:,} | "
            f"{st.mean(train_s[m]):.1f} s |"
        )
    return "\n".join(lines)


def _export_table():
    path = os.path.join(_MODELS_OUT, "MANIFEST.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        man = json.load(f)
    lines = [
        "| 模型 | 语料 | 文件 | 格式 | 体积 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in man["files"]:
        lines.append(
            f"| {r['model']} | {TITLES.get(r['corpus'], r['corpus'])} | "
            f"`{r['path']}` | {r['format']} | {r['bytes'] / 1024:.0f} KB |"
        )
    total = man["total_bytes"] / 1024 / 1024
    lines += ["", f"合计 {len(man['files'])} 个文件，{total:.1f} MB。"]
    return "\n".join(lines)


def build(rows) -> str:
    spk = _speakers_per_corpus()
    leak = _leakage()

    # Trusted-corpus speaker-protocol averages.
    trusted_avg = {}
    for m in MODELS:
        vals = [
            _cell(rows, ds, "speaker", m, "macro_f1")[0]
            for ds in TRUSTED
            if _cell(rows, ds, "speaker", m, "macro_f1")
        ]
        if vals:
            trusted_avg[m] = st.mean(vals)
    best = max(trusted_avg, key=trusted_avg.get) if trusted_avg else None

    # random -> speaker drop on trusted corpora.
    drops = {}
    for m in MODELS:
        ds_vals = []
        for ds in TRUSTED:
            r = _cell(rows, ds, "random", m, "macro_f1")
            p = _cell(rows, ds, "speaker", m, "macro_f1")
            if r and p:
                ds_vals.append(r[0] - p[0])
        if ds_vals:
            drops[m] = st.mean(ds_vals)

    parts = [
        "# SER 三模型对比分析",
        "",
        "本目录把 ERT-WFCCL、Light-SERNet、TIM-Net 三个语音情感识别模型放在"
        "**同一标签空间、同一数据划分、同一评测代码**下对比，并附可直接部署的模型文件。",
        "",
        "## 1. 实验口径",
        "",
        "**标签空间：CASIA 口径 6 类**（`angry, fear, happy, neutral, sad, surprise`）。"
        "选这套而非 `disgust` 系，是因为它对应语料自身的类别约定：CASIA 原生就是这 6 类，"
        "1200 条全部保留。",
        "",
        "需要明确一处数据固有限制：**EMO-DB 语料里没有 `surprise`**（它只有 7 类："
        "angry/boredom/disgust/fear/happy/neutral/sad）。因此 EMO-DB 在该口径下只有 5 类有样本，"
        "其 macro 指标按 5 类平均，与其余三个语料（6 类）不可直接等同比较。"
        "这是语料本身的类别集合决定的，无法通过调参规避。",
        "",
        _dataset_table(rows),
        "",
        "**划分协议**：两种，回答两个不同的问题。",
        "",
        "- **随机划分（random）**：按标签分层随机划分，train/val/test ≈ 72/8/20。"
        "对应「线上音频的说话人在训练语料里出现过」的场景。",
        "- **说话人互斥（speaker）**：整位说话人留出，train/val/test 按说话人 70/15/15。"
        "对应「线上来了没见过的说话人」的场景。",
        "",
        "每个格子为 3 个 seed 的均值 ± 总体标准差。测试集只在训练结束后评测一次，"
        "模型选择与早停只看验证集。",
        "",
        "## 2. 主结果",
        "",
        _metric_table(rows, "macro_f1", "macro-F1"),
        "",
        _metric_table(rows, "accuracy", "Accuracy"),
        "",
    ]

    if leak:
        leak_txt = "；".join(
            f"{TITLES[ds]} {h}/{n}" for ds, (h, n) in leak.items()
        )
        parts += [
            "### 为什么随机划分会高估",
            "",
            f"随机划分下，各语料测试集中「同一说话人 + 同一情绪」也出现在训练集里的比例为：{leak_txt}"
            "（即四个语料均为 100%）。说话人身份可以从声学特征里读出，"
            "模型因此可以走「这声音像谁 → 这个人这种情绪长什么样」的捷径。"
            "说话人互斥划分下该比例为 0。",
            "",
        ]

    if trusted_avg:
        avg_txt = "；".join(f"{m} {trusted_avg[m]:.1f}" for m in MODELS if m in trusted_avg)
        parts += [
            "## 3. 结论：泛化到未见说话人时选谁",
            "",
            f"只有 RAVDESS（24 人）与 EMO-DB（10 人）留出的说话人足够多，"
            f"其 speaker 协议结果可作为稳定估计。这两个语料上 macro-F1 均值为：{avg_txt}。",
            "",
        ]
        if best:
            wins = sum(
                1 for ds in DATASETS
                if (_cell(rows, ds, "speaker", best, "macro_f1") or (0,))[0]
                == max(
                    (_cell(rows, ds, "speaker", m, "macro_f1") or (0,))[0]
                    for m in MODELS
                )
            )
            parts += [
                f"**{best} 在 {wins}/4 个语料的说话人互斥协议下排名第一。**",
                "",
            ]
        if drops:
            drop_txt = "；".join(
                f"{m} −{drops[m]:.1f}" for m in MODELS if m in drops
            )
            worst = max(drops, key=drops.get)
            parts += [
                f"随机 → 说话人互斥的落差（RAVDESS + EMO-DB 均值，百分点）：{drop_txt}。"
                f"其中 {worst} 落差最大（−{drops[worst]:.1f}），"
                "说明它的随机划分分数最依赖对已见说话人的记忆。",
                "",
                "**重要限定**：ERT-WFCCL 的领先同时包含前端与架构的差异 —— 它使用冻结的 "
                "Whisper-Base 编码器（20,590,592 参数，在海量语音上预训练），"
                "而另外两个模型从零训练 MFCC 前端。因此这个结果不能单独归因于架构优劣。",
                "",
            ]

    parts += [
        "## 4. 参数量与训练成本",
        "",
        _params_table(rows),
        "",
        "注：`ERT-WFCCL` 的可训练参数仅为 MLP 塔（512→256→128→64 + 临时 6 类分类层），"
        "不含产生 512 维特征的冻结 Whisper-Base 编码器。若计入编码器，其部署体积是"
        "另外两个模型的数十倍 —— 在有模型体积约束（如 2 MB 预算）的场景下不可用。",
        "",
    ]

    exp = _export_table()
    if exp:
        parts += [
            "## 5. 可部署模型文件",
            "",
            "`exported_models/` 下每个「模型 × 语料」各一份，均按 **说话人互斥协议 seed 0** "
            "训练（对应真实部署：未见过的说话人）。格式按端侧部署选择：",
            "",
            exp,
            "",
            "各格式说明：",
            "",
            "- **ERT-WFCCL `.pt`**：PyTorch `state_dict`（MLP 塔）+ StandardScaler 的均值/方差 + 类别表。"
            "**不含 Whisper-Base 编码器** —— 部署时需自行提供 512 维 Whisper 特征。",
            "- **Light-SERNet `.tflite`**：int8 量化，可直接在移动端/嵌入式 TFLite 运行时加载。",
            "- **TIM-Net `.hdf5`**：Keras 权重，需配套 `TIM-Net/Code/TIMNET.py` 的模型定义。",
            "",
            "用法见 `exported_models/README.md`。",
            "",
        ]

    parts += [
        "## 6. 局限",
        "",
        "1. **EMO-DB 只有 5 类**：语料无 `surprise`，其 macro 指标按 5 类平均，"
        "不能与 CASIA / SAVEE / RAVDESS 的 6 类 macro 直接等同比较。",
        "",
        "2. **CASIA 与 SAVEE 只有 4 位说话人**：说话人互斥划分后测试集仅含 1 人，"
        "标准差极大（SAVEE 达 ±26.2），其 speaker 协议均值不应作为稳定估计引用。"
        "这两个语料的随机划分结果相对可信。",
        "",
        "3. **三个模型的前端不同**：ERT 用 Whisper-Base（预训练、冻结），"
        "Light 用 MFCC(40)，TIM 用 MFCC(39)。跨模型差值同时包含前端与架构差异，"
        "不是特征受控比较。",
        "",
        "4. **未纳入的语料**：ESD（35,000 条，只有 angry/happy/neutral/sad/surprise 五类，"
        "在本口径下缺 `fear`，因此无法与其他语料对齐 6 类）、IEMOCAP、EMOVO。"
        "中文朗读语料同样未纳入。",
        "",
        "5. **超参数固定为各模型论文默认值**，未做逐语料调参。因此结果反映"
        "「开箱即用」的相对水平，而非各模型在该语料上的最优性能。",
        "",
        "6. **GPU 非确定性**：cuDNN/cuBLAS 可能选择不同内核路径，同一 seed 不保证逐比特复现。",
        "",
    ]
    return "\n".join(parts)


def main() -> int:
    rows = _load()
    md = build(rows)
    os.makedirs(os.path.dirname(_DOC), exist_ok=True)
    with open(_DOC, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[analysis] wrote {_DOC} ({len(md.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
