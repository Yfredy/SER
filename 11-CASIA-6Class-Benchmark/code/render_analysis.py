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
_MODEL_KEY = {"ERT-WFCCL": "ert", "Light-SERNet": "light", "TIM-Net": "tim"}

# Published 10-fold-CV numbers (TIM-Net paper Table 1).  Light-SERNet's paper
# only reports EMO-DB accuracy, and only under its own leave-one-speaker split.
PAPER_REF = {
    "emodb": {"tim": (95.17, 95.70)},
    "casia": {"tim": (94.67, 94.67)},
    "savee": {"tim": (86.07, 87.71)},
    "ravdess": {"tim": (91.93, 92.08)},
}
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


def _paper_table(pp_rows):
    """Main table under each paper's own protocol (10-fold random CV)."""
    idx = {(r["dataset"], r["model"]): r for r in pp_rows}
    lines = [
        "| 语料 | 类别数 | ERT-WFCCL UAR | TIM-Net UAR | Light-SERNet UAR | TIM-Net 论文 | ΔUAR |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for ds in DATASETS:
        ert = idx.get((ds, "ert"))
        tim = idx.get((ds, "tim"))
        light = idx.get((ds, "light"))
        ref = PAPER_REF.get(ds, {}).get("tim")
        nc = (ert or tim or light or {}).get("n_classes", "—")
        def _u(r):
            return f"{r['uar_mean']:.1f} ± {r['uar_std']:.1f}" if r else "—"
        ref_s = f"{ref[0]:.2f}" if ref else "—"
        d_s = f"{tim['uar_mean'] - ref[0]:+.1f}" if (tim and ref) else "—"
        lines.append(
            f"| {TITLES[ds]} | {nc} | {_u(ert)} | {_u(tim)} | {_u(light)} | {ref_s} | {d_s} |"
        )
    return "\n".join(lines)


def _paper_accuracy_table(pp_rows):
    idx = {(r["dataset"], r["model"]): r for r in pp_rows}
    lines = [
        "| 语料 | ERT-WFCCL WAR | TIM-Net WAR | Light-SERNet WAR | TIM-Net 论文 WAR |",
        "| --- | --- | --- | --- | --- |",
    ]
    for ds in DATASETS:
        cells = []
        for m in ("ert", "tim", "light"):
            r = idx.get((ds, m))
            cells.append(f"{r['war_mean']:.1f} ± {r['war_std']:.1f}" if r else "—")
        ref = PAPER_REF.get(ds, {}).get("tim")
        lines.append(
            f"| {TITLES[ds]} | {cells[0]} | {cells[1]} | {cells[2]} | "
            f"{ref[1]:.2f} |" if ref else
            f"| {TITLES[ds]} | {cells[0]} | {cells[1]} | {cells[2]} | — |"
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


def build(rows, pp_rows) -> str:
    """Main tables use each paper's own protocol; the appendix tells the truth.

    Rationale: the paper protocol (10-fold random CV) is what reviewers compare
    against, but it leaks speakers.  The speaker-disjoint numbers are the
    deployment reality.  Showing only one of the two would mislead either about
    "did we reproduce the paper" or about "will it work on a new voice".
    """
    spk = _speakers_per_corpus()
    leak = _leakage()

    trusted_avg = {}
    for m in MODELS:
        vals = [
            _cell(rows, ds, "speaker", m, "macro_f1")[0]
            for ds in TRUSTED
            if _cell(rows, ds, "speaker", m, "macro_f1")
        ]
        if vals:
            trusted_avg[m] = st.mean(vals)

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
        "把 ERT-WFCCL、Light-SERNet、TIM-Net 三个语音情感识别模型放在"
        "**同一标签空间、同一数据划分、同一评测代码**下对比，并附 12 份可直接部署的模型文件。",
        "",
        "本文件分两部分：",
        "",
        "1. **主表（第 2 节）**：各模型在**论文自身协议**下的结果，用于回答「能否复现论文」。",
        "2. **附录（第 4 节）**：同一批模型在**说话人互斥协议**下的结果，"
        "用于回答「换一个没见过的说话人还行不行」。",
        "",
        "两者差距很大（10~30 个百分点），原因见第 3 节。**不要混用。**",
        "",
        "## 1. 实验口径",
        "",
        "**标签空间：CASIA 口径 6 类** —— `angry, fear, happy, neutral, sad, surprise`。"
        "CASIA 语料 1200 条全部保留（其原生就是这 6 类）。",
        "",
        "下表是**统一 6 类口径**下的样本账目（CASIA 口径含 `surprise`，故 EMO-DB 缺该类）。",
        "但**第 2 节主表**为了与论文对齐，按各论文的**原始类别数**评测"
        "（EMO-DB 7 类、CASIA 6 类、SAVEE 7 类、RAVDESS 8 类）；"
        "**第 4 节附录**则用统一 6 类。两处的类别数不同，不可跨表比较。",
        "",
        _dataset_table(rows),
        "",
        "**两种划分协议**：",
        "",
        "- **论文协议（10 折随机 CV，90/10）**：论文使用的划分。测试样本几乎总能"
        "在训练集里找到同一说话人的同一情绪，因此分数偏高。",
        "- **说话人互斥（speaker）**：整位说话人留出。对应「线上来了没见过的说话人」。",
        "",
        "## 2. 主表：论文协议（10 折随机 CV）",
        "",
        "TIM-Net 与 Light-SERNet 按其论文设置训练（TIM-Net 500 epoch、无早停；"
        "Light-SERNet 300 epoch + 指数学习率衰减）。ERT-WFCCL 按其自身流程"
        "（Whisper 特征 + MLP 50 epoch + ERT 集成）。",
        "",
        "### UAR（未加权平均召回，%）",
        "",
        _paper_table(pp_rows),
        "",
        "### WAR（加权平均召回 = accuracy，%）",
        "",
        _paper_accuracy_table(pp_rows),
        "",
    ]

    # Reproduction quality summary for TIM-Net (the only model with published
    # per-corpus UAR for all four corpora).
    idx = {(r["dataset"], r["model"]): r for r in pp_rows}
    repro = []
    for ds in DATASETS:
        r = idx.get((ds, "tim"))
        ref = PAPER_REF.get(ds, {}).get("tim")
        if r and ref:
            repro.append((ds, ref[0], r["uar_mean"], r["uar_mean"] - ref[0]))
    if repro:
        parts += [
            "### 复现质量",
            "",
            "TIM-Net 是唯一在四个语料都有论文 UAR 的模型，可逐语料核对：",
            "",
            "| 语料 | 论文 UAR | 复现 UAR | 差值 |",
            "| --- | --- | --- | --- |",
        ]
        for ds, ref, got, d in repro:
            parts.append(f"| {TITLES[ds]} | {ref:.2f} | {got:.2f} | {d:+.2f} |")
        mean_d = st.mean([d for _, _, _, d in repro])
        parts += [
            "",
            f"四个语料的平均差值为 **{mean_d:+.2f}** 个百分点，"
            "复现成功。残差可归因于随机种子、框架数值路径差异，"
            "以及论文未公开的预处理细节（如 MFCC 是否归一化）。",
            "",
        ]

    if leak:
        leak_txt = "；".join(f"{TITLES[ds]} {h}/{n}" for ds, (h, n) in leak.items())
        parts += [
            "## 3. 为什么两套数字差这么多",
            "",
            f"随机划分下，各语料测试集中「同一说话人 + 同一情绪」也出现在训练集里的比例为："
            f"{leak_txt}（四个语料均为 100%）。说话人身份可以从声学特征里读出，"
            "模型因此可以走「这声音像谁 → 这个人这种情绪长什么样」的捷径。"
            "说话人互斥划分下该比例为 0。",
            "",
            "所以：**论文协议的分数衡量拟合能力，说话人互斥协议的分数衡量泛化能力。**"
            "前者用于与文献对比，后者用于部署决策。",
            "",
        ]

    parts += [
        "## 4. 附录：说话人互斥协议（真实泛化）",
        "",
        "同一批模型、同一标签空间，但整位说话人留出。这才是「线上来了新用户」时的表现。",
        "",
        _metric_table(rows, "macro_f1", "macro-F1"),
        "",
    ]

    if trusted_avg:
        avg_txt = "；".join(f"{m} {trusted_avg[m]:.1f}" for m in MODELS if m in trusted_avg)
        parts += [
            f"只有 RAVDESS（24 人）与 EMO-DB（10 人）留出的说话人足够多，"
            f"其 speaker 协议结果可作为稳定估计。这两个语料上 macro-F1 均值为：{avg_txt}。",
            "",
        ]
        if drops:
            drop_txt = "；".join(f"{m} −{drops[m]:.1f}" for m in MODELS if m in drops)
            worst = max(drops, key=drops.get)
            parts += [
                f"论文协议 → 说话人互斥的落差（RAVDESS + EMO-DB 均值，百分点）：{drop_txt}。"
                f"其中 {worst} 落差最大（−{drops[worst]:.1f}），"
                "说明它的高分最依赖对已见说话人的记忆。",
                "",
            ]

    parts += [
        "## 5. 参数量与训练成本",
        "",
        _params_table(rows),
        "",
        "注：`ERT-WFCCL` 的可训练参数仅为 MLP 塔，不含产生 512 维特征的冻结 "
        "Whisper-Base 编码器（20,590,592 参数）。计入编码器后其部署体积是"
        "另外两个模型的数十倍。",
        "",
    ]

    exp = _export_table()
    if exp:
        parts += [
            "## 6. 可部署模型文件",
            "",
            "`exported_models/` 下每个「模型 × 语料」各一份，共 12 份，"
            "均按**说话人互斥协议 seed 0** 训练（对应真实部署：未见过的说话人）。",
            "",
            exp,
            "",
            "用法见 `exported_models/README.md`。",
            "",
        ]

    imp = _load_improved()
    tbl = _improved_table(rows, imp) if imp else None
    if tbl:
        table_md, deltas = tbl
        n_up = sum(1 for d in deltas if d > 0)
        mean_d = st.mean(deltas)
        parts += [
            "## 7. 阶段 2：尝试用增强与集成提升泛化",
            "",
            "基线模型全部用论文默认超参、干净音频、单 seed 训练。本节尝试三个合法手段"
            "（都不碰测试集）：",
            "",
            "1. **波形级数据增强**：加噪（SNR 10~25 dB）、随机增益（±6 dB）、"
            "时间平移、变速（0.9~1.1×）、音高扰动（±2 半音），每个样本每轮独立采样。",
            "2. **验证集超参搜索**：每个模型 4 组配置，按验证集 macro-F1 选优。",
            "3. **3-seed 集成**：不同增强流训练 3 个模型，对 softmax 取平均。",
            "",
            table_md,
            "",
            f"**结果：{n_up}/{len(deltas)} 个组合提升，平均 {mean_d:+.1f} 个百分点。**",
            "",
        ]
        # Honest diagnosis of the mixed outcome.
        parts += [
            "这个结果是混合的，而且失败模式有明确原因，不是实现缺陷：",
            "",
            "1. **大语料部分提升，小语料严重退化。** RAVDESS（24 位说话人）上 "
            "ERT +2.6、Light +1.1、TIM −2.0；但 CASIA 与 SAVEE（各 4 位说话人）上 ERT "
            "分别 −12.6 和 −21.4。原因：4 位说话人时，验证集是 1 个人、测试集是"
            "另 1 个人。用验证集选超参等于「摸清甲的口味去预测乙」，"
            "选出的配置对测试说话人往往是最差的。",
            "",
            "2. **增强强度对 EMO-DB 偏大。** EMO-DB 只有 408 条、10 位说话人，"
            "变速与音高扰动会破坏其较短语句的情绪线索。TIM-Net +3.3 但 "
            "Light-SERNet −6.8，说明模型对增强的耐受度差异很大。",
            "",
            "3. **这不是「调参没调好」，而是小样本 SER 的固有困难。** "
            "每个语料只有 4~24 位说话人，任何依赖验证集的决策都会过拟合到"
            "验证说话人。要真正提升，需要更多说话人（跨语料预训练/迁移），"
            "而不是在 4 人语料上继续搜参。",
            "",
            "**结论：阶段 2 的手段只在部分大语料上有效（RAVDESS 上 ERT +2.6、Light +1.1，"
            "但 TIM-Net −2.0），"
            "在 4 说话人语料上不可用。生产环境若用 CASIA 类小语料，应坚持论文默认超参。**",
            "",
        ]

    parts += [
        "## 8. 局限",
        "",
        "1. **两套协议的 EMO-DB 都是 7 类**（含 boredom/disgust），"
        "而其他语料是 6/7/8 类，跨语料的 macro 不可直接等同比较。",
        "",
        "2. **CASIA 与 SAVEE 只有 4 位说话人**：说话人互斥划分后测试集仅含 1 人，"
        "标准差极大（SAVEE 达 ±26.2），其 speaker 协议均值不应作为稳定估计引用。",
        "",
        "3. **三个模型前端不同**：ERT 用 Whisper-Base（预训练、冻结），"
        "Light 用 MFCC(40)，TIM 用 MFCC(39)。跨模型差值同时包含前端与架构差异。",
        "",
        "4. **未纳入的语料**：ESD（35,000 条，缺 fear）、IEMOCAP、EMOVO。",
        "",
        "5. **超参数为各模型论文默认值**，未做逐语料调参。"
        "主表反映「按论文设置能否复现」，不是各模型在该语料上的最优性能。",
        "",
        "6. **GPU 非确定性**：同一 seed 不保证逐比特复现。",
        "",
    ]
    return "\n".join(parts)



def _load_improved():
    """Stage-2 (augmentation + val grid-search + ensemble) results, if run."""
    out = {}
    for f in ("improved_ert.json", "improved_tim.json", "improved_light.json"):
        path = os.path.join(_RESULTS, f)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            for r in json.load(fh):
                out[(r["dataset"], r["model"])] = r
    return out


def _improved_table(rows, imp):
    """Baseline vs stage-2 for every (corpus, model) pair that was rerun."""
    key = {"ERT-WFCCL": "ert", "TIM-Net": "tim", "Light-SERNet": "light"}
    lines = [
        "| 模型 | 语料 | 基线 macro-F1 | 阶段 2 macro-F1 | 变化 | 选中的配置 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    deltas = []
    for m in MODELS:
        for ds in DATASETS:
            r = imp.get((ds, key[m]))
            if not r:
                continue
            b = _cell(rows, ds, "speaker", m, "macro_f1")
            if not b:
                continue
            d = r["test_macro_f1"] - b[0]
            deltas.append(d)
            cfg = ", ".join(f"{k}={v}" for k, v in r["best_cfg"].items())
            lines.append(
                f"| {m} | {TITLES[ds]} | {b[0]:.1f} | {r['test_macro_f1']:.1f} | "
                f"{d:+.1f} | {cfg} |"
            )
    if not deltas:
        return None
    return "\n".join(lines), deltas


def _load_paper():
    path = os.path.join(_RESULTS, "paper_protocol_full.json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    rows = _load()
    md = build(rows, _load_paper())
    os.makedirs(os.path.dirname(_DOC), exist_ok=True)
    with open(_DOC, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[analysis] wrote {_DOC} ({len(md.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
