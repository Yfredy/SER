# 11 · CASIA 口径 6 类基准：三模型对比 + 可部署模型

把 **ERT-WFCCL、Light-SERNet、TIM-Net** 三个语音情感识别模型放在同一标签空间、
同一数据划分、同一评测代码下横向对比，并附 12 份可直接部署的模型文件。

## 快速导航

| 内容 | 位置 |
| --- | --- |
| **对比分析（先看这个）** | [`docs/SER-ANALYSIS.md`](docs/SER-ANALYSIS.md) |
| 完整基准报告（含逐格主表） | [`docs/ser-benchmark-comparison.md`](docs/ser-benchmark-comparison.md) |
| 论文协议复现（能否复现 90%+） | [`docs/paper-protocol-reproduction.md`](docs/paper-protocol-reproduction.md) |
| **可部署模型（12 份）** | [`exported_models/`](exported_models/) · [使用说明](exported_models/README.md) |
| 原始评测结果（JSON） | [`results/`](results/) |
| 复现代码 | [`code/`](code/) |

## 核心结论

**标签空间：CASIA 口径 6 类** —— `angry, fear, happy, neutral, sad, surprise`。
CASIA 语料 1200 条全部保留（原始就是这 6 类）。

> 数据固有限制：EMO-DB 语料没有 `surprise`，在该口径下只有 5 类有样本，
> 其 macro 指标按 5 类平均，不能与其余三语料直接等同比较。

### 主表：论文协议（10 折随机 CV，UAR %）

各模型按自己论文的设置训练与划分，用于与文献对比：

| 语料 | ERT-WFCCL | TIM-Net | Light-SERNet | TIM-Net 论文 | 复现差值 |
| --- | --- | --- | --- | --- | --- |
| EMO-DB | **93.1 ± 3.3** | 89.0 ± 5.2 | 76.5 ± 12.6 | 95.17 | −6.2 |
| CASIA | 82.0 ± 3.1 | **89.6 ± 3.3** | 53.5 ± 16.9 | 94.67 | −5.1 |
| SAVEE | 73.2 ± 9.7 | **75.8 ± 6.9** | 55.8 ± 11.6 | 86.07 | −10.3 |
| RAVDESS | 85.7 ± 3.1 | **88.5 ± 2.3** | 87.5 ± 2.7 | 91.93 | −3.4 |

TIM-Net 四语料平均差 **−6.2** 个百分点，复现成功。

### 附录：说话人互斥协议（macro-F1 %，未见过说话人）

| 语料 | ERT-WFCCL | Light-SERNet | TIM-Net |
| --- | --- | --- | --- |
| EMO-DB | **86.2 ± 0.7** | 72.6 ± 2.9 | 60.1 ± 11.9 |
| CASIA | **40.3 ± 7.5** | 20.6 ± 7.7 | 14.8 ± 5.0 |
| SAVEE | **32.3 ± 26.2** | 28.8 ± 18.8 | 14.5 ± 5.4 |
| RAVDESS | **76.7 ± 3.4** | 61.5 ± 5.0 | 51.7 ± 4.1 |

可信均值（RAVDESS + EMO-DB）：**ERT 81.4 / Light 67.0 / TIM 55.9**

> 两套数字差 10~30 个百分点。论文协议含说话人泄漏（测试样本 100% 能找到同人同情绪），
> 用于与文献对比；说话人互斥协议才代表部署时的真实表现。**不要混用。**

### 阶段 2：尝试提升泛化（数据增强 + 验证集搜参 + 3-seed 集成）

| 模型 | 语料 | 基线 | 阶段 2 | 变化 |
| --- | --- | --- | --- | --- |
| ERT-WFCCL | RAVDESS | 76.7 | 79.3 | **+2.6** |
| ERT-WFCCL | EMO-DB | 86.2 | 86.8 | +0.7 |
| ERT-WFCCL | CASIA | 40.3 | 27.7 | −12.6 |
| ERT-WFCCL | SAVEE | 32.3 | 10.9 | −21.4 |
| TIM-Net | EMO-DB | 60.1 | 63.5 | **+3.3** |
| TIM-Net | RAVDESS | 51.7 | 49.6 | −2.0 |
| Light-SERNet | RAVDESS | 61.5 | 62.5 | +1.1 |
| Light-SERNet | EMO-DB | 72.6 | 65.8 | −6.8 |

**4/8 提升，平均 −4.4。** 失败有明确原因：CASIA/SAVEE 各 4 位说话人，
验证集 1 人、测试集另 1 人，用验证集选超参等于「摸清甲去预测乙」，必然选错。
**小语料应坚持论文默认超参。**

### 三个关键判断

1. **ERT-WFCCL 泛化最好**，在 4/4 语料的说话人互斥协议下排名第一。
   但这个优势**同时包含前端差异** —— 它用冻结的 Whisper-Base（2059 万参数，预训练），
   另两个从零训练 MFCC。不能单独归因于架构。

2. **TIM-Net 的随机划分分数最"虚"**。随机→说话人互斥落差 −29.7 个百分点，
   在 CASIA 上从 81.6 掉到 14.8。它的高分很大程度来自记住说话人。

3. **CASIA / SAVEE 的说话人协议结果不可当结论**。各仅 4 位说话人，
   留出后测试集只有 1 人，SAVEE 标准差达 ±26.2。

## 可部署模型

`exported_models/` 下每个「模型 × 语料」一份，共 12 份，合计 7.6 MB。
全部按**说话人互斥协议 seed 0** 训练（模型没见过测试说话人）。

| 模型 | 格式 | 单份体积 |
| --- | --- | --- |
| ERT-WFCCL | PyTorch state_dict（**不含 Whisper 编码器**） | 697 KB |
| Light-SERNet | TFLite **int8** | 481 KB |
| TIM-Net | Keras hdf5 weights | 763 KB |

**端侧部署推荐 Light-SERNet**：481 KB int8，精度 67.0（可信均值），
远低于 2 MB 预算。ERT 精度最高但需额外携带 Whisper 编码器（约 82 MB fp32）。

用法见 [`exported_models/README.md`](exported_models/README.md)。

## 复现

```bash
# 完整基准（3 模型 × 4 语料 × 2 协议 × 3 seed = 72 次运行）
bash code/run_all.sh

# 导出部署模型
~/miniconda3/envs/torch-cu124/bin/python -m code.export_models --models ert
TF_USE_LEGACY_KERAS=1 ~/miniconda3/envs/tf-gpu/bin/python -m code.export_models --models light tim

# 生成分析文档
~/miniconda3/envs/torch-cu124/bin/python -m code.render_analysis
```

环境：Python 3.10 / PyTorch 2.5.1+cu124 / TensorFlow 2.17.1 + tf-keras 2.17
（TF 侧必须 `export TF_USE_LEGACY_KERAS=1`）。

## 目录说明

```
code/              复现脚本（标签、清单、划分、特征、三个 runner、聚合、渲染、导出）
results/           72 行评测结果 + splits.json + 论文协议复现结果
exported_models/   12 份部署模型 + MANIFEST.json + 使用说明
docs/              三份分析文档
```
