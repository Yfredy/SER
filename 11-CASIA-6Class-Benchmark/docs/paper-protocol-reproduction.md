# 论文协议复现（原始类别 + 论文自身划分）

本文件回答「能否复现论文数字」，与统一 6 类基准
（`ser-benchmark-comparison.md`）回答的「哪个模型泛化到未见说话人更好」是**两个不同的问题**。

两者的差别不是小数点级别：同一模型、同一语料，换协议后能差 20~30 个百分点。混用会得出相反结论。

## 复现结果

| 语料 | 模型 | 划分 | 类别数 | 样本 | UAR (%) | WAR/Acc (%) | 论文 UAR | 论文 WAR | ΔUAR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RAVDESS | TIM-Net | random 10 折 | 8 | 1440 | 88.50 ± 2.32 | 88.40 ± 2.43 | 91.93 | 92.08 | −3.43 |
| EMO-DB | TIM-Net | random 10 折 | 7 | 535 | 89.02 ± 5.17 | 89.51 ± 4.73 | 95.17 | 95.70 | −6.15 |
| RAVDESS | Light-SERNet | random 10 折 | 8 | 1440 | 87.47 ± 2.65 | 87.36 ± 2.15 | — | — | — |
| EMO-DB | Light-SERNet | random 10 折 | 7 | 535 | 76.46 ± 12.59 | 76.28 ± 13.22 | — | 94.15 | — |
| EMO-DB | Light-SERNet | **留一说话人** | 7 | 535 | 61.33 ± 13.52 | 66.50 ± 12.75 | — | 94.15 | — |

UAR = 未加权平均召回（macro recall）；WAR = 加权平均召回 = accuracy。

## 结论一：TIM-Net 复现到论文的 93%~96% 水平

| 语料 | 论文 UAR | 复现 UAR | 复现率 |
| --- | --- | --- | --- |
| RAVDESS | 91.93 | 88.50 | 96.3% |
| EMO-DB | 95.17 | 89.02 | 93.5% |

差 3.4 / 6.2 个百分点，方向一致、量级合理。**复现是成功的**，剩余差距可归因于：
- 随机种子（论文 seed=46，划分本身随机）
- 硬件/框架版本差异导致的数值路径不同
- 论文未公开的细节（如是否对 MFCC 做归一化）

EMO-DB 折间标准差 ±5.17（范围 79.7~95.0）说明这个语料本身对划分敏感，6.2 分的差距在噪声范围内并不异常。

## 结论二：Light-SERNet 的 94.15% 无法用随机折复现，且它其实不是随机折

这是本次复现最重要的发现。

**官方代码的 EMO-DB 协议是「留一说话人」**，不是随机 10 折：
`LIGHT-SERNET/filter_dataset.py:65` 的 `seperate_speaker_id_emodb` 按说话人 ID
（`["03","08","09","10","11","12","13","14","15","16"]`，共 10 人）返回 10 组索引，
`train.py:123` 按 `K_FOLD=10` 逐组留出。10 折 = 10 位说话人，一人一折。

我按这个协议重跑，得到 **UAR 61.33 / Acc 66.50**，距论文的 94.15 差 **27.7 个点**。

差距的来源已定位到三处，且是叠加的：

1. **官方用测试集做模型选择。** `train.py:113-115` 把 `x[test]` 同时传给
   `validation_data` 和 `evaluate`，`BestModelWeights` 回调按 `val_accuracy`
   挑权重 —— 而 `val_accuracy` 就是在测试集上算的。等于每折在测试集上挑了最好的 epoch。
   我的复现用训练集切出的 10% 做验证，测试集全程不参与选择。
2. **官方 EMO-DB 用 7 类**（含 `boredom`），`boredom` 与 `neutral` 在 EMO-DB 中高度可分，
   而 `boredom` 样本占 81/535 = 15%。类别更容易 + 样本更多。
3. **官方跑 300 epoch 且带学习率衰减**（`exp(-0.15)` 每 20 epoch）。
   我最初漏了衰减，补上后结果反而更低（61.3 vs 68.5），说明瓶颈不在优化器。

**结论：94.15% 这个数字包含了测试集泄漏，不应作为泛化性能引用。**
它衡量的是「在见过的说话人上、且允许用测试集挑 epoch」时的拟合能力。

## 结论三：Light-SERNet 在 EMO-DB 上极不稳定

随机 10 折的逐折 UAR：

```
45.5  82.1  65.2  80.3  92.7  74.9  76.9  75.1  87.5  84.3
```

极差 47.2，标准差 ±12.59。留一说话人协议下同样不稳（37.0~81.3，±13.52）。

对比 TIM-Net 在 EMO-DB：`90.5 91.9 94.3 87.0 95.0 81.4 92.8 92.7 79.7 85.0`，极差 15.3，±5.17。

**Light-SERNet 的 EMO-DB 结果高度依赖划分**，单次运行的数字不可信，必须报均值±标准差。

## 与统一 6 类基准的关系

同一模型在两个口径下的对比（EMO-DB）：

| 口径 | Light-SERNet | TIM-Net | 说明 |
| --- | --- | --- | --- |
| 论文协议（7 类、随机折） | 76.46 | 89.02 | 含说话人泄漏 |
| CASIA 口径 6 类、随机折 | 83.6 | 86.1 | 去掉 boredom，加入 surprise |
| CASIA 口径 6 类、说话人互斥 | 72.6 | 60.1 | 无泄漏，最接近真实部署 |

注意第一行与第二行 Light 的排序反转（76.5 → 83.6）：**类别集合从 7 类减到 6 类，
Light-SERNet 反而提升**，而 TIM-Net 略降。这说明两个模型对「去掉 boredom」的敏感度不同，
进一步说明跨论文、跨口径的数字不能直接横向比较。

（第三行来自 `ser-benchmark-comparison.md` 与 `SER-ANALYSIS.md` 的 CASIA 口径基准。）

## 复现方法

```bash
# TIM-Net：论文随机 10 折
TF_USE_LEGACY_KERAS=1 ~/miniconda3/envs/tf-gpu/bin/python \
    -m benchmark_ser.reproduce_paper --models tim --datasets ravdess emodb \
    --split random --folds 10 --epochs 500

# Light-SERNet：官方留一说话人
TF_USE_LEGACY_KERAS=1 ~/miniconda3/envs/tf-gpu/bin/python \
    -m benchmark_ser.reproduce_paper --models light --datasets emodb \
    --split speaker --epochs 300
```

原始结果：`benchmark_ser/results/paper_protocol.json`。
耗时：TIM-Net 两语料约 48 分钟，Light-SERNet 每语料约 7 分钟。

## 复现的边界

- **只覆盖 RAVDESS 与 EMO-DB**。CASIA / SAVEE 未跑（各仅 4 位说话人，留一划分下测试集只有
  1 人，统计意义有限）。CASIA / SAVEE 的论文数字（94.67 / 86.07）未验证。
- **IEMOCAP 与 EMOVO 未纳入**（本地无数据）。
- **Light-SERNet 的 RAVDESS 数字论文未报告**，表中 87.47 是我的实测，无对照。
- 论文的 seed、数据预处理细节未完全公开，3.4~6.2 分的残差无法进一步归因。
