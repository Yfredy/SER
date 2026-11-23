# SER 三模型四语料对比报告

CASIA 口径 6 类情绪空间（`angry, fear, happy, neutral, sad, surprise`）下，ERT-WFCCL、Light-SERNet、TIM-Net 在 EMO-DB、CASIA、SAVEE、RAVDESS 四个语料上的随机协议与说话人协议基准对比。所有数字由 `benchmark_ser/results/merged.json` 与 `manifest.audit()` 实时计算。

## 1. 环境与方法

| 项目 | 值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4080 SUPER 16GB |
| Python | 3.10 |
| PyTorch | 2.5.1+cu124 |
| TensorFlow | 2.17.1 + tf-keras 2.17（运行时须 `export TF_USE_LEGACY_KERAS=1`，TIM-Net 的 `K.reverse` 在 Keras 3 下会报错） |

### 协议定义

- **random（随机划分）**：按标签分层随机划分，每类测试集约 20%、验证集取剩余样本的约 10%（train/val/test ≈ 72/8/20）；每个非空类别保证至少有 1 个测试样本。
- **speaker（说话人划分）**：按说话人互斥划分，名义比例 70/15/15；验证集与测试集各保证至少 1 位说话人。若测试集缺类，则整体迁移携带该类的训练集说话人（保持说话人互斥）。说话人很少时名义比例无法精确达成，实际按说话人计的 train/val/test 为：EMO-DB 6/2/2；CASIA 2/1/1；SAVEE 2/1/1；RAVDESS 16/4/4（说话人粒度导致实际样本比例偏离名义比例）。

### 可比性边界

三个模型各自沿用其原论文前端（ERT-WFCCL：Whisper-Base；Light-SERNet：MFCC(40)；TIM-Net：MFCC(39)），因此 `ERT−TIM` 与 `ERT−Light` 差值并非特征受控（feature-controlled）比较，而是同时度量前端与架构的差异。同一模型内部的比较不受前端差异影响，但跨语料比较仍受各语料类别集合差异（EMO-DB 无 `surprise`）与划分比例差异的影响，见第 6 节。

### 固定超参数（各模型沿用论文默认值，不做逐语料调参）

| 模型 | 特征 | 模型/训练配置 |
| --- | --- | --- |
| ERT-WFCCL | Whisper-Base 512 维全局特征 → StandardScaler（仅训练集拟合） | MLP 512→256→128→64（dropout 0.3/0.2，Adam lr=5e-4，batch 64，epochs 50，CosineAnnealingWarmRestarts T_0=50，验证集早停）→ ERTWFCCL(eps=0.5, n_estimators=200) |
| Light-SERNet | 16 kHz 单声道，居中补齐/裁剪至 3.0 s → (184, 40, 1) MFCC | Light_SERNet_V1；Adam(1e-4) + SparseCategoricalFocalLoss(γ=2)，batch 32，epochs 200，patience 40（验证集早停） |
| TIM-Net | librosa MFCC (188, 39) float32 | TIMNET(nb_filters=39, kernel_size=2, nb_stacks=1, dilations=8, dropout=0.1, activation=relu) + WeightLayer + softmax Dense；Adam(1e-3, β₁=0.93, β₂=0.98) + 交叉熵（标签平滑 0.1），batch 64，epochs 200，patience 40 |

## 2. 数据集一览

CASIA 口径 6 类情绪空间：`angry, fear, happy, neutral, sad, surprise`。下表为各语料原始 wav 数、统一 6 类下的保留数、各类样本数（全部六类均列出，缺失类为 0）、说话人数与被丢弃标签。

| 语料 | 原始 wav | 保留 (6 类) | angry | fear | happy | neutral | sad | surprise | 说话人 | 被丢弃标签 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EMO-DB | 535 | 408 | 127 | 69 | 71 | 79 | 62 | 0 | 10 | boredom: 81, disgust: 46 |
| CASIA | 1200 | 1200 | 200 | 200 | 200 | 200 | 200 | 200 | 4 | — |
| SAVEE | 480 | 420 | 60 | 60 | 60 | 120 | 60 | 60 | 4 | disgust: 60 |
| RAVDESS | 1440 | 1056 | 192 | 192 | 192 | 96 | 192 | 192 | 24 | calm: 192, disgust: 192 |

## 3. 主表：`语料 × 协议 × 模型`

每个格子为 3 个 seed 的 `均值 ± 标准差 (n=3)`。差值列为 `ERT−TIM` 与 `ERT−Light`（ERT-WFCCL 减去对应模型的均值，单位：百分点）。

### macro-F1（均值 ± 标准差，n=3，单位 %）

| 语料 | 协议 | ERT-WFCCL | Light-SERNet | TIM-Net | ERT−TIM | ERT−Light | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EMO-DB | 随机划分 (random) | 92.6 ± 2.6 (n=3) | 83.6 ± 3.8 (n=3) | 86.1 ± 5.8 (n=3) | +6.5 | +9.0 |  |
| EMO-DB | 说话人划分 (speaker) | 86.2 ± 0.7 (n=3) | 72.6 ± 2.9 (n=3) | 60.1 ± 11.9 (n=3) | +26.1 | +13.6 |  |
| CASIA | 随机划分 (random) | 84.1 ± 0.5 (n=3) | 83.6 ± 2.5 (n=3) | 81.6 ± 4.6 (n=3) | +2.5 | +0.4 | EMO-DB 无 `surprise` 类：6 类仅 5 类有样本，macro 指标按 5 类平均，不可与其他三语料（6 类）直接比较 |
| CASIA | 说话人划分 (speaker) | 40.3 ± 7.5 (n=3) | 20.6 ± 7.7 (n=3) | 14.8 ± 5.0 (n=3) | +25.4 | +19.7 | EMO-DB 无 `surprise` 类：6 类仅 5 类有样本，macro 指标按 5 类平均，不可与其他三语料（6 类）直接比较 |
| SAVEE | 随机划分 (random) | 72.6 ± 1.5 (n=3) | 73.7 ± 4.1 (n=3) | 75.2 ± 6.5 (n=3) | -2.6 | -1.1 |  |
| SAVEE | 说话人划分 (speaker) | 32.3 ± 26.2 (n=3) | 28.8 ± 18.8 (n=3) | 14.5 ± 5.4 (n=3) | +17.8 | +3.5 |  |
| RAVDESS | 随机划分 (random) | 87.7 ± 1.7 (n=3) | 80.0 ± 2.1 (n=3) | 85.0 ± 3.1 (n=3) | +2.6 | +7.7 |  |
| RAVDESS | 说话人划分 (speaker) | 76.7 ± 3.4 (n=3) | 61.5 ± 5.0 (n=3) | 51.7 ± 4.1 (n=3) | +25.0 | +15.2 |  |

### Accuracy（均值 ± 标准差，n=3，单位 %）

| 语料 | 协议 | ERT-WFCCL | Light-SERNet | TIM-Net | ERT−TIM | ERT−Light | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EMO-DB | 随机划分 (random) | 93.0 ± 2.9 (n=3) | 84.4 ± 4.1 (n=3) | 86.0 ± 6.7 (n=3) | +7.0 | +8.6 |  |
| EMO-DB | 说话人划分 (speaker) | 87.2 ± 0.8 (n=3) | 71.9 ± 3.0 (n=3) | 64.7 ± 8.8 (n=3) | +22.5 | +15.3 |  |
| CASIA | 随机划分 (random) | 84.0 ± 0.5 (n=3) | 83.6 ± 2.6 (n=3) | 81.8 ± 4.5 (n=3) | +2.2 | +0.4 | EMO-DB 无 `surprise` 类：6 类仅 5 类有样本，macro 指标按 5 类平均，不可与其他三语料（6 类）直接比较 |
| CASIA | 说话人划分 (speaker) | 43.2 ± 7.2 (n=3) | 27.9 ± 7.1 (n=3) | 24.8 ± 3.9 (n=3) | +18.4 | +15.3 | EMO-DB 无 `surprise` 类：6 类仅 5 类有样本，macro 指标按 5 类平均，不可与其他三语料（6 类）直接比较 |
| SAVEE | 随机划分 (random) | 76.2 ± 1.0 (n=3) | 76.2 ± 3.9 (n=3) | 78.2 ± 5.6 (n=3) | -2.0 | +0.0 |  |
| SAVEE | 说话人划分 (speaker) | 46.7 ± 19.6 (n=3) | 43.5 ± 15.1 (n=3) | 28.6 ± 4.7 (n=3) | +18.1 | +3.2 |  |
| RAVDESS | 随机划分 (random) | 88.2 ± 1.6 (n=3) | 81.2 ± 2.7 (n=3) | 85.0 ± 3.0 (n=3) | +3.2 | +7.0 |  |
| RAVDESS | 说话人划分 (speaker) | 77.1 ± 3.0 (n=3) | 63.6 ± 3.2 (n=3) | 55.7 ± 2.8 (n=3) | +21.4 | +13.4 |  |

## 4. 该看哪一列：部署场景 → 协议

本文的两种协议回答两个不同的问题，选错会让选型和上线预期都偏掉：

| 部署场景 | 线上说话人是否在训练语料中出现过 | 应看协议 |
| --- | --- | --- |
| 固定说话人（如少数配音演员的语料库产品） | 是 | **random** |
| 任意用户上传音频、新录音 | 否 | **speaker** |

### 为什么 random 协议会高估

随机划分下，各语料测试集中「同一说话人 + 同一情绪」也出现在训练集里的比例为：EMO-DB 81/81；CASIA 240/240；SAVEE 84/84；RAVDESS 209/209。说话人身份本身就能从声学特征里读出，因此模型可以走「这声音像谁 → 这个人这种情绪长什么样」的捷径，而不必学到情绪本身的声学模式。作为对照，speaker 协议下该比例为 0（训练集与测试集说话人互斥），这正是「线上来了没见过的说话人」所对应的条件。

### 未见过说话人时的可信结论

只有 RAVDESS + EMO-DB 的留出说话人足够多（4 人 / 2 人），其 speaker 协议结果可作为稳定估计；CASIA 与 SAVEE 各仅 4 位说话人、测试集只有 1 人，标准差过大，不应被当作结论引用。在这两个语料上按 speaker 协议的 macro-F1 均值为：ERT-WFCCL 81.4；Light-SERNet 67.0；TIM-Net 55.9。

**按 speaker 协议，ERT-WFCCL 在 4/4 个语料上排名第一**（逐语料排名见第 3 节主表；差值列已给出逐格对比）。需要强调：该优势同时包含前端与架构差异（见「可比性边界」），不能单独归因于架构优劣。

### 场景选择的量化影响

同一模型在 random 与 speaker 两个协议下 macro-F1 的落差（RAVDESS + EMO-DB 均值）为：ERT-WFCCL −8.7；Light-SERNet −14.8；TIM-Net −29.7（单位：百分点；负值表示 speaker 协议更低，即换成没见过的说话人后掉这么多）。其中 TIM-Net 的落差最大（−29.7），说明它的随机协议分数最依赖对已见说话人的记忆。若按 random 列做容量规划而实际部署面对新说话人，预期会被系统性高估。

## 5. 参数量与单次训练耗时

| 模型 | 参数量 | 24 次运行 train_s 合计 | 单次运行 train_s 均值 |
| --- | --- | --- | --- |
| ERT-WFCCL | 173,638 | 25.4 s | 1.1 s |
| Light-SERNet | 463,110 | 415.5 s | 17.3 s |
| TIM-Net | 106,952 | 840.2 s | 35.0 s |

注：`train_s` 为单次运行的训练墙钟时间（特征提取与预测不计入）。各 run 的 `train_s` 从亚秒级到约 60 s 不等、噪声占主导，见「局限」第 4 条。

注：ERT-WFCCL 的参数量 173,638 仅为其可训练的 MLP 塔（512→256→128→64，含 BatchNorm 与临时 6 类分类层），不含产生 512 维特征的冻结 Whisper-Base 编码器（20,590,592 参数，训练中不更新）。Light-SERNet 与 TIM-Net 的 MFCC 前端无参数，故其参数量为端到端计数。

## 6. 局限

1. **EMO-DB 没有 `surprise` 类**（5 类有样本），其 macro 指标是按 5 个类平均，而 CASIA / SAVEE / RAVDESS 按 6 个类平均，因此 EMO-DB 的 macro-F1 不能与其他三个语料直接等同比较（主表中已在 EMO-DB 行旁标注）。

2. **说话人协议方差大**：CASIA 与 SAVEE 各仅 4 / 4 位说话人，按说话人切分后测试集只含 1 位说话人（实际分配见第 1 节），实测标准差显著大于 random 协议。具体地，SAVEE speaker 协议：seed 0 留出 KL；seed 1 留出 KL；seed 2 留出 JE。三个 seed 并非三个等难度的抽样：它们仅覆盖 2 位不同说话人，其中 KL 被 2 个 seed 重复留出，因此 SAVEE speaker 协议的标准差极大（约 ±26.2 个 F1 百分点），其 `均值 ± 标准差` 不应被当作稳定估计。

3. **ESD（35,000 条）与中文朗读语料未纳入本轮**：ESD 体量会使单轮全量实验超过 10 小时，且其只有 5 类，与 CASIA 口径 6 类不一致；如需纳入应作为独立扩展计划。

4. **训练耗时噪声**：各模型 24 次运行的 `train_s` 跨度为 ERT-WFCCL 0.6–2.3 s；Light-SERNet 9.3–33.7 s；TIM-Net 18.4–69.9 s，单次运行从亚秒级到约 60 s 不等、噪声占主导，因此逐 seed 的耗时差异不具意义，只有跨模型的合计（见第 5 节）才具有参考价值。

5. **指标分母**：macro 指标按各测试集实际出现的类别数（`n_classes_present`）取平均，即 CASIA 为 6、其余语料为 5；读者不应把 CASIA 的 macro-F1 与其他语料朴素地直接比较。

6. **标准差类型**：主表中所有的 `±` 均为总体标准差（`statistics.pstdev`，ddof=0），即直接度量 3 个 seed 的离散程度，而非对更大总体的无偏估计。

7. **GPU 非确定性**：cuDNN / cuBLAS 的默认选择可能因运行时条件选择不同内核路径，导致同一 seed 在不同硬件或驱动下结果不可逐比特复现；本报告的 `n=3` 交叉 seed 平均在一定程度上吸收了该波动。
