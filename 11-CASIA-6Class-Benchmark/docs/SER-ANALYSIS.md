# SER 三模型对比分析

本目录把 ERT-WFCCL、Light-SERNet、TIM-Net 三个语音情感识别模型放在**同一标签空间、同一数据划分、同一评测代码**下对比，并附可直接部署的模型文件。

## 1. 实验口径

**标签空间：CASIA 口径 6 类**（`angry, fear, happy, neutral, sad, surprise`）。选这套而非 `disgust` 系，是因为它对应语料自身的类别约定：CASIA 原生就是这 6 类，1200 条全部保留。

需要明确一处数据固有限制：**EMO-DB 语料里没有 `surprise`**（它只有 7 类：angry/boredom/disgust/fear/happy/neutral/sad）。因此 EMO-DB 在该口径下只有 5 类有样本，其 macro 指标按 5 类平均，与其余三个语料（6 类）不可直接等同比较。这是语料本身的类别集合决定的，无法通过调参规避。

| 语料 | 原始 wav | 保留 | angry | fear | happy | neutral | sad | surprise | 说话人 | 丢弃标签 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EMO-DB | 535 | 408 | 127 | 69 | 71 | 79 | 62 | 0 | 10 | boredom: 81, disgust: 46 |
| CASIA | 1200 | 1200 | 200 | 200 | 200 | 200 | 200 | 200 | 4 | — |
| SAVEE | 480 | 420 | 60 | 60 | 60 | 120 | 60 | 60 | 4 | disgust: 60 |
| RAVDESS | 1440 | 1056 | 192 | 192 | 192 | 96 | 192 | 192 | 24 | calm: 192, disgust: 192 |

**划分协议**：两种，回答两个不同的问题。

- **随机划分（random）**：按标签分层随机划分，train/val/test ≈ 72/8/20。对应「线上音频的说话人在训练语料里出现过」的场景。
- **说话人互斥（speaker）**：整位说话人留出，train/val/test 按说话人 70/15/15。对应「线上来了没见过的说话人」的场景。

每个格子为 3 个 seed 的均值 ± 总体标准差。测试集只在训练结束后评测一次，模型选择与早停只看验证集。

## 2. 主结果

### macro-F1（均值 ± 标准差，n=3，单位 %）

| 语料 | 协议 | ERT-WFCCL | Light-SERNet | TIM-Net | ERT−TIM | ERT−Light |
| --- | --- | --- | --- | --- | --- | --- |
| EMO-DB | 随机 | 92.6 ± 2.6 | 83.6 ± 3.8 | 86.1 ± 5.8 | +6.5 | +9.0 |
| EMO-DB | 说话人互斥 | 86.2 ± 0.7 | 72.6 ± 2.9 | 60.1 ± 11.9 | +26.1 | +13.6 |
| CASIA | 随机 | 84.1 ± 0.5 | 83.6 ± 2.5 | 81.6 ± 4.6 | +2.5 | +0.4 |
| CASIA | 说话人互斥 | 40.3 ± 7.5 | 20.6 ± 7.7 | 14.8 ± 5.0 | +25.4 | +19.7 |
| SAVEE | 随机 | 72.6 ± 1.5 | 73.7 ± 4.1 | 75.2 ± 6.5 | -2.6 | -1.1 |
| SAVEE | 说话人互斥 | 32.3 ± 26.2 | 28.8 ± 18.8 | 14.5 ± 5.4 | +17.8 | +3.5 |
| RAVDESS | 随机 | 87.7 ± 1.7 | 80.0 ± 2.1 | 85.0 ± 3.1 | +2.6 | +7.7 |
| RAVDESS | 说话人互斥 | 76.7 ± 3.4 | 61.5 ± 5.0 | 51.7 ± 4.1 | +25.0 | +15.2 |

### Accuracy（均值 ± 标准差，n=3，单位 %）

| 语料 | 协议 | ERT-WFCCL | Light-SERNet | TIM-Net | ERT−TIM | ERT−Light |
| --- | --- | --- | --- | --- | --- | --- |
| EMO-DB | 随机 | 93.0 ± 2.9 | 84.4 ± 4.1 | 86.0 ± 6.7 | +7.0 | +8.6 |
| EMO-DB | 说话人互斥 | 87.2 ± 0.8 | 71.9 ± 3.0 | 64.7 ± 8.8 | +22.5 | +15.3 |
| CASIA | 随机 | 84.0 ± 0.5 | 83.6 ± 2.6 | 81.8 ± 4.5 | +2.2 | +0.4 |
| CASIA | 说话人互斥 | 43.2 ± 7.2 | 27.9 ± 7.1 | 24.8 ± 3.9 | +18.4 | +15.3 |
| SAVEE | 随机 | 76.2 ± 1.0 | 76.2 ± 3.9 | 78.2 ± 5.6 | -2.0 | +0.0 |
| SAVEE | 说话人互斥 | 46.7 ± 19.6 | 43.5 ± 15.1 | 28.6 ± 4.7 | +18.1 | +3.2 |
| RAVDESS | 随机 | 88.2 ± 1.6 | 81.2 ± 2.7 | 85.0 ± 3.0 | +3.2 | +7.0 |
| RAVDESS | 说话人互斥 | 77.1 ± 3.0 | 63.6 ± 3.2 | 55.7 ± 2.8 | +21.4 | +13.4 |

### 为什么随机划分会高估

随机划分下，各语料测试集中「同一说话人 + 同一情绪」也出现在训练集里的比例为：EMO-DB 81/81；CASIA 240/240；SAVEE 84/84；RAVDESS 209/209（即四个语料均为 100%）。说话人身份可以从声学特征里读出，模型因此可以走「这声音像谁 → 这个人这种情绪长什么样」的捷径。说话人互斥划分下该比例为 0。

## 3. 结论：泛化到未见说话人时选谁

只有 RAVDESS（24 人）与 EMO-DB（10 人）留出的说话人足够多，其 speaker 协议结果可作为稳定估计。这两个语料上 macro-F1 均值为：ERT-WFCCL 81.4；Light-SERNet 67.0；TIM-Net 55.9。

**ERT-WFCCL 在 4/4 个语料的说话人互斥协议下排名第一。**

随机 → 说话人互斥的落差（RAVDESS + EMO-DB 均值，百分点）：ERT-WFCCL −8.7；Light-SERNet −14.8；TIM-Net −29.7。其中 TIM-Net 落差最大（−29.7），说明它的随机划分分数最依赖对已见说话人的记忆。

**重要限定**：ERT-WFCCL 的领先同时包含前端与架构的差异 —— 它使用冻结的 Whisper-Base 编码器（20,590,592 参数，在海量语音上预训练），而另外两个模型从零训练 MFCC 前端。因此这个结果不能单独归因于架构优劣。

## 4. 参数量与训练成本

| 模型 | 可训练参数 | 冻结前端 | 总参数 | 单次训练耗时（均值） |
| --- | --- | --- | --- | --- |
| ERT-WFCCL | 173,638 | 20,590,592 | 20,764,230 | 1.1 s |
| Light-SERNet | 463,110 | 0 | 463,110 | 17.3 s |
| TIM-Net | 106,952 | 0 | 106,952 | 35.0 s |

注：`ERT-WFCCL` 的可训练参数仅为 MLP 塔（512→256→128→64 + 临时 6 类分类层），不含产生 512 维特征的冻结 Whisper-Base 编码器。若计入编码器，其部署体积是另外两个模型的数十倍 —— 在有模型体积约束（如 2 MB 预算）的场景下不可用。

## 5. 可部署模型文件

`exported_models/` 下每个「模型 × 语料」各一份，均按 **说话人互斥协议 seed 0** 训练（对应真实部署：未见过的说话人）。格式按端侧部署选择：

| 模型 | 语料 | 文件 | 格式 | 体积 |
| --- | --- | --- | --- | --- |
| ERT-WFCCL | EMO-DB | `ERT-WFCCL_emodb.pt` | torch state_dict | 697 KB |
| ERT-WFCCL | CASIA | `ERT-WFCCL_casia.pt` | torch state_dict | 697 KB |
| ERT-WFCCL | SAVEE | `ERT-WFCCL_savee.pt` | torch state_dict | 697 KB |
| ERT-WFCCL | RAVDESS | `ERT-WFCCL_ravdess.pt` | torch state_dict | 697 KB |
| Light-SERNet | EMO-DB | `Light-SERNet_emodb_int8.tflite` | tflite int8 | 480 KB |
| Light-SERNet | CASIA | `Light-SERNet_casia_int8.tflite` | tflite int8 | 481 KB |
| Light-SERNet | SAVEE | `Light-SERNet_savee_int8.tflite` | tflite int8 | 481 KB |
| Light-SERNet | RAVDESS | `Light-SERNet_ravdess_int8.tflite` | tflite int8 | 481 KB |
| TIM-Net | EMO-DB | `TIM-Net_emodb.hdf5` | keras hdf5 weights | 763 KB |
| TIM-Net | CASIA | `TIM-Net_casia.hdf5` | keras hdf5 weights | 763 KB |
| TIM-Net | SAVEE | `TIM-Net_savee.hdf5` | keras hdf5 weights | 767 KB |
| TIM-Net | RAVDESS | `TIM-Net_ravdess.hdf5` | keras hdf5 weights | 767 KB |

合计 12 个文件，7.6 MB。

各格式说明：

- **ERT-WFCCL `.pt`**：PyTorch `state_dict`（MLP 塔）+ StandardScaler 的均值/方差 + 类别表。**不含 Whisper-Base 编码器** —— 部署时需自行提供 512 维 Whisper 特征。
- **Light-SERNet `.tflite`**：int8 量化，可直接在移动端/嵌入式 TFLite 运行时加载。
- **TIM-Net `.hdf5`**：Keras 权重，需配套 `TIM-Net/Code/TIMNET.py` 的模型定义。

用法见 `exported_models/README.md`。

## 6. 局限

1. **EMO-DB 只有 5 类**：语料无 `surprise`，其 macro 指标按 5 类平均，不能与 CASIA / SAVEE / RAVDESS 的 6 类 macro 直接等同比较。

2. **CASIA 与 SAVEE 只有 4 位说话人**：说话人互斥划分后测试集仅含 1 人，标准差极大（SAVEE 达 ±26.2），其 speaker 协议均值不应作为稳定估计引用。这两个语料的随机划分结果相对可信。

3. **三个模型的前端不同**：ERT 用 Whisper-Base（预训练、冻结），Light 用 MFCC(40)，TIM 用 MFCC(39)。跨模型差值同时包含前端与架构差异，不是特征受控比较。

4. **未纳入的语料**：ESD（35,000 条，只有 angry/happy/neutral/sad/surprise 五类，在本口径下缺 `fear`，因此无法与其他语料对齐 6 类）、IEMOCAP、EMOVO。中文朗读语料同样未纳入。

5. **超参数固定为各模型论文默认值**，未做逐语料调参。因此结果反映「开箱即用」的相对水平，而非各模型在该语料上的最优性能。

6. **GPU 非确定性**：cuDNN/cuBLAS 可能选择不同内核路径，同一 seed 不保证逐比特复现。
