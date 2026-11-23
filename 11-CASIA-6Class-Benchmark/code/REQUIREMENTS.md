# 运行依赖

代码默认按以下**同级目录**布局查找三个模型的官方实现与数据：

```
<工作根>/
├── code/                    # 本目录（benchmark_ser 包内容）
├── results/                 # 输出：评测 JSON
├── ERT-WFCCL/projects/ERT-WFCCL/     # ERT-WFCCL 官方代码
├── _local/ERT-WFCCL-data/whisper-base/  # Whisper-Base 权重
├── TIM-Net_SER/Code/                  # TIM-Net 官方代码
├── LIGHT-SERNET/                      # Light-SERNet 官方代码
└── dataset/
    ├── CASIA/CASIA/<类别>/*.wav
    ├── SAVEE/wav/<类别>/*.wav
    └── RAVDESS/wav/<类别>/*.wav
```

外加 EMO-DB：`_local/datasets/emodb_by_class/<类别>/*.wav`。

## 与本仓库现有目录的对应关系

本仓库已收录两个模型的官方代码，映射如下：

| 代码期望路径 | 本仓库现有位置 |
| --- | --- |
| `TIM-Net_SER/Code/` | `5-TIM-Net/Code/` |
| `LIGHT-SERNET/` | `10-LIGHT-SERNET/` |

因此克隆后需要建软链接：

```bash
cd 11-CASIA-6Class-Benchmark
ln -s ../5-TIM-Net   TIM-Net_SER
ln -s ../10-LIGHT-SERNET LIGHT-SERNET
```

ERT-WFCCL 与 Whisper-Base 权重不在本仓库内，需自行获取：
- ERT-WFCCL 代码：论文作者仓库
- Whisper-Base：`openai/whisper-base`（HuggingFace），约 145 MB

## 数据集

四个语料均为公开数据，需自行下载并整理成「按类别分目录」的结构：

| 语料 | 类别目录 | 说明 |
| --- | --- | --- |
| EMO-DB | angry / boredom / disgust / fear / happy / neutral / sad | 7 类 |
| CASIA | angry / fear / happy / neutral / sad / surprise | 6 类，1200 条 |
| SAVEE | anger / disgust / fear / happiness / neutral / sadness / surprise | 7 类 |
| RAVDESS | angry / calm / disgust / fearful / happy / neutral / sad / surprised | 8 类 |

音频统一为 16 kHz 单声道 PCM_16。

## 环境

```bash
# PyTorch 侧（ERT-WFCCL + 聚合 + 渲染）
conda create -n torch-cu124 python=3.10
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install numpy scikit-learn librosa soundfile safetensors pytest

# TensorFlow 侧（Light-SERNet + TIM-Net）
conda create -n tf-gpu python=3.10
pip install tensorflow==2.17.1 tf-keras==2.17.0 numpy librosa soundfile pytest
```

TensorFlow 侧运行时**必须** `export TF_USE_LEGACY_KERAS=1`
（TIM-Net 的 `TIMNET` 层用了 `K.reverse`，Keras 3 不支持）。

## 只使用导出的模型

如果只想跑推理、不重训，无需上述依赖 —— 直接看
[`../exported_models/README.md`](../exported_models/README.md)。
