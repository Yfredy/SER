# 可部署模型文件

每个「模型 × 语料」一份，共 12 份。全部按 **说话人互斥协议 seed 0** 训练，
即模型在训练时**没有见过测试集的说话人** —— 对应真实部署场景。

## 标签空间

CASIA 口径 6 类，顺序固定：

```python
CLASSES = ["angry", "fear", "happy", "neutral", "sad", "surprise"]
```

模型输出为 6 维 softmax 概率，索引与上表对应。

**注意**：EMO-DB 语料本身没有 `surprise`，其模型该维度未被有效训练，
在 EMO-DB 上使用时应只取其余 5 类。

## 文件清单

| 模型 | 语料 | 文件 | 格式 | 体积 |
| --- | --- | --- | --- | --- |
| ERT-WFCCL | EMO-DB | `ERT-WFCCL_emodb.pt` | PyTorch state_dict | 697 KB |
| ERT-WFCCL | CASIA | `ERT-WFCCL_casia.pt` | PyTorch state_dict | 697 KB |
| ERT-WFCCL | SAVEE | `ERT-WFCCL_savee.pt` | PyTorch state_dict | 697 KB |
| ERT-WFCCL | RAVDESS | `ERT-WFCCL_ravdess.pt` | PyTorch state_dict | 697 KB |
| Light-SERNet | EMO-DB | `Light-SERNet_emodb_int8.tflite` | TFLite int8 | 480 KB |
| Light-SERNet | CASIA | `Light-SERNet_casia_int8.tflite` | TFLite int8 | 481 KB |
| Light-SERNet | SAVEE | `Light-SERNet_savee_int8.tflite` | TFLite int8 | 481 KB |
| Light-SERNet | RAVDESS | `Light-SERNet_ravdess_int8.tflite` | TFLite int8 | 481 KB |
| TIM-Net | EMO-DB | `TIM-Net_emodb.hdf5` | Keras weights | 763 KB |
| TIM-Net | CASIA | `TIM-Net_casia.hdf5` | Keras weights | 763 KB |
| TIM-Net | SAVEE | `TIM-Net_savee.hdf5` | Keras weights | 763 KB |
| TIM-Net | RAVDESS | `TIM-Net_ravdess.hdf5` | Keras weights | 767 KB |

`MANIFEST.json` 记录全部文件的元信息（类别、协议、seed、体积）。

## 使用方式

### Light-SERNet（TFLite int8，推荐端侧）

```python
import numpy as np, tensorflow as tf

interp = tf.lite.Interpreter(model_path="Light-SERNet_casia_int8.tflite")
interp.allocate_tensors()
inp, out = interp.get_input_details()[0], interp.get_output_details()[0]

# 输入：(1, 184, 40, 1) float32，16kHz 单声道、居中补齐/裁剪至 3.0s 的 MFCC
x = mfcc_features.astype(np.float32)[None, ..., None]
interp.set_tensor(inp["index"], x)
interp.invoke()
probs = interp.get_tensor(out["index"])[0]     # shape (6,)
label = CLASSES[int(np.argmax(probs))]
```

特征提取参考 `benchmark_ser/features.py` 的 `light_sernet_mfcc`。

### TIM-Net（Keras weights）

```python
import sys; sys.path.insert(0, "TIM-Net/Code")   # 或 5-TIM-Net/Code
from benchmark_ser.run_tf_tim import build_model

model = build_model(6)
model.load_weights("TIM-Net_casia.hdf5")

# 输入：(1, 188, 39) float32，librosa MFCC(39)，22.05kHz
probs = model.predict(x, verbose=0)[0]
```

需 `TF_USE_LEGACY_KERAS=1`（`TIMNET` 用了 `K.reverse`，Keras 3 不支持）。

### ERT-WFCCL（PyTorch state_dict）

```python
import torch, numpy as np

ck = torch.load("ERT-WFCCL_casia.pt", map_location="cpu", weights_only=False)

# 1) 先用 Whisper-Base 提取 512 维特征，再用存档的 scaler 标准化
mean  = np.array(ck["scaler_mean"],  dtype=np.float32)
scale = np.array(ck["scaler_scale"], dtype=np.float32)
x = (whisper_512_features - mean) / scale

# 2) 载入 MLP 塔并前向
# 模型结构见 benchmark_ser/run_torch.py -> ert_features.train_mlp 返回的网络
```

**重要**：`.pt` 里**不含** Whisper-Base 编码器（20,590,592 参数）。
部署时必须自行提供 Whisper 特征提取，或改用 Light-SERNet / TIM-Net。

## 精度参考

这些模型对应的评测结果见 `../docs/SER-ANALYSIS.md`。
其中说话人互斥协议下的 macro-F1（RAVDESS + EMO-DB 均值）为：

| 模型 | macro-F1 |
| --- | --- |
| ERT-WFCCL | 81.4 |
| Light-SERNet | 67.0 |
| TIM-Net | 55.9 |

注意：这是训练时的留出说话人结果，实际线上数据分布不同，需自行验证。
