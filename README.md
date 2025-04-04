# Speech Emotion Recognition

[miniconda macos安装教程](https://www.anaconda.com/docs/getting-started/miniconda/install#quickstart-install-instructions)

## SER最新项目
- SVM: 该方法使用不同的SVM核(线性、RBF和多项式)和基于话语的特征，比较了12种模型的准确率。结果表明，使用多项式核和基于话语的Fbank特征的SVM模型平均准确率最高。
- STRL-SER: 该方法分别在框架层和话语层建立深度情感学习模块，获得框架层的时空特征表征和话语层的全局特征表征。它还使用了多头注意机制来融合特征。
- AMSNet: 该方法使用多尺度SER网络AMSNet融合帧级特征和话语级特征。提出了一种基于连接注意机制的融合方法，增加了不同特征之间的互补性。
- Wav2vec2.0+MTL: 该方法基于预训练好的Wav2vec2.0建立多任务SER框架，以SER为主要任务，ASR为辅助任务，从音频中获取更多的情感信息。
- TIM-Net: 本文使用的TIM-Net是一种时间情绪建模方法。该方法可以通过双向时间建模捕获远程时间依赖关系，并动态融合多尺度信息，更好地适应时间尺度的变化。
- MFCC+谱图+Wav2vec2.0: 该方法利用MFCC、谱图和Wav2vec2.0提取的特征，并使用共同的注意机制从原始音频中提取互补的声学信息，用于SER。
- 自注意+MTL: 该方法首先从语音谱图中提取特征，然后利用自注意机制进一步获得情感特征。最后，通过MTL将性别分类作为辅助任务，提高了SER绩效。
- MMER: 该方法利用了多种模态，包括音频、文本、增强音频和增强文本。它还采用了三个辅助任务:基于监督对比学习的情绪识别、语音识别和基于增强对比学习的情绪识别。
- MTL-SER: SR+ASR+SER, 该方法结合了三种模态，包括音频、文本和增强音频。它还采用了三个辅助任务:基于监督对比学习的情绪识别、语音识别和基于增强对比学习的情绪识别。

![模型对比](/pic/模型对比.png "不同模型对比")

| Method                             | UA (%)          | WA (%)          | FLOPs(G) | Inference time (s) | p-value   |
|------------------------------------|-----------------|-----------------|----------|--------------------|-----------|
| SVM (Kurpukdee et al., 2017)       | --              | 58.40 ± 0.40    | 0.02     | 0.02               | 4.52E–13  |
| STRL-SER (Chen, Lin et al., 2023)  | 79.32 ± 0.50    | 81.60 ± 0.85    | 12.00    | 0.34               | 1.78E–03  |
| AMSNet (Chen, Li et al., 2023)     | 70.51 ± 0.65    | 69.22 ± 0.70    | 10.75    | 0.27               | 7.27E–11  |
| Wav2vec2.0+MTL (Cai et al., 2021)  | --              | 78.15 ± 0.60    | 17.31    | 0.55               | 4.59E–07  |
| TIM-Net (Ye et al., 2023)          | 72.50 ± 0.80    | 71.65 ± 0.78    | 71.57    | 15.50              | 3.89E–10  |
| MFCC+Spectrogram+wav2vec2.0 (Zou et al., 2022) | 72.95 ± 0.70 | 71.64 ± 0.65 | 45.09 | 0.35 | 3.86E–10 |
| Self-attention+MTL (Li et al., 2019)| 80.60 ± 0.60    | 80.82 ± 0.55    | 26.10    | 0.96               | 1.46E–04  |
| MMER (Ghosh et al., 2023)          | 76.20 ± 0.65    | 78.90 ± 0.70    | 138.77   | 1.60               | 1.70E–06  |
| **MTL-SER**                | **82.19 ± 0.75**| **82.63 ± 0.77**| 20.80    | 0.59               | --        |

## LIGHT-SERNET
传统语音情绪识别模型需要大量的计算和存储资源，难以与嵌入式系统中的其他机器交互任务同时实现，因此本文提出了只用CNN进行语音情绪识别任务的模型。
论文创新点：本文提出了一种轻量级的全卷积神经网络(FCNN)来进行语音情绪识别，FCNN使用具有不同滤波器大小的三个并行路径提取各种特征图。
![light-ser模型](/pic/light-ser.png "light-ser模型性能")
Size表示模型大小，WA表示加权准确率，UA表示普通准确率,WF1表示加权F1 score