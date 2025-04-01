# Speech Emotion Recognition

SER最新项目

[miniconda macos安装教程](https://www.anaconda.com/docs/getting-started/miniconda/install#quickstart-install-instructions)

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

## MTL-SER



## Wav2vec2

## TIM-NET

## CNN+LSTM

## SVM

## MLP

## Emo-Box

## 

