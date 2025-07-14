# Training High Performance Spiking Neural Network by Temporal Model Calibration (TMC)
This repo contains source codes for the ICML 2025 paper Training High Performance Spiking Neural Network by Temporal Model Calibration. 

## Prerequisites
The Following Setup is tested and it is working:
 * Python>=3.5
 * Pytorch>=1.9.0
 * Cuda>=10.2

 ## Dataset Preparation for CIFAR10/100, ImageNet and N-Caltech101.
To proceed, please download the CIFAR10/100, ImageNet and N-Caltech101 datasets on your own.

## Dataset Preparation for DVS-CIFAR10
For CIFAR10-DVS dataset, please refer the Google Drive link below:
Training set: [1](https://drive.google.com/file/d/1pzYnhoUvtcQtxk_Qmy4d2VrhWhy5R-t9/view?usp=sharing)
Test set: [2](https://drive.google.com/file/d/1q1k6JJgVH3ZkHWMg2zPtrZak9jRP6ggG/view?usp=sharing)


## Cofe for TMC
We provide the code for CIFAR10/100, DVS-CIFAR10, N-Caltech101 and ImageNet. 

TMC training: execute `bash main.sh`.

## Code Reference
Our code is developed based on the code from [Shikuang Deng, Yuhang Li, Shanghang Zhang, and Shi Gu. Temporal efficient training of spiking neural network via gradient re-weighting. arXiv preprint arXiv:2202.11946, 2022.].

## Citation
Reference paper [TMC](https://openreview.net/pdf?id=l7ZmdeFyM1).

@inproceedings{yantraining,
  title={Training High Performance Spiking Neural Network by Temporal Model Calibration},
  author={Yan, Jiaqi and Wang, Changping and Ma, De and Tang, Huajin and Zheng, Qian and Pan, Gang},
  booktitle={Forty-second International Conference on Machine Learning}
}
