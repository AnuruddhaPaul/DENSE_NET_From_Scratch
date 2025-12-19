# DenseNet-121 from Scratch on MNIST

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)

This repository contains a full implementation of the **DenseNet-121** architecture built from scratch using PyTorch. The model is specifically adapted to train on the **MNIST dataset** (grayscale handwritten digits).

## 📌 Project Overview

**DenseNet (Densely Connected Convolutional Networks)**, winner of the CVPR 2017 Best Paper Award, introduced a radical change to how layers connect. Instead of summing features (like ResNet), DenseNet **concatenates** the output of every layer to every subsequent layer.

This architecture encourages extreme feature reuse, strengthens gradient flow, and significantly reduces the number of parameters compared to ResNet.

### ⚙️ Adaptations for MNIST
The standard DenseNet is designed for ImageNet ($224 \times 224$). To make it work for MNIST, we applied the following changes:

1.  **Input Channels:** Modified the first convolution layer to accept **1 channel** (Grayscale) instead of 3 (RGB).
2.  **Image Resizing:** MNIST images ($28 \times 28$) are resized to **$32 \times 32$**.
    * *Reason:* DenseNet-121 uses 3 Transition Layers (pooling steps).
    * $32 \to 16 \to 8 \to 4$.
    * This ensures the feature maps remain large enough for the final classification layer.
3.  **Initial Convolution:** We use a $3 \times 3$ convolution with stride 1 (instead of $7 \times 7$ stride 2) to prevent immediate loss of spatial information on small images.

## 🏗️ Architecture Details

### The Core Concept: Concatenation

Unlike ResNet ($x + f(x)$), DenseNet concatenates features:
$$x_{\ell} = H_{\ell}([x_0, x_1, ..., x_{\ell-1}])$$

### Key Components
1.  **Dense Block:** A stack of layers where the feature maps size remains constant, but the number of channels increases by $k$ (Growth Rate) at every step.
2.  **Growth Rate ($k$):** The number of filters added by each layer (e.g., $k=32$). This keeps the network "narrow" compared to wide ResNets.
3.  **Transition Layer:** Located between Dense Blocks. It performs **$1 \times 1$ convolution** (to reduce channel count) and **$2 \times 2$ Average Pooling** (to reduce spatial size).

### Network Configuration (DenseNet-121)
| Block | Layers |
| :--- | :--- |
| **Dense Block 1** | 6 Layers |
| **Transition Layer** | (Reduces channels & size $/2$) |
| **Dense Block 2** | 12 Layers |
| **Transition Layer** | (Reduces channels & size $/2$) |
| **Dense Block 3** | 24 Layers |
| **Transition Layer** | (Reduces channels & size $/2$) |
| **Dense Block 4** | 16 Layers |
| **Classification** | Global Avg Pool $\to$ FC |

## 🚀 Getting Started

### Prerequisites
* Python 3.x
* PyTorch
* Torchvision

### Installation
Clone the repository:
```bash
git clone [https://github.com/your-username/densenet-mnist-scratch.git](https://github.com/your-username/densenet-mnist-scratch.git)
cd densenet-mnist-scratch
pip install torch torchvision
