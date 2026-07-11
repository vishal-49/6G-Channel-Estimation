# 📡 A Reconfigurable Framework and Benchmark for Deep Learning-Based 6G Channel Estimation

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B)
![License](https://img.shields.io/badge/License-MIT-green)

### 🚀 Deep Learning Framework for 6G Wireless Channel Estimation

An end-to-end deep learning framework for accurate 6G channel estimation using **RDMSNet**, complete benchmarking, and an interactive Streamlit web application.

🌐 **Live Demo:** https://6g-channel-estimation-g8s93qkzeqgbjbj6tr2s63.streamlit.app

<br>

<div align="center">

| 🚀 Framework | 🤖 Models | 📊 Validation | 🌐 Deployment |
|:-----------:|:---------:|:-------------:|:-------------:|
| **RDMSNet** | **7 Deep Learning Models** | **5-Fold Cross Validation** | **Streamlit Web App** |

</div>

---


</div>

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Motivation](#-motivation)
- [Features](#-features)
- [Framework Architecture](#-framework-architecture)
- [Dataset](#-dataset)
- [Benchmark Models](#-benchmark-models)
- [Results](#-results)
- [Web Application](#-web-application)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Technologies Used](#-technologies-used)
- [Future Work](#-future-work)
- [Author](#-author)

---

# 📖 Overview

Accurate **Channel State Information (CSI)** is one of the fundamental requirements for reliable and high-speed communication in next-generation **6G wireless networks**. As wireless environments become increasingly dynamic due to high mobility, multipath propagation, Doppler effects, and complex channel conditions, conventional channel estimation techniques often struggle to maintain both estimation accuracy and computational efficiency.

This project presents a **reconfigurable deep learning framework** designed for benchmarking and evaluating multiple neural network architectures for 6G channel estimation. The framework provides a unified experimental environment that enables fair comparison of different models using standardized datasets, identical training strategies, and common evaluation metrics.

The primary contribution of this work is **RDMSNet (Residual Dense Multi-Scale Network)**, a lightweight yet powerful architecture that integrates Residual Dense Blocks, Squeeze-and-Excitation (SE) Attention, Atrous Spatial Pyramid Pooling (ASPP), and Multi-Scale Feature Fusion to effectively capture complex wireless channel characteristics.

To ensure reliable performance evaluation, the framework incorporates **5-Fold Cross Validation** and reports multiple evaluation metrics including **NMSE, RMSE, MSE, PSNR, and R² Score**. Additionally, an interactive **Streamlit Web Application** is provided for real-time inference, making the framework useful for both research and practical demonstrations.

---

# 🎯 Problem Statement

Future **6G wireless communication systems** demand highly accurate and efficient channel estimation to support ultra-high data rates, ultra-low latency, massive connectivity, and intelligent communication services. However, accurately estimating wireless channels remains a significant challenge due to rapidly varying propagation environments, severe multipath fading, Doppler shifts, noise, and increasing system complexity.

Conventional channel estimation techniques, such as **Least Squares (LS)** and **Minimum Mean Square Error (MMSE)**, often suffer from limited estimation accuracy under highly dynamic channel conditions and may require significant computational resources or prior statistical information. These limitations make them less suitable for next-generation wireless networks.

Although recent deep learning-based approaches have shown promising improvements, many existing models focus on a single architecture and lack a unified framework for systematic comparison, benchmarking, and reproducible experimentation.

This project addresses these challenges by developing a **reconfigurable deep learning framework** capable of benchmarking multiple architectures under identical experimental conditions while proposing **RDMSNet**, an optimized model designed to improve channel estimation accuracy and robustness for 6G communication systems.

---

# 💡 Motivation

The rapid evolution of wireless communication technologies has created a growing demand for intelligent and adaptive channel estimation techniques. As 6G networks aim to support applications such as autonomous vehicles, holographic communication, extended reality (XR), and massive Internet of Things (IoT), traditional estimation methods become increasingly inadequate due to their limited ability to model complex and nonlinear wireless channels.

The motivation behind this project is to build a **flexible, modular, and research-oriented framework** that enables fair comparison of multiple deep learning architectures under a unified experimental setup. Rather than evaluating a single model, this framework allows researchers to benchmark different approaches using the same dataset, preprocessing pipeline, training strategy, and evaluation metrics.

Furthermore, this work introduces **RDMSNet**, a lightweight architecture designed to improve estimation accuracy while maintaining computational efficiency. The framework is intended to accelerate research, simplify experimentation, and provide a solid baseline for future deep learning-based 6G channel estimation studies.

---

# ✨ Features

This framework is designed to provide a complete research and benchmarking environment for **Deep Learning-Based 6G Channel Estimation**.

<table>
<tr>
<td width="50%">

### 🧠 Deep Learning Models
- RDMSNet (Proposed)
- CNN Baseline
- RCNN
- U-Net
- Attention-Based Models
- Multi-Scale CNN
- Benchmark Architecture Support

</td>

<td width="50%">

### 📊 Performance Evaluation
- 5-Fold Cross Validation
- NMSE
- RMSE
- MSE
- PSNR
- R² Score

</td>
</tr>

<tr>
<td>

### ⚙️ Framework
- Modular Code Structure
- Easy Model Training
- Easy Testing
- Benchmark Pipeline
- Reconfigurable Architecture

</td>

<td>

### 🌐 Deployment
- Streamlit Web Application
- Interactive Prediction
- GPU Support (PyTorch)
- HDF5 Dataset Support
- Research Friendly

</td>
</tr>
</table>

---

# 🏗️ Framework Architecture

The proposed framework follows a modular deep learning pipeline designed specifically for **6G channel estimation**. Each stage is independently configurable, allowing researchers to benchmark multiple architectures while maintaining a consistent training and evaluation pipeline.

<p align="center">

> 📌 **Architecture Diagram will be added here**

<!-- Replace with your image later -->

<img src="assets/architecture.png" width="95%">

</p>

## 🔄 Workflow

```text
Input CSI
     │
     ▼
Data Preprocessing
     │
     ▼
Feature Extraction
     │
     ▼
Residual Dense Learning
     │
     ▼
SE Attention
     │
     ▼
ASPP Module
     │
     ▼
Multi-Scale Feature Fusion
     │
     ▼
Channel Reconstruction
     │
     ▼
Performance Evaluation
```

## 🧩 Key Components

| Component | Purpose |
|-----------|---------|
| **Input Layer** | Receives complex-valued CSI data from the dataset. |
| **Feature Extraction** | Learns low-level spatial representations from wireless channels. |
| **Residual Dense Blocks** | Improves feature reuse and gradient propagation. |
| **SE Attention Module** | Enhances informative channel features while suppressing irrelevant information. |
| **ASPP Module** | Captures multi-scale contextual information using different dilation rates. |
| **Multi-Scale Fusion** | Combines features extracted at different receptive fields. |
| **Output Layer** | Reconstructs the estimated wireless channel. |

---
