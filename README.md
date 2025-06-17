
# 📦 CIFAR-10 圖像分類模型（PyTorch）

本專案使用 PyTorch 建構卷積神經網路（CNN），用於 CIFAR-10 圖像分類任務。包含資料預處理、模型設計、訓練、測試及預測結果可視化。

This project implements a Convolutional Neural Network (CNN) for image classification on the CIFAR-10 dataset using PyTorch. It includes data preprocessing, model architecture, training/testing pipeline, and prediction visualization.

---

## 📂 專案架構 Project Structure

```
cnn-cifar10-classification/
├── main.py              # 主程式，含模型建立、訓練、測試與預測
├── README.md            # 說明文件
└── requirements.txt     # 套件需求
```

---

## 🧠 模型簡介 Model Overview

- **資料集 Dataset**：CIFAR-10（10 類彩色圖片）
- **資料增強 Data Augmentation**：隨機水平翻轉、旋轉、亮度/對比調整
- **模型架構 Architecture**：
  - 5 層卷積層 (Conv2D + BatchNorm + ReLU + MaxPool)
  - Dropout
  - 兩層全連接層 (Linear)
- **損失函數 Loss**：CrossEntropyLoss
- **優化器 Optimizer**：AdamW
- **學習率調整 Learning Rate Scheduler**：ReduceLROnPlateau

---

## 🔧 執行方式 How to Run

1. 安裝必要套件：
   ```bash
   pip install -r requirements.txt
   ```

2. 執行主程式：
   ```bash
   python main.py
   ```

> ✅ 若有可用 GPU（CUDA），程式會自動使用；否則會回退至 CPU 執行。

---

## 📊 預測結果可視化 Visualization

模型訓練完成後，會自動顯示 10 張測試圖片的預測結果與正確標籤，協助評估模型表現。

---

## ✅ 成果摘要 Summary

- 使用 PyTorch 成功訓練 CNN 模型於 CIFAR-10 資料集。
- 損失函數穩定下降，準確率逐漸提升。
- 預測結果透過 `matplotlib` 圖像輸出可視化，輔助模型評估與展示。

---

## 📦 相依套件 Requirements

```
torch
torchvision
matplotlib
numpy
```

---

## 🔖 作者 Author

余峻廷（Jun-Ting Yu）  
轉職 AI 領域中，積極投入深度學習應用實作與職能提升。

---

## 📌 備註 Note

此專案為學習與實作用途，若您有任何建議或改進方向，歡迎提出 Issue 或 Pull Request。
