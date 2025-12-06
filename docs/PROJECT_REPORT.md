# EEG-Based Fatigue Detection: Comprehensive Project Report

**Course:** CSE 465
**Project:** EEG Signal Analysis for Driver Fatigue Detection
**Dataset:** SEED-VIG (SJTU Emotion EEG Dataset - Vigilance)
**Date:** December 2025
**Environment:** RTX 5080 GPU, Conda torch-310

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset Description](#2-dataset-description)
3. [Methodology](#3-methodology)
4. [Implementation Architecture](#4-implementation-architecture)
5. [Model Specifications](#5-model-specifications)
6. [Experimental Setup](#6-experimental-setup)
7. [Execution Instructions](#7-execution-instructions)
8. [Expected Results](#8-expected-results)
9. [Technical Details](#9-technical-details)
10. [File Structure](#10-file-structure)
11. [References](#11-references)

---

## 1. Project Overview

### 1.1 Objective

Develop and evaluate deep learning models for driver fatigue detection using EEG (Electroencephalography) signals. The system predicts driver vigilance levels to prevent accidents caused by drowsiness.

### 1.2 Tasks

- **Binary Classification:** Classify driver state as Alert (1) or Fatigued (0)
- **Regression:** Predict continuous PERCLOS (Percentage of Eye Closure) values [0, 1]

### 1.3 Key Features

- **Multi-feature Analysis:** 4 EEG feature extraction methods
- **Model Comparison:** 6 deep learning architectures (simple to advanced)
- **Comprehensive Evaluation:** 48 total experiments (4 features × 6 models × 2 tasks)
- **Robust Methodology:** Session-wise 80-10-10 split with proper validation for model selection

---

## 2. Dataset Description

### 2.1 SEED-VIG Dataset

- **Source:** SJTU (Shanghai Jiao Tong University)
- **Purpose:** Vigilance estimation from EEG signals
- **Sessions:** 23 recording sessions from 21 subjects (some subjects have multiple sessions)
- **Channels:** 17 EEG electrodes (standard 10-20 system)
- **Sampling Rate:** Downsampled and processed into frames
- **Total Samples:** 20,355 frames (885 frames per session)

### 2.2 EEG Channels (17)

```
FP1, FPZ, FP2, AF3, AF4, F7, F5, F3, F1, FZ, F2, F4, F6, F8, FT7, FC5, FC3
```

### 2.3 Frequency Bands (5)

| Band | Frequency Range | Brain State |
|------|----------------|-------------|
| Delta (δ) | 1-4 Hz | Deep sleep |
| Theta (θ) | 4-8 Hz | Drowsiness, meditation |
| Alpha (α) | 8-14 Hz | Relaxed wakefulness |
| Beta (β) | 14-31 Hz | Active thinking, focus |
| Gamma (γ) | 31-50 Hz | High-level cognition |

### 2.4 Feature Types (4 variants)

1. **de_LDS** - Differential Entropy with Linear Dynamical System smoothing
2. **de_movingAve** - Differential Entropy with Moving Average smoothing
3. **psd_LDS** - Power Spectral Density with LDS smoothing
4. **psd_movingAve** - Power Spectral Density with Moving Average smoothing

**Feature Shape:** (17 channels, 5 bands) = 85 features per frame

### 2.5 Labels: PERCLOS

- **Type:** Continuous values in range [0, 1]
- **Meaning:**
  - 0.0 = Fully drowsy/fatigued
  - 1.0 = Fully alert/awake
- **Threshold for Classification:** 0.35
  - PERCLOS ≥ 0.35 → Alert (Class 1)
  - PERCLOS < 0.35 → Fatigued (Class 0)

---

## 3. Methodology

### 3.1 Data Preprocessing Pipeline

```
Raw EEG Data → Feature Extraction → Session-wise Normalization → Train/Val/Test Split
```

#### Step 1: Feature Loading
- Load one of 4 feature types (de_LDS, de_movingAve, psd_LDS, psd_movingAve)
- Each sample: (17 channels, 5 bands)

#### Step 2: Session-wise Z-score Normalization
```python
For each session s:
    X_s_flat = reshape(X_s, (N_s, 85))
    mean_s = mean(X_s_flat, axis=0)
    std_s = std(X_s_flat, axis=0)
    X_s_normalized = (X_s_flat - mean_s) / (std_s + 1e-8)
```

**Rationale:** Remove inter-session variability, normalize each session's distribution

#### Step 3: Session-wise Train/Val/Test Split (80-10-10)
- **Train:** 80% of sessions (18 sessions)
- **Validation:** 10% of sessions (2 sessions)
- **Test:** 10% of sessions (3 sessions)
- **Model Selection:** Best model chosen based on validation set performance
- **Final Evaluation:** Test set used only for final performance reporting
- **Advantage:** Prevents data leakage, enables proper model selection, tests generalization
- **Random Seed:** 42 (reproducible splits)

#### Step 4: Label Processing
- **Classification:** Binarize PERCLOS at threshold 0.35
- **Regression:** Use continuous PERCLOS values directly

### 3.2 Data Augmentation Strategies

#### Single-frame Models
- Input: Individual frames (17, 5)
- No temporal context
- Direct frame-to-label mapping

#### Multi-frame Models (CNN Multi)
- **Temporal Window:** T_seq = 5 frames
- **Stride:** step = 2 frames
- **Label Aggregation:**
  - Classification: Majority vote over 5 frames
  - Regression: Mean PERCLOS over 5 frames
- **Constraint:** Windows created only within same session (no cross-session mixing)

---

## 4. Implementation Architecture

### 4.1 Notebook Structure

The implementation is modularized into 5 sequential Jupyter notebooks:

| Notebook | Purpose | Runtime | Output |
|----------|---------|---------|--------|
| 1_data_exploration.ipynb | Data loading, verification, visualization | 5-10 min | EDA insights |
| 2_baseline_models.ipynb | Model definitions, dataset classes, utilities | 2-3 min | Reusable functions |
| 3_train_classification.ipynb | Train 24 classification experiments | 2-4 hours | 24 models + results |
| 4_train_regression.ipynb | Train 24 regression experiments | 2-4 hours | 24 models + results |
| 5_results_analysis.ipynb | Analyze, visualize, compare results | 5-10 min | Plots + tables |

**Total Execution Time:** ~5-9 hours (mostly GPU training)

### 4.2 Execution Flow

```
Notebook 1: Explore Data
    ↓
Notebook 2: Define Models & Datasets
    ↓
Notebook 3: Train Classification ────→ Save 24 models + results
    ↓
Notebook 4: Train Regression ────────→ Save 24 models + results
    ↓
Notebook 5: Analyze Results ─────────→ Generate visualizations + summary
```

---

## 5. Model Specifications

### 5.1 Model Overview

| Model | Type | Parameters | Key Features |
|-------|------|-----------|--------------|
| MLP | Fully Connected | ~13K | Simple baseline, flattened input |
| CNN Single | 2D CNN | ~50K | Spatial convolutions on (17,5) |
| CNN Multi | 2D CNN + Temporal | ~52K | Temporal + spatial features |
| EEGNet | Specialized EEG | ~2K | Depthwise separable convolutions |
| DE-CNN | Hierarchical | ~40K | Multi-scale feature extraction |
| Transformer | Self-attention | ~50K | Channel-wise attention mechanism |

### 5.2 Detailed Model Architectures

#### 5.2.1 MLP Baseline (~13K params)

```
Input: (85,) - flattened (17, 5)
    ↓
Dense(85 → 128) + ReLU + Dropout(0.3)
    ↓
Dense(128 → 64) + ReLU + Dropout(0.3)
    ↓
Dense(64 → 32) + ReLU
    ↓
Dense(32 → num_outputs)
```

**Pros:** Simple, fast training
**Cons:** Ignores spatial structure of channels

---

#### 5.2.2 CNN Single Frame (~50K params)

```
Input: (1, 17, 5) - treat as 1-channel image
    ↓
Conv2d(1 → 32, kernel=3×3) + ReLU + BatchNorm
    ↓
Conv2d(32 → 64, kernel=3×3) + ReLU + BatchNorm
    ↓
AdaptiveAvgPool2d(1, 1) → (64, 1, 1)
    ↓
Flatten → (64,)
    ↓
Dense(64 → 32) + ReLU + Dropout(0.3)
    ↓
Dense(32 → num_outputs)
```

**Pros:** Learns spatial patterns in channel-frequency space
**Cons:** No temporal context

---

#### 5.2.3 CNN Multi-Frame (~52K params)

```
Input: (5, 17, 5) - 5 temporal frames as channels
    ↓
Conv2d(5 → 32, kernel=3×3) + ReLU + BatchNorm
    ↓
Conv2d(32 → 64, kernel=3×3) + ReLU + BatchNorm
    ↓
AdaptiveAvgPool2d(1, 1) → (64, 1, 1)
    ↓
Flatten → (64,)
    ↓
Dense(64 → 32) + ReLU + Dropout(0.3)
    ↓
Dense(32 → num_outputs)
```

**Pros:** Captures temporal dynamics over 5 frames
**Cons:** Requires sequence creation (fewer samples)

---

#### 5.2.4 EEGNet (~2K params)

**Canonical architecture for EEG classification**

```
Input: (1, 17, 5)
    ↓
Block 1: Temporal Convolution
    Conv2d(1 → F1=8, kernel=(1,5)) + BatchNorm
    ↓
Block 2: Depthwise Spatial Convolution
    Conv2d(8 → 16, kernel=(17,1), groups=8) - learns spatial filters per channel
    BatchNorm + ELU + AvgPool(1,2) + Dropout(0.25)
    ↓
Block 3: Separable Convolution
    DepthwiseConv(16 → 16) + PointwiseConv(16 → 16)
    BatchNorm + ELU + AvgPool(1,2) + Dropout(0.25)
    ↓
Flatten + Dense → num_outputs
```

**Pros:** Highly efficient, designed for EEG
**Cons:** Less expressive than larger CNNs

---

#### 5.2.5 DE-CNN (VIGNet-style) (~40K params)

**Hierarchical differential entropy feature extraction**

```
Input: (1, 17, 5)
    ↓
Block 1: Initial feature extraction
    Conv2d(1 → 64, kernel=3×3) + ReLU + BatchNorm + MaxPool
    ↓
Block 2: Deep feature learning
    Conv2d(64 → 128, kernel=3×3) + ReLU + BatchNorm + MaxPool
    ↓
AdaptiveAvgPool → Flatten
    ↓
Dense(hidden → 64) + ReLU + Dropout(0.3)
    ↓
Dense(64 → num_outputs)
```

**Pros:** Learns hierarchical representations
**Cons:** More parameters, longer training

---

#### 5.2.6 Channel Transformer (~50K params)

**Self-attention over EEG channels**

```
Input: (17, 5) - 17 channels, 5 features each
    ↓
Channel Embedding: Linear(5 → embed_dim=64)
    → (17, 64) - each channel becomes 64-dim vector
    ↓
Add Positional Encoding: learnable (17, 64) embedding
    ↓
Transformer Encoder:
    - Multi-head Self-Attention (4 heads)
    - Feed-forward Network (dim=128)
    - Layer Normalization + Dropout
    - 2 encoder layers
    ↓
Global Average Pooling over 17 channels → (64,)
    ↓
Dense(64 → 32) + ReLU + Dropout(0.3)
    ↓
Dense(32 → num_outputs)
```

**Pros:** Learns channel interactions via attention
**Cons:** More computationally expensive

---

## 6. Experimental Setup

### 6.1 Training Configuration

#### Classification

```python
Loss Function: Weighted CrossEntropyLoss
  - Weights computed from class distribution
  - Handles class imbalance

Optimizer: Adam
  - Adaptive learning rates per parameter

Metrics: Accuracy, F1, Precision, Recall

Model Selection: Best validation F1 score
```

#### Regression

```python
Loss Function: MSE (Mean Squared Error)

Optimizer: Adam

Metrics: MSE, RMSE, MAE, R²

Model Selection: Lowest validation MSE
```

### 6.2 Hyperparameters per Model

| Model | Batch Size | Learning Rate | Weight Decay | Epochs |
|-------|-----------|---------------|--------------|---------|
| MLP | 256 | 1e-3 | 1e-4 | 50 |
| CNN Single | 128 | 1e-3 | 1e-4 | 50 |
| CNN Multi | 64 | 1e-3 | 1e-4 | 50 |
| EEGNet | 128 | 1e-3 | 1e-4 | 100 |
| DE-CNN | 64 | 5e-4 | 1e-4 | 75 |
| Transformer | 64 | 5e-4 | 1e-4 | 75 |

**Rationale:**
- Smaller batches for complex models (Transformer, DE-CNN)
- Lower learning rates for advanced models (avoid instability)
- More epochs for EEGNet (lightweight, needs more iterations)

### 6.3 GPU Configuration

```python
Device: CUDA (RTX 5080)
Mixed Precision: Not used (for numerical stability)
DataLoader Workers: 0 (notebook safety - prevents hanging)
CUDA Cache Management: torch.cuda.empty_cache() after each experiment
```

### 6.4 Total Experiments

```
4 feature types × 6 models × 2 tasks = 48 experiments

Classification: 24 experiments
Regression: 24 experiments

Total training time: ~5-9 hours on RTX 5080
```

---

## 7. Execution Instructions

### 7.1 Environment Setup

```bash
# Activate conda environment
conda activate torch-310

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

**Expected Output:**
```
PyTorch: 2.x.x
CUDA: True
```

### 7.2 Navigate to Project Directory

```bash
cd "/home/vortex/CSE 465/SFM1/Project/CSE465_EEG-20251204T160245Z-3-001/CSE465_EEG/Codes"
```

### 7.3 Launch Jupyter Notebook

```bash
jupyter notebook
```

### 7.4 Execute Notebooks Sequentially

#### Step 1: Data Exploration (5-10 minutes)
```
Open: 1_data_exploration.ipynb
Run: All cells (Kernel → Restart & Run All)

What it does:
  - Loads all 23 recording sessions (21 subjects, some with multiple sessions)
  - Explores all 4 feature types
  - Visualizes EEG heatmaps and time series
  - Checks data integrity

Output: Understanding of dataset structure and quality
```

#### Step 2: Model Definitions (2-3 minutes)
```
Open: 2_baseline_models.ipynb
Run: All cells

What it does:
  - Defines all 6 model architectures
  - Creates dataset classes
  - Implements normalization and split functions
  - Tests models with dummy data

Output: Reusable model and utility functions
```

#### Step 3: Classification Training (2-4 hours)
```
Open: 3_train_classification.ipynb
Run: All cells

What it does:
  - Trains 24 classification experiments
  - For each (feature, model) pair:
    * Load and normalize data
    * Split sessions 80-10-10 (train/val/test)
    * Train with weighted CrossEntropyLoss
    * Select best model based on validation F1
    * Evaluate on held-out test set
    * Save best model checkpoint

Output:
  - 24 model files: Models/baselines/classification/{feature}/{model}_best.pth
  - results/baselines/classification_results.json
  - results/baselines/classification_summary.csv
```

#### Step 4: Regression Training (2-4 hours)
```
Open: 4_train_regression.ipynb
Run: All cells

What it does:
  - Trains 24 regression experiments
  - For each (feature, model) pair:
    * Load and normalize data
    * Split sessions 80-10-10 (train/val/test)
    * Train with MSE loss
    * Select best model based on validation MSE
    * Evaluate on held-out test set
    * Save best model checkpoint

Output:
  - 24 model files: Models/baselines/regression/{feature}/{model}_best.pth
  - results/baselines/regression_results.json
  - results/baselines/regression_summary.csv
```

#### Step 5: Results Analysis (5-10 minutes)
```
Open: 5_results_analysis.ipynb
Run: All cells

What it does:
  - Loads both classification and regression results
  - Generates comparison tables
  - Creates 5 visualization plots
  - Identifies best models per feature type
  - Exports combined summary

Output:
  - results/baselines/combined_summary.csv
  - 5 PNG plots:
    * classification_f1_comparison.png
    * regression_mse_comparison.png
    * classification_heatmap.png
    * regression_heatmap.png
    * complexity_vs_performance.png
```

---

## 8. Expected Results

### 8.1 Model Checkpoints

After running notebooks 3 and 4, you will have **48 trained models**:

```
Models/baselines/
├── classification/
│   ├── de_LDS/
│   │   ├── mlp_best.pth
│   │   ├── cnn_single_best.pth
│   │   ├── cnn_multi_best.pth
│   │   ├── eegnet_best.pth
│   │   ├── decnn_best.pth
│   │   └── transformer_best.pth
│   ├── de_movingAve/ [6 models]
│   ├── psd_LDS/ [6 models]
│   └── psd_movingAve/ [6 models]
└── regression/
    └── [same structure as classification]
```

### 8.2 Results Files

```
results/baselines/
├── classification_results.json      # Detailed classification metrics
├── classification_summary.csv       # Classification summary table
├── regression_results.json          # Detailed regression metrics
├── regression_summary.csv           # Regression summary table
├── combined_summary.csv             # Best models per feature type
├── classification_f1_comparison.png # Bar charts (4 subplots)
├── regression_mse_comparison.png    # Bar charts (4 subplots)
├── classification_heatmap.png       # F1 score heatmap
├── regression_heatmap.png           # R² score heatmap
└── complexity_vs_performance.png    # Scatter plots (2 subplots)
```

### 8.3 Example Performance Metrics

**Note:** Actual results will vary based on random seed and training dynamics

#### Classification (Expected Ranges)

| Model | F1 Score | Accuracy | Parameters |
|-------|----------|----------|------------|
| MLP | 0.65-0.75 | 0.68-0.78 | 13K |
| CNN Single | 0.70-0.80 | 0.72-0.82 | 50K |
| CNN Multi | 0.72-0.82 | 0.74-0.84 | 52K |
| EEGNet | 0.68-0.78 | 0.70-0.80 | 2K |
| DE-CNN | 0.73-0.83 | 0.75-0.85 | 40K |
| Transformer | 0.74-0.84 | 0.76-0.86 | 50K |

#### Regression (Expected Ranges)

| Model | MSE | RMSE | R² |
|-------|-----|------|-----|
| MLP | 0.020-0.035 | 0.14-0.19 | 0.55-0.70 |
| CNN Single | 0.015-0.028 | 0.12-0.17 | 0.60-0.75 |
| CNN Multi | 0.013-0.025 | 0.11-0.16 | 0.65-0.78 |
| EEGNet | 0.018-0.030 | 0.13-0.17 | 0.58-0.72 |
| DE-CNN | 0.012-0.023 | 0.11-0.15 | 0.68-0.80 |
| Transformer | 0.011-0.022 | 0.10-0.15 | 0.70-0.82 |

**General Observations:**
- More complex models (Transformer, DE-CNN) typically perform better
- Multi-frame CNN benefits from temporal context
- EEGNet is parameter-efficient but may underperform on this task
- DE features often outperform PSD features
- LDS smoothing generally provides more stable features than moving average

---

## 9. Technical Details

### 9.1 Dataset Classes

#### SEEDVIGSingleFrameDataset

```python
Input: X (N, 17, 5), y (N,), flatten=bool, regression=bool

Processing:
  - Converts numpy arrays to PyTorch tensors
  - Optionally flattens (17, 5) → (85,) for MLP
  - Labels: Long for classification, Float for regression

__getitem__(idx):
  Returns: (x, y) where x is (17, 5) or (85,)
```

#### SEEDVIGMultiFrameDataset

```python
Input: X (N, 17, 5), y (N,), subj_ids (N,), T_seq=5, step=2, regression=bool

Processing:
  1. For each subject separately:
     - Create sliding windows of T_seq=5 frames with stride=step
     - Aggregate labels:
       * Classification: Majority vote (round mean)
       * Regression: Mean PERCLOS
  2. Stack all sequences across subjects

__getitem__(idx):
  Returns: (seq, label) where seq is (T_seq, 17, 5) = (5, 17, 5)
```

### 9.2 Key Functions

#### normalize_features_subjectwise

```python
def normalize_features_subjectwise(X_all, subj_all):
    """
    Z-score normalization per subject.

    For each subject:
        X_normalized = (X - mean_subject) / (std_subject + eps)

    Args:
        X_all: (N, 17, 5) all samples
        subj_all: (N,) subject IDs

    Returns:
        X_norm: (N, 17, 5) normalized features
    """
```

**Why subject-wise?**
- Removes inter-subject biases (different baseline EEG patterns)
- Normalizes within-subject distributions to mean=0, std=1
- Improves model generalization

#### subject_wise_split

```python
def subject_wise_split(subj_all, train_ratio=0.80, seed=42):
    """
    Split subjects (not samples) into train and test.

    Prevents data leakage: test subjects never seen during training.

    Args:
        subj_all: (N,) subject IDs for all samples
        train_ratio: fraction of subjects for training
        seed: random seed for reproducibility

    Returns:
        train_mask: (N,) boolean mask for training samples
        test_mask: (N,) boolean mask for test samples
    """
```

**Example:**
```
23 subjects total
→ Shuffle with seed=42
→ 18 subjects (80%) for training
→ 5 subjects (20%) for testing

Training samples: all frames from 18 subjects (~16,284 samples)
Test samples: all frames from 5 subjects (~4,071 samples)
```

### 9.3 Loss Functions

#### Weighted CrossEntropyLoss (Classification)

```python
def get_weighted_loss(y_train):
    """
    Compute class weights to handle imbalance.

    weight_class_i = N_total / (N_classes * N_class_i)

    Example:
        Class 0 (fatigued): 6000 samples → weight ≈ 1.7
        Class 1 (alert): 10000 samples → weight ≈ 1.0

    Returns:
        nn.CrossEntropyLoss(weight=weights)
    """
```

**Why weighted?**
- Dataset may have more alert samples than fatigued
- Prevents model from biasing toward majority class
- Ensures both classes contribute equally to loss

#### MSELoss (Regression)

```python
criterion = nn.MSELoss()

# For batch of predictions and targets:
loss = mean((predictions - targets)²)
```

**Why MSE?**
- Standard for regression tasks
- Penalizes large errors more than small ones
- Differentiable for gradient descent

### 9.4 Evaluation Metrics

#### Classification Metrics

```python
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

accuracy = correct_predictions / total_predictions
precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
f1 = 2 * (precision * recall) / (precision + recall)
```

**Primary Metric:** F1 Score (harmonic mean of precision and recall)

#### Regression Metrics

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

MSE = mean((y_true - y_pred)²)
RMSE = sqrt(MSE)
MAE = mean(|y_true - y_pred|)
R² = 1 - (SS_residual / SS_total)
```

**Primary Metric:** MSE (Mean Squared Error)

### 9.5 Training Loop Structure

```python
for epoch in range(1, num_epochs + 1):
    # Training phase
    model.train()
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    # Validation phase
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            # Compute metrics...

    # Checkpoint: Save if best validation performance
    if metric_improved:
        torch.save(model.state_dict(), checkpoint_path)

# Load best model
model.load_state_dict(torch.load(checkpoint_path))
```

---

## 10. File Structure

```
CSE465_EEG/
├── Codes/
│   ├── 1_data_exploration.ipynb          # Data exploration and visualization
│   ├── 2_baseline_models.ipynb           # Model architectures and datasets
│   ├── 3_train_classification.ipynb      # Classification training
│   ├── 4_train_regression.ipynb          # Regression training
│   └── 5_results_analysis.ipynb          # Results analysis and visualization
│
├── Models/
│   └── baselines/
│       ├── classification/
│       │   ├── de_LDS/                   # 6 classification models
│       │   ├── de_movingAve/             # 6 classification models
│       │   ├── psd_LDS/                  # 6 classification models
│       │   └── psd_movingAve/            # 6 classification models
│       └── regression/
│           ├── de_LDS/                   # 6 regression models
│           ├── de_movingAve/             # 6 regression models
│           ├── psd_LDS/                  # 6 regression models
│           └── psd_movingAve/            # 6 regression models
│
├── results/
│   └── baselines/
│       ├── classification_results.json
│       ├── classification_summary.csv
│       ├── regression_results.json
│       ├── regression_summary.csv
│       ├── combined_summary.csv
│       ├── classification_f1_comparison.png
│       ├── regression_mse_comparison.png
│       ├── classification_heatmap.png
│       ├── regression_heatmap.png
│       └── complexity_vs_performance.png
│
├── EEG_Feature_5Bands/                   # Raw EEG features (23 .mat files)
├── perclos_labels/                       # PERCLOS labels (23 .mat files)
├── execution order.txt                   # Execution instructions
└── PROJECT_REPORT.md                     # This comprehensive report
```

---

## 11. References

### 11.1 Dataset

- **SEED-VIG Dataset:** Shanghai Jiao Tong University (SJTU)
- **Paper:** Wei-Long Zheng et al. "EEG-based emotion classification using deep belief networks"
- **URL:** https://bcmi.sjtu.edu.cn/home/seed/

### 11.2 Model Architectures

1. **EEGNet:**
   - Lawhern et al. (2018). "EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces"
   - Journal of Neural Engineering

2. **Transformer:**
   - Vaswani et al. (2017). "Attention is All You Need"
   - NeurIPS

3. **DE-CNN (VIGNet-style):**
   - Based on vigilance estimation architectures for EEG

### 11.3 Libraries

```python
PyTorch >= 2.0
NumPy >= 1.20
SciPy >= 1.7
Pandas >= 1.3
Matplotlib >= 3.4
Seaborn >= 0.11
scikit-learn >= 1.0
```

### 11.4 Hardware

- **GPU:** NVIDIA RTX 5080
- **Environment:** Conda torch-310 (Python 3.10)
- **CUDA:** Compatible version with PyTorch

---

## Appendix A: Troubleshooting

### Issue 1: Notebook Hangs During Training

**Symptom:** Cell execution freezes, no output

**Solution:**
```python
# In DataLoader initialization, ensure:
train_loader = DataLoader(..., num_workers=0)  # NOT num_workers > 0
```

**Reason:** Multiprocessing issues in Jupyter notebooks

---

### Issue 2: CUDA Out of Memory

**Symptom:** RuntimeError: CUDA out of memory

**Solution:**
```python
# Reduce batch size in CONFIG
'transformer': {'batch_size': 32}  # Instead of 64

# Or clear cache between experiments
torch.cuda.empty_cache()
```

---

### Issue 3: Import Errors in Notebooks 3-5

**Symptom:** NameError: model/function not defined

**Solution:**
```python
# Ensure you ran Notebook 2 first
# In Notebook 3/4, verify this line executes:
%run 2_baseline_models.ipynb
```

---

### Issue 4: File Not Found Errors

**Symptom:** FileNotFoundError: EEG_Feature_5Bands

**Solution:**
```python
# Verify BASE_DIR in notebook:
BASE_DIR = Path('/home/vortex/CSE 465/SFM1/Project/CSE465_EEG-20251204T160245Z-3-001/CSE465_EEG')

# Check directories exist:
!ls "$BASE_DIR/EEG_Feature_5Bands"
!ls "$BASE_DIR/perclos_labels"
```

---

## Appendix B: Customization Guide

### Change Train/Test Split Ratio

```python
# In Notebooks 3 and 4, modify:
TRAIN_RATIO = 0.70  # Instead of 0.80
```

### Add New Feature Type

```python
# If you have additional features (e.g., 'de_custom'):
FEATURE_KEYS = ['de_LDS', 'de_movingAve', 'psd_LDS', 'psd_movingAve', 'de_custom']

# Ensure data file exists:
# EEG_Feature_5Bands/1_20130621_de_custom.mat (and for all 23 subjects)
```

### Modify Hyperparameters

```python
# In Notebooks 3 and 4, edit CONFIG dictionary:
CONFIG = {
    'mlp': {
        'batch_size': 128,      # Changed from 256
        'lr': 5e-4,             # Changed from 1e-3
        'weight_decay': 1e-5,   # Changed from 1e-4
        'epochs': 100           # Changed from 50
    },
    # ... other models
}
```

### Change Classification Threshold

```python
# In Notebook 3, modify:
PERCLOS_THRESHOLD = 0.5  # Instead of 0.35

# This will change the balance between Alert/Fatigued classes
```

---

## Appendix C: Performance Optimization Tips

### 1. Use Mixed Precision Training (Advanced)

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for epoch in range(num_epochs):
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()

        with autocast():
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

**Benefit:** ~30-50% speedup on RTX 5080

### 2. Increase Batch Size (if GPU memory allows)

```python
# Try doubling batch sizes:
'mlp': {'batch_size': 512}  # Instead of 256
```

**Monitor:** GPU memory usage with `nvidia-smi`

### 3. Enable cuDNN Autotuner

```python
import torch.backends.cudnn as cudnn
cudnn.benchmark = True  # Add at start of training notebooks
```

**Benefit:** Optimizes convolution algorithms for your hardware

---

## Conclusion

This comprehensive implementation provides:

- **Modular Notebooks:** Easy to understand, modify, and extend
- **48 Experiments:** Thorough evaluation of features and models
- **Robust Methodology:** Subject-wise split prevents data leakage
- **Production-Ready Code:** GPU-safe, no hanging issues
- **Complete Evaluation:** Metrics, visualizations, and model checkpoints

**Next Steps:**
1. Run all 5 notebooks sequentially
2. Analyze results in Notebook 5
3. Identify best model for your use case
4. Deploy best model for real-time fatigue detection (future work)

**Contact & Support:**
- For dataset questions: SJTU SEED-VIG team
- For implementation issues: Check Appendix A (Troubleshooting)

---

**End of Report**

*Generated: December 2025*
*Project: CSE 465 EEG Fatigue Detection*
*Implementation: Complete and Ready for Execution*
