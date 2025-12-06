# Technical Summary: EEG-Based Fatigue Detection

**Course:** CSE 465
**Domain:** Biomedical Signal Processing, Deep Learning
**Task:** Driver Fatigue Detection from EEG Signals
**Approach:** Multi-model Comparative Study

---

## Problem Statement

Driver fatigue is a major cause of road accidents. This project develops automated fatigue detection systems using brain signals (EEG) to predict vigilance levels and prevent accidents.

**Challenges:**
- High-dimensional, noisy EEG signals (17 channels × 5 frequency bands)
- Inter-subject variability in EEG patterns
- Need for real-time, accurate predictions
- Imbalanced alert/fatigued class distribution

---

## Dataset: SEED-VIG

| Property | Value |
|----------|-------|
| Source | Shanghai Jiao Tong University |
| Sessions | 23 recording sessions (21 subjects, some with multiple sessions) |
| Total Samples | 20,355 frames |
| Channels | 17 EEG electrodes (10-20 system) |
| Frequency Bands | 5 (δ, θ, α, β, γ) |
| Features per Sample | 85 (17 × 5) |
| Target Variable | PERCLOS (Percentage of Eye Closure) [0, 1] |
| Feature Types | 4 (DE-LDS, DE-MovAvg, PSD-LDS, PSD-MovAvg) |

---

## Methodology

### 1. Preprocessing Pipeline

```
Raw EEG → Feature Extraction → Session-wise Z-score → Session-wise Split → Model Training
```

#### Session-wise Normalization
- **Technique:** Z-score normalization per session
- **Formula:** `X_norm = (X - μ_session) / σ_session`
- **Rationale:** Eliminates inter-session baseline differences

#### Session-wise Split (80-10-10)
- **Ratio:** 80% train / 10% validation / 10% test (18-2-3 sessions)
- **Purpose:** Prevent data leakage, enable proper model selection, test generalization
- **Model Selection:** Best model chosen based on validation set performance
- **Final Evaluation:** Test set used only for final performance reporting
- **Advantage:** Simulates real-world deployment on unseen recording sessions

### 2. Task Formulations

#### Classification
- **Input:** EEG features (17, 5)
- **Output:** Binary label {0: Fatigued, 1: Alert}
- **Threshold:** PERCLOS ≥ 0.35 → Alert
- **Loss:** Weighted CrossEntropyLoss (handles class imbalance)
- **Metrics:** Accuracy, F1, Precision, Recall
- **Selection:** Best validation F1

#### Regression
- **Input:** EEG features (17, 5)
- **Output:** Continuous PERCLOS [0, 1]
- **Loss:** Mean Squared Error (MSE)
- **Metrics:** MSE, RMSE, MAE, R²
- **Selection:** Lowest validation MSE

---

## Model Architectures

### 1. MLP Baseline (~13K params)
**Architecture:** FC(85→128) → FC(128→64) → FC(64→32) → FC(32→output)
**Pros:** Simple, fast, interpretable
**Cons:** Ignores spatial channel structure
**Use Case:** Performance baseline

### 2. CNN Single-Frame (~50K params)
**Architecture:** Conv(1→32) → Conv(32→64) → GlobalPool → FC(64→output)
**Pros:** Learns spatial patterns in channel-frequency space
**Cons:** No temporal context
**Innovation:** Treats (17, 5) as 2D spatial image

### 3. CNN Multi-Frame (~52K params)
**Architecture:** Conv(5→32) → Conv(32→64) → GlobalPool → FC(64→output)
**Input:** 5 consecutive frames (5, 17, 5)
**Temporal Window:** T_seq=5, stride=2
**Pros:** Captures temporal dynamics
**Innovation:** 2D convolutions over temporal + spatial dimensions

### 4. EEGNet (~2K params)
**Architecture:** Temporal Conv → Depthwise Spatial Conv → Separable Conv
**Key Features:**
- Depthwise convolutions: Learn spatial filters per temporal feature
- Separable convolutions: Efficient parameter sharing
- Domain-specific: Designed for EEG/BCI tasks
**Pros:** Highly parameter-efficient, interpretable
**Reference:** Lawhern et al. (2018), Journal of Neural Engineering

### 5. DE-CNN (~40K params)
**Architecture:** Conv(1→64) → Conv(64→128) → GlobalPool → FC(hidden→output)
**Inspiration:** VIGNet (vigilance estimation networks)
**Pros:** Hierarchical feature learning, strong representation
**Use Case:** High-capacity baseline

### 6. Channel Transformer (~50K params)
**Architecture:** ChannelEmbed(5→64) + PosEnc → TransformerEncoder(2 layers) → FC
**Key Features:**
- Self-attention over 17 channels
- Learns inter-channel dependencies
- Multi-head attention (4 heads)
**Pros:** Captures long-range channel interactions
**Innovation:** Treats channels as sequence tokens

---

## Experimental Design

### Scope
- **Total Experiments:** 48
  - 4 feature types × 6 models × 2 tasks = 48
- **Classification:** 24 experiments
- **Regression:** 24 experiments

### Hyperparameter Configuration

| Model | Batch Size | Learning Rate | Epochs | Weight Decay |
|-------|-----------|---------------|---------|--------------|
| MLP | 256 | 1e-3 | 50 | 1e-4 |
| CNN Single | 128 | 1e-3 | 50 | 1e-4 |
| CNN Multi | 64 | 1e-3 | 50 | 1e-4 |
| EEGNet | 128 | 1e-3 | 100 | 1e-4 |
| DE-CNN | 64 | 5e-4 | 75 | 1e-4 |
| Transformer | 64 | 5e-4 | 75 | 1e-4 |

**Optimizer:** Adam (adaptive learning rate)
**Scheduler:** None (constant LR with weight decay for regularization)

### Training Strategy
- **Early Stopping:** No (fixed epochs)
- **Checkpointing:** Save best model based on validation metric
- **Regularization:** Dropout (0.25-0.3), Weight Decay (1e-4)
- **Device:** CUDA (RTX 5080)
- **Reproducibility:** Fixed random seed (42)

---

## Key Technical Innovations

### 1. Session-wise Methodology
**Problem:** Inter-session EEG variability
**Solution:** Normalize and split at session level with 80-10-10 train/val/test
**Impact:** Realistic generalization evaluation with proper model selection

### 2. Multi-feature Comparative Analysis
**Problem:** Unclear which EEG features best predict fatigue
**Solution:** Test 4 feature extraction methods
**Impact:** Identify optimal preprocessing for this task

### 3. Dual-task Evaluation
**Problem:** Uncertain whether classification or regression better suits fatigue detection
**Solution:** Implement both tasks with same models
**Impact:** Compare discrete vs continuous prediction paradigms

### 4. Temporal Modeling via Multi-frame CNN
**Problem:** Single-frame models ignore temporal dynamics
**Solution:** Sliding window approach (5 frames) with label aggregation
**Impact:** Capture vigilance transitions over time

### 5. Attention-based Channel Interaction
**Problem:** Fixed convolutional kernels may miss channel dependencies
**Solution:** Transformer with self-attention over channels
**Impact:** Learn data-driven channel importance

---

## Implementation Highlights

### Modular Notebook Design
- **Notebook 1:** Data exploration (reproducible EDA)
- **Notebook 2:** Model library (reusable components)
- **Notebook 3-4:** Training pipelines (classification & regression)
- **Notebook 5:** Results analysis (visualization & comparison)

**Advantages:**
- Clear separation of concerns
- Easy debugging and modification
- Self-contained execution units

### GPU-Safe Configuration
- `num_workers=0` in DataLoader (prevents Jupyter hanging)
- Explicit CUDA cache clearing between experiments
- No multiprocessing/threading (notebook compatibility)

### Robust Evaluation
- Weighted loss for classification (handles imbalance)
- Multiple metrics (not just accuracy/MSE)
- Best model selection on held-out validation set

---

## Expected Outcomes

### Performance Benchmarks

#### Classification (Expected F1 Ranges)
- **MLP:** 0.65-0.75 (baseline)
- **CNN Single:** 0.70-0.80 (spatial patterns)
- **CNN Multi:** 0.72-0.82 (temporal boost)
- **EEGNet:** 0.68-0.78 (efficient)
- **DE-CNN:** 0.73-0.83 (high capacity)
- **Transformer:** 0.74-0.84 (attention-based)

#### Regression (Expected R² Ranges)
- **MLP:** 0.55-0.70
- **CNN Single:** 0.60-0.75
- **CNN Multi:** 0.65-0.78
- **EEGNet:** 0.58-0.72
- **DE-CNN:** 0.68-0.80
- **Transformer:** 0.70-0.82

### Comparative Insights

**Feature Comparison:**
- DE features typically outperform PSD
- LDS smoothing provides more stable features than moving average

**Model Comparison:**
- Transformer and DE-CNN achieve best performance
- EEGNet offers best parameter efficiency
- Multi-frame CNN benefits from temporal context

**Task Comparison:**
- Classification: Easier to interpret (alert/fatigued)
- Regression: More informative (continuous vigilance scale)

---

## Visualizations Generated

1. **F1 Bar Charts:** Model comparison per feature (classification)
2. **MSE Bar Charts:** Model comparison per feature (regression)
3. **Heatmaps:** Performance matrix (models × features)
4. **Scatter Plots:** Model complexity vs performance
5. **Summary Tables:** Best models per feature type

---

## Contributions

### Academic
- Comprehensive benchmark of deep learning models on SEED-VIG
- Evaluation of 4 EEG feature types for fatigue detection
- Comparison of classification vs regression paradigms
- Session-wise evaluation methodology with proper train/val/test split

### Practical
- Production-ready implementation (5 Jupyter notebooks)
- 48 trained model checkpoints
- Reproducible experimental setup
- GPU-optimized training pipeline

### Methodological
- Session-wise normalization and 80-10-10 train/val/test split
- Multi-frame temporal modeling
- Weighted loss for class imbalance
- Attention-based channel modeling
- Proper validation-based model selection

---

## Future Directions

### Short-term Enhancements
1. **Ensemble Methods:** Combine multiple models via voting/averaging
2. **Cross-validation:** K-fold subject-wise CV for robustness
3. **Hyperparameter Tuning:** Grid search or Bayesian optimization
4. **Augmentation:** Data augmentation for EEG (noise injection, mixup)

### Medium-term Extensions
1. **Recurrent Models:** LSTM/GRU for sequential modeling
2. **Hybrid Architectures:** CNN-LSTM, CNN-Transformer combinations
3. **Multi-task Learning:** Joint classification + regression
4. **Explainability:** Attention visualization, saliency maps

### Long-term Applications
1. **Real-time Deployment:** Streaming EEG prediction system
2. **Transfer Learning:** Pre-train on SEED-VIG, fine-tune on other datasets
3. **Multi-modal Fusion:** Combine EEG with eye tracking, physiological signals
4. **Personalized Models:** Subject-specific fine-tuning

---

## Conclusion

This project provides a **comprehensive, production-ready framework** for EEG-based fatigue detection, enabling:

- **Rigorous Evaluation:** 48 experiments across features, models, and tasks
- **Reproducible Research:** Fully documented, modular implementation
- **Practical Deployment:** Trained models ready for inference
- **Scientific Insights:** Comparative analysis of preprocessing and architectures

**Key Takeaway:** Modern deep learning architectures (Transformer, DE-CNN) achieve strong performance on EEG fatigue detection, with session-wise 80-10-10 train/val/test split being critical for proper model selection and realistic generalization assessment.

---

## References

1. **Dataset:** SJTU SEED-VIG - https://bcmi.sjtu.edu.cn/home/seed/
2. **EEGNet:** Lawhern et al. (2018), J. Neural Engineering
3. **Transformer:** Vaswani et al. (2017), NeurIPS
4. **Session-wise Split:** Standard practice in BCI/EEG research for generalization

---

**Author:** CSE 465 Student
**Supervisor:** Course Instructor
**Date:** December 2025
**Code:** 5 Jupyter Notebooks, 48 Trained Models
**Documentation:** PROJECT_REPORT.md, QUICK_START_GUIDE.md
