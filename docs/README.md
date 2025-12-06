# EEG-Based Fatigue Detection using Deep Learning

A comprehensive implementation of driver fatigue detection from EEG signals using multiple deep learning architectures.

## Overview

This project implements and compares 6 deep learning models across 4 EEG feature types for both binary classification (alert vs fatigued) and continuous regression (PERCLOS prediction) tasks.

**Total Experiments:** 48 (4 features × 6 models × 2 tasks)

## Dataset

- **Source:** SEED-VIG (SJTU Emotion EEG Dataset - Vigilance)
- **Sessions:** 23 recording sessions (21 subjects, some with multiple sessions)
- **Samples:** 20,355 EEG frames
- **Channels:** 17 EEG electrodes
- **Features:** 5 frequency bands (Delta, Theta, Alpha, Beta, Gamma)
- **Target:** PERCLOS (Percentage of Eye Closure) [0, 1]

## Models

| Model | Parameters | Description |
|-------|-----------|-------------|
| **MLP** | ~13K | Fully connected baseline |
| **CNN Single** | ~50K | 2D CNN on single frames |
| **CNN Multi** | ~52K | 2D CNN on temporal sequences |
| **EEGNet** | ~2K | Specialized EEG architecture |
| **DE-CNN** | ~40K | Hierarchical feature extraction |
| **Transformer** | ~50K | Self-attention over channels |

## Features

### 4 EEG Feature Types
- `de_LDS` - Differential Entropy with LDS smoothing
- `de_movingAve` - Differential Entropy with Moving Average
- `psd_LDS` - Power Spectral Density with LDS smoothing
- `psd_movingAve` - Power Spectral Density with Moving Average

### 2 Tasks
- **Classification:** Binary prediction (Alert=1, Fatigued=0)
- **Regression:** Continuous PERCLOS prediction

## Project Structure

```
CSE465_EEG/
├── Codes/
│   ├── 1_data_exploration.ipynb       # EDA and visualization
│   ├── 2_baseline_models.ipynb        # Model architectures
│   ├── 3_train_classification.ipynb   # Classification training
│   ├── 4_train_regression.ipynb       # Regression training
│   └── 5_results_analysis.ipynb       # Results and plots
│
├── Models/baselines/
│   ├── classification/                # 24 classification models
│   └── regression/                    # 24 regression models
│
├── results/baselines/
│   ├── *_results.json                 # Detailed metrics
│   ├── *_summary.csv                  # Summary tables
│   └── *.png                          # Visualization plots
│
├── EEG_Feature_5Bands/                # EEG feature data
├── perclos_labels/                    # PERCLOS labels
│
├── README.md                          # This file
├── PROJECT_REPORT.md                  # Comprehensive documentation
├── QUICK_START_GUIDE.md              # Quick reference
├── TECHNICAL_SUMMARY.md              # Technical details
└── execution order.txt                # Simple execution guide
```

## Quick Start

### Prerequisites

```bash
# Activate conda environment
conda activate torch-310

# Required libraries
PyTorch >= 2.0
NumPy, SciPy, Pandas
Matplotlib, Seaborn
scikit-learn
```

### Execution

```bash
# Navigate to project directory
cd "/home/vortex/CSE 465/SFM1/Project/CSE465_EEG-20251204T160245Z-3-001/CSE465_EEG/Codes"

# Launch Jupyter
jupyter notebook
```

**Run notebooks sequentially:**
1. `1_data_exploration.ipynb` - Explore data (~5 min)
2. `2_baseline_models.ipynb` - Define models (~2 min)
3. `3_train_classification.ipynb` - Train classification (~2-4 hours)
4. `4_train_regression.ipynb` - Train regression (~2-4 hours)
5. `5_results_analysis.ipynb` - Analyze results (~5 min)

**Total runtime:** ~5-9 hours (RTX 5080 GPU)

## Methodology

### Preprocessing
1. **Feature Loading:** Load one of 4 feature types
2. **Normalization:** Session-wise Z-score normalization
3. **Split:** Session-wise 80-10-10 train/validation/test split (18-2-3 sessions)
4. **Label Processing:**
   - Classification: Threshold PERCLOS at 0.35
   - Regression: Use continuous PERCLOS values

### Training
- **Classification Loss:** Weighted CrossEntropyLoss
- **Regression Loss:** MSE
- **Optimizer:** Adam
- **Metrics:**
  - Classification: Accuracy, F1, Precision, Recall
  - Regression: MSE, RMSE, MAE, R²

### Evaluation
- Session-wise split prevents data leakage
- Model selection on validation set (10% of sessions)
- Final evaluation on held-out test set (10% of sessions)
- Comprehensive metrics and visualizations

## Results

After running all notebooks, you will have:

### Trained Models (48 total)
- 24 classification models in `Models/baselines/classification/`
- 24 regression models in `Models/baselines/regression/`

### Result Files
- `classification_results.json` - Detailed classification metrics
- `regression_results.json` - Detailed regression metrics
- `combined_summary.csv` - Best models per feature type
- 5 visualization plots (bar charts, heatmaps, scatter plots)

### Expected Performance

**Classification F1 Scores:** 0.65-0.84 (model dependent)
**Regression R² Scores:** 0.55-0.82 (model dependent)

Best performers typically: **Transformer** and **DE-CNN**

## Key Features

- **Modular Design:** 5 self-contained Jupyter notebooks
- **Reproducible:** Fixed random seeds, documented hyperparameters
- **GPU-Safe:** No multiprocessing issues in notebooks
- **Comprehensive:** 48 experiments with full evaluation
- **Production-Ready:** All models saved with checkpoints

## Documentation

- **PROJECT_REPORT.md** - Complete technical documentation (comprehensive)
- **QUICK_START_GUIDE.md** - Quick reference for execution
- **TECHNICAL_SUMMARY.md** - Academic/presentation summary
- **execution order.txt** - Simple step-by-step guide

## Hardware Requirements

- **GPU:** NVIDIA RTX 5080 (or equivalent CUDA-capable GPU)
- **Memory:** ~16GB RAM recommended
- **Storage:** ~2GB for models and results

## Citation

If you use this code or methodology, please cite:

```
SEED-VIG Dataset:
Wei-Long Zheng et al. "EEG-based emotion classification using deep belief networks"
SJTU Emotion EEG Dataset (SEED-VIG)
URL: https://bcmi.sjtu.edu.cn/home/seed/
```

## License

This project is for educational purposes (CSE 465 course project).

## Contact

For questions or issues, please refer to:
- **Full Documentation:** PROJECT_REPORT.md
- **Troubleshooting:** PROJECT_REPORT.md → Appendix A
- **Customization:** PROJECT_REPORT.md → Appendix B

## Acknowledgments

- **Dataset:** SJTU BCMI Lab for SEED-VIG dataset
- **EEGNet Architecture:** Lawhern et al. (2018)
- **Course:** CSE 465 - Signal Processing and Machine Learning

---

**Status:** ✅ Complete and Ready to Execute

**Last Updated:** December 2025
