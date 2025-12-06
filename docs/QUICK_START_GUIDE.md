# Quick Start Guide - EEG Fatigue Detection

## Overview
This project implements EEG-based fatigue detection using 6 deep learning models across 4 feature types for both classification and regression tasks (48 total experiments).

---

## Prerequisites

```bash
# Activate conda environment
conda activate torch-310

# Navigate to project directory
cd "/home/vortex/CSE 465/SFM1/Project/CSE465_EEG-20251204T160245Z-3-001/CSE465_EEG/Codes"

# Launch Jupyter
jupyter notebook
```

---

## Execution Order (Sequential)

### 1. Data Exploration (~5 min)
```
Notebook: 1_data_exploration.ipynb
Action: Kernel → Restart & Run All

Purpose:
  ✓ Load and verify 23 sessions (21 subjects, some with multiple sessions)
  ✓ Explore 4 feature types (de_LDS, de_movingAve, psd_LDS, psd_movingAve)
  ✓ Visualize EEG patterns
  ✓ Check data integrity

Output: Understanding of dataset
```

---

### 2. Model Definitions (~2 min)
```
Notebook: 2_baseline_models.ipynb
Action: Kernel → Restart & Run All

Purpose:
  ✓ Define 6 models (MLP, CNN Single, CNN Multi, EEGNet, DE-CNN, Transformer)
  ✓ Create dataset classes
  ✓ Define preprocessing functions
  ✓ Test models with dummy data

Output: Reusable model definitions
```

---

### 3. Classification Training (~2-4 hours)
```
Notebook: 3_train_classification.ipynb
Action: Kernel → Restart & Run All

Purpose:
  ✓ Train 24 classification experiments (4 features × 6 models)
  ✓ Binary task: Alert (1) vs Fatigued (0)
  ✓ 80-10-10 split: Train/Validation/Test
  ✓ Metrics: Accuracy, F1, Precision, Recall
  ✓ Save best models based on validation F1

Output:
  - 24 model files: Models/baselines/classification/{feature}/{model}_best.pth
  - results/baselines/classification_results.json
  - results/baselines/classification_summary.csv
```

---

### 4. Regression Training (~2-4 hours)
```
Notebook: 4_train_regression.ipynb
Action: Kernel → Restart & Run All

Purpose:
  ✓ Train 24 regression experiments (4 features × 6 models)
  ✓ Predict continuous PERCLOS [0, 1]
  ✓ 80-10-10 split: Train/Validation/Test
  ✓ Metrics: MSE, RMSE, MAE, R²
  ✓ Save best models based on validation MSE

Output:
  - 24 model files: Models/baselines/regression/{feature}/{model}_best.pth
  - results/baselines/regression_results.json
  - results/baselines/regression_summary.csv
```

---

### 5. Results Analysis (~5 min)
```
Notebook: 5_results_analysis.ipynb
Action: Kernel → Restart & Run All

Purpose:
  ✓ Load all 48 experiment results
  ✓ Generate comparison tables
  ✓ Create visualizations
  ✓ Identify best models per feature type

Output:
  - results/baselines/combined_summary.csv
  - 5 visualization plots:
    * classification_f1_comparison.png
    * regression_mse_comparison.png
    * classification_heatmap.png
    * regression_heatmap.png
    * complexity_vs_performance.png
```

---

## Expected Runtime

| Stage | Notebook | Time |
|-------|----------|------|
| 1 | Data Exploration | 5-10 min |
| 2 | Model Definitions | 2-3 min |
| 3 | Classification Training | 2-4 hours |
| 4 | Regression Training | 2-4 hours |
| 5 | Results Analysis | 5-10 min |
| **Total** | | **~5-9 hours** |

---

## Key Features

### Feature Types (4)
- **de_LDS** - Differential Entropy with LDS smoothing
- **de_movingAve** - Differential Entropy with Moving Average
- **psd_LDS** - Power Spectral Density with LDS smoothing
- **psd_movingAve** - Power Spectral Density with Moving Average

### Models (6)
| Model | Parameters | Key Strength |
|-------|-----------|--------------|
| MLP | ~13K | Simple baseline |
| CNN Single | ~50K | Spatial patterns |
| CNN Multi | ~52K | Temporal + spatial |
| EEGNet | ~2K | EEG-specialized, efficient |
| DE-CNN | ~40K | Hierarchical features |
| Transformer | ~50K | Channel attention |

### Tasks (2)
- **Classification:** Alert vs Fatigued (threshold: PERCLOS ≥ 0.35)
- **Regression:** Continuous PERCLOS prediction [0, 1]

---

## Dataset Summary

- **Source:** SEED-VIG (SJTU)
- **Sessions:** 23 recording sessions (21 subjects, some with multiple sessions)
- **Samples:** 20,355 total (885 per session)
- **Channels:** 17 EEG electrodes
- **Bands:** 5 (Delta, Theta, Alpha, Beta, Gamma)
- **Input Shape:** (17 channels, 5 bands)
- **Split:** 80-10-10 train/validation/test (18-2-3 sessions)

---

## Final Outputs

### Models (48 total)
```
Models/baselines/
├── classification/ (24 models in 4 feature subdirectories)
└── regression/ (24 models in 4 feature subdirectories)
```

### Results
```
results/baselines/
├── classification_results.json
├── classification_summary.csv
├── classification_detailed.csv
├── regression_results.json
├── regression_summary.csv
├── regression_detailed.csv
├── combined_summary.csv
└── 5 PNG plots
```

---

## Troubleshooting

### Notebook hangs during training
**Fix:** Verify `num_workers=0` in DataLoader (already set in all notebooks)

### CUDA out of memory
**Fix:** Reduce batch size in CONFIG dictionary

### Import errors in Notebooks 3-5
**Fix:** Ensure Notebook 2 ran successfully first

### File not found errors
**Fix:** Verify BASE_DIR path matches your system

---

## Post-Execution

After running all notebooks:

1. **Check Results:** Open `results/baselines/combined_summary.csv`
2. **View Plots:** Browse visualization PNGs in `results/baselines/`
3. **Identify Best Model:** Check Notebook 5 output for best performers
4. **Load Trained Model:**

```python
import torch
from pathlib import Path

# Example: Load best classification model for de_LDS
model_path = Path('../Models/baselines/classification/de_LDS/transformer_best.pth')
model = ChannelTransformer(num_outputs=2)  # Classification
model.load_state_dict(torch.load(model_path))
model.eval()
```

---

## Quick Commands

```bash
# Check GPU
nvidia-smi

# Activate environment
conda activate torch-310

# Launch Jupyter
cd "/home/vortex/CSE 465/SFM1/Project/CSE465_EEG-20251204T160245Z-3-001/CSE465_EEG/Codes"
jupyter notebook

# Monitor GPU during training (in separate terminal)
watch -n 1 nvidia-smi
```

---

## Support

- **Full Documentation:** See `PROJECT_REPORT.md` for comprehensive details
- **Execution Order:** See `execution order.txt` for simple steps
- **Code Location:** All notebooks in `Codes/` directory

---

**Ready to run!** Start with Notebook 1 and proceed sequentially.
