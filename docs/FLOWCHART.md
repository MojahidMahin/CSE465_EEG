# EEG-Based Driver Fatigue Detection - Notebooks 2-4 Flow Chart

## Overview
This document provides a complete visual flowchart of how data flows through Notebooks 2, 3, and 4 of the EEG classification/regression pipeline.

---

## NOTEBOOK 2: Baseline Models & Dataset Classes (Foundation Layer)

### Purpose
Define all reusable components: data loaders, datasets, and model architectures.

```
NOTEBOOK 2 STRUCTURE
════════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. IMPORTS & SETUP                                                          │
│ ├─ PyTorch, NumPy, SciPy                                                    │
│ └─ Check GPU availability & CUDA version                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. DATA LOADING FUNCTION                                                    │
│ ├─ load_seedvig_5band_all_features(base_dir, feature_key)                  │
│ │  ├─ Load 23 .mat files from EEG_Feature_5Bands/                          │
│ │  ├─ Extract 4 feature types:                                             │
│ │  │  ├─ de_LDS (Differential Entropy + LDS)                              │
│ │  │  ├─ de_movingAve (Differential Entropy + Moving Avg)                 │
│ │  │  ├─ psd_LDS (Power Spectral Density + LDS)                           │
│ │  │  └─ psd_movingAve (PSD + Moving Avg)                                 │
│ │  ├─ Transpose from (17, 885, 5) → (885, 17, 5)                          │
│ │  └─ Load PERCLOS labels (0-1 continuous values)                          │
│ │                                                                            │
│ └─ OUTPUTS: X_all (N, 17, 5), y_all (N,), subj_all (N,)                   │
│              N = total EEG frames across all 23 sessions                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. NORMALIZATION FUNCTION                                                   │
│ ├─ normalize_features_subjectwise(X_all, subj_all)                         │
│ │  ├─ FOR EACH SESSION (subject):                                          │
│ │  │  ├─ Get all frames from that session                                  │
│ │  │  ├─ Apply Z-score normalization: (x - mean) / std                     │
│ │  │  └─ Handle numerical stability (add 1e-8 to std)                      │
│ │  └─ CRITICAL: Session-wise normalization handles inter-session variability│
│ │                                                                            │
│ └─ OUTPUT: X_norm (N, 17, 5) - normalized per session                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. TRAIN/VALIDATION/TEST SPLIT FUNCTION                                    │
│ ├─ subject_wise_split(subj_all, train_ratio=0.80, val_ratio=0.10)         │
│ │  ├─ Get unique sessions (1-23 sessions total)                            │
│ │  ├─ Shuffle sessions with seed=42                                        │
│ │  ├─ Split into:                                                          │
│ │  │  ├─ Train: 18 sessions (80%)                                          │
│ │  │  ├─ Validation: 2 sessions (10%)                                      │
│ │  │  └─ Test: 3 sessions (10%)                                            │
│ │  ├─ Create boolean masks for each set                                    │
│ │  └─ CRITICAL: Entire sessions go to single split (no data leakage)       │
│ │                                                                            │
│ └─ OUTPUTS: train_mask, val_mask, test_mask (all boolean arrays)           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ 5. PYTORCH DATASET CLASSES                                                  │
│                                                                              │
│ A) SEEDVIGSingleFrameDataset                                                │
│    ├─ For models: MLP, CNN Single, EEGNet, DE-CNN, Transformer             │
│    ├─ Input: X (N, 17, 5), y (N,)                                          │
│    ├─ __init__: Store as torch tensors                                     │
│    ├─ __getitem__: Return single frame (17, 5) or flattened (85,)          │
│    └─ Handles: flatten for MLP, regression flag                            │
│                                                                              │
│ B) SEEDVIGMultiFrameDataset                                                 │
│    ├─ For model: CNN Multi only                                            │
│    ├─ Input: X (N, 17, 5), y (N,), subj_ids (N,)                           │
│    ├─ __init__:                                                            │
│    │  ├─ FOR EACH SESSION:                                                 │
│    │  │  ├─ Create sliding windows (T_seq=5, stride=2)                     │
│    │  │  ├─ Each window: (5, 17, 5) temporal sequence                      │
│    │  │  ├─ Label: majority vote (classification) / mean (regression)      │
│    │  │  └─ Aggregate within session only                                  │
│    │  └─ Store all sequences                                               │
│    └─ __getitem__: Return (T_seq, 17, 5) and aggregated label              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ 6. MODEL ARCHITECTURES (6 Models Total)                                    │
│                                                                              │
│ Model 1: MLPBaseline (13K params)                                           │
│   Input:  (batch, 85) flattened                                            │
│   ├─ FC(85 → 128) + ReLU + Dropout                                          │
│   ├─ FC(128 → 64) + ReLU + Dropout                                          │
│   ├─ FC(64 → 32) + ReLU                                                     │
│   └─ FC(32 → num_outputs)                                                   │
│                                                                              │
│ Model 2: CNN2DSingleFrame (23K params)                                      │
│   Input:  (batch, 17, 5) → unsqueeze → (batch, 1, 17, 5)                  │
│   ├─ Conv2d(1, 32, 3×3) + BN + ReLU + MaxPool2d((2,1)) → (32, 8, 5)       │
│   ├─ Conv2d(32, 64, 3×3) + BN + ReLU + AdaptiveAvgPool2d → (64, 1, 1)     │
│   ├─ Flatten + FC(64 → 64) + ReLU + Dropout                                │
│   └─ FC(64 → num_outputs)                                                   │
│                                                                              │
│ Model 3: CNN2DMultiFrame (101K params)                                      │
│   Input:  (batch, T=5, 17, 5)                                              │
│   ├─ Spatial Processing:                                                   │
│   │  ├─ Reshape to (B*T, 1, 17, 5)                                         │
│   │  ├─ Apply spatial CNN on each frame independently                      │
│   │  └─ Output per frame: (64,) feature vector                             │
│   ├─ Temporal Processing:                                                  │
│   │  ├─ Stack frame features: (B, T, 64)                                   │
│   │  ├─ Transpose to (B, 64, T)                                            │
│   │  ├─ Conv1d(64, 128) temporal aggregation                               │
│   │  └─ Global temporal pooling                                            │
│   └─ Classification Head: FC layers to num_outputs                          │
│                                                                              │
│ Model 4: EEGNet (730 params) - Domain-specific, ultra-compact               │
│   Input:  (batch, 17, 5) → (batch, 1, 17, 5)                              │
│   ├─ Block 1: Conv2d(1, F1=8, kernel=(1,5))                                │
│   ├─ Block 2: Depthwise Conv2d(8, 16, kernel=(17,1)) + AvgPool + Dropout   │
│   ├─ Block 3: Separable Conv (F1*D → F2=16)                                │
│   └─ Linear(F2 → num_outputs)                                               │
│                                                                              │
│ Model 5: DECNN (118K params) - Deep hierarchical CNN                       │
│   Input:  (batch, 17, 5) → (batch, 1, 17, 5)                              │
│   ├─ Conv2d(1, 32) + BN + ReLU + MaxPool2d((2,1))                          │
│   ├─ Conv2d(32, 64) + BN + ReLU + MaxPool2d((2,1))                         │
│   ├─ Conv2d(64, 128) + BN + ReLU + AdaptiveAvgPool2d                       │
│   ├─ Deep Classification Head:                                             │
│   │  ├─ FC(128 → 128) + ReLU + Dropout                                     │
│   │  ├─ FC(128 → 64) + ReLU + Dropout                                      │
│   │  └─ FC(64 → num_outputs)                                               │
│                                                                              │
│ Model 6: ChannelTransformer (70K params) - Attention-based                  │
│   Input:  (batch, 17, 5)                                                   │
│   ├─ Channel Embedding: 5 bands → 64 dims linear layer                     │
│   │  Output: (batch, 17, 64)                                               │
│   ├─ Positional Encoding: Add learnable (1, 17, 64)                        │
│   ├─ Transformer Encoder:                                                  │
│   │  ├─ Self-attention learns inter-channel relationships                  │
│   │  ├─ num_heads=4, d_model=64                                            │
│   │  ├─ feedforward_dim=128                                                │
│   │  └─ 2 layers                                                           │
│   ├─ Mean pooling over 17 channels                                         │
│   └─ Classification Head: FC(64 → 32 → num_outputs)                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ 7. MODEL TESTING WITH DUMMY DATA                                            │
│ ├─ Test each model with random input matching its expected shape           │
│ ├─ Count total and trainable parameters                                    │
│ └─ Verify forward pass works without errors                                │
└─────────────────────────────────────────────────────────────────────────────┘

NOTEBOOK 2 OUTPUTS (for import into Notebooks 3 & 4):
├─ load_seedvig_5band_all_features()
├─ normalize_features_subjectwise()
├─ subject_wise_split()
├─ SEEDVIGSingleFrameDataset
├─ SEEDVIGMultiFrameDataset
├─ MLPBaseline
├─ CNN2DSingleFrame
├─ CNN2DMultiFrame
├─ EEGNet
├─ DECNN
└─ ChannelTransformer
```

---

## NOTEBOOK 3: Classification Training Pipeline

### Purpose
Train 24 classification models (4 features × 6 models) to predict binary alertness (Alert vs Fatigued).

```
NOTEBOOK 3 STRUCTURE
════════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. IMPORT FROM NOTEBOOK 2                                                   │
│ ├─ Use %run 2_baseline_models.ipynb to execute Notebook 2 first             │
│ └─ All functions and classes now available                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. CONFIGURATION                                                            │
│ ├─ Feature types: de_LDS, de_movingAve, psd_LDS, psd_movingAve             │
│ ├─ Models: mlp, cnn_single, cnn_multi, eegnet, decnn, transformer          │
│ ├─ PERCLOS threshold: 0.35 (>= 0.35 → Alert, < 0.35 → Fatigued)           │
│ ├─ Hyperparameters per model (batch size, learning rate, epochs)           │
│ └─ Paths: Save directory, results directory                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. TRAINING & EVALUATION FUNCTIONS                                          │
│                                                                              │
│ get_weighted_loss(y_train):                                                 │
│   ├─ Compute class weights: weights = 1 / count                            │
│   ├─ Normalize weights                                                      │
│   └─ Return weighted CrossEntropyLoss                                       │
│                                                                              │
│ evaluate_classification(model, data_loader, device):                        │
│   ├─ Set model.eval() (disable dropout/batch norm training mode)           │
│   ├─ No gradients computation (torch.no_grad())                            │
│   ├─ For each batch:                                                       │
│   │  ├─ Forward pass through model                                         │
│   │  ├─ argmax to get class predictions                                    │
│   │  └─ Collect predictions and labels                                     │
│   └─ Compute metrics: Accuracy, F1, Precision, Recall (sklearn)            │
│                                                                              │
│ train_classification(...):                                                  │
│   ├─ Move model to GPU device                                              │
│   ├─ Setup optimizer: Adam with lr and weight_decay                        │
│   ├─ Loop for num_epochs:                                                  │
│   │  ├─ Training phase:                                                    │
│   │  │  ├─ model.train() (enable dropout/batch norm training mode)         │
│   │  │  ├─ For each training batch:                                        │
│   │  │  │  ├─ Forward pass                                                 │
│   │  │  │  ├─ Compute weighted CrossEntropyLoss                            │
│   │  │  │  ├─ Backward pass                                                │
│   │  │  │  └─ Optimizer step                                               │
│   │  │  └─ Accumulate training loss                                        │
│   │  │                                                                      │
│   │  ├─ Validation phase (for model selection):                            │
│   │  │  ├─ Evaluate on validation set                                      │
│   │  │  ├─ Check if validation F1 > best_f1                               │
│   │  │  ├─ If yes: save model checkpoint (best_f1 updated)                │
│   │  │  └─ IF NO: don't save (keep previous best)                         │
│   │  │                                                                      │
│   │  └─ Print progress every 10 epochs                                     │
│   │                                                                          │
│   ├─ After all epochs:                                                     │
│   │  ├─ Load best model from checkpoint                                    │
│   │  ├─ Evaluate on TEST set (final evaluation, only happens once)         │
│   │  └─ Clear CUDA cache                                                   │
│   └─ Return test metrics                                                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. MODEL FACTORY                                                            │
│ ├─ create_model(model_name, num_outputs=2)                                 │
│ │  ├─ Check model_name and instantiate correct class                       │
│ │  └─ Return model instance                                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ 5. SINGLE EXPERIMENT RUNNER                                                 │
│                                                                              │
│ run_single_classification_experiment(feature_key, model_name)               │
│                                                                              │
│ STEP 1: Load Data                                                           │
│   ├─ load_seedvig_5band_all_features(BASE_DIR, feature_key)                │
│   └─ Returns: X_all (N, 17, 5), y_all (N,), subj_all (N,)                  │
│                                                                              │
│ STEP 2: Normalize                                                           │
│   ├─ X_norm = normalize_features_subjectwise(X_all, subj_all)              │
│   └─ Each session normalized independently                                 │
│                                                                              │
│ STEP 3: Create Binary Labels                                               │
│   ├─ y_binary = (y_all >= 0.35) → 0 or 1                                   │
│   │  └─ 1 = Alert (PERCLOS >= 0.35)                                        │
│   │  └─ 0 = Fatigued (PERCLOS < 0.35)                                      │
│   └─ Check class distribution                                              │
│                                                                              │
│ STEP 4: Split at Session Level (80-10-10)                                  │
│   ├─ train_mask, val_mask, test_mask = subject_wise_split(subj_all)       │
│   ├─ 18 sessions → train, 2 sessions → val, 3 sessions → test             │
│   ├─ Split X and y accordingly                                             │
│   └─ Prevents temporal data leakage                                        │
│                                                                              │
│ STEP 5: Create Datasets                                                    │
│   ├─ IF model == 'mlp':                                                    │
│   │  ├─ train_ds = SEEDVIGSingleFrameDataset(X_train, y_train,             │
│   │  │                                        flatten=True)                │
│   │  └─ Returns flattened (85,) inputs                                      │
│   │                                                                          │
│   ├─ ELIF model in [cnn_single, eegnet, decnn, transformer]:               │
│   │  ├─ train_ds = SEEDVIGSingleFrameDataset(X_train, y_train,             │
│   │  │                                        flatten=False)               │
│   │  └─ Returns 2D (17, 5) inputs                                           │
│   │                                                                          │
│   ├─ ELIF model == 'cnn_multi':                                            │
│   │  ├─ train_ds = SEEDVIGMultiFrameDataset(X_train, y_train, subj_train,  │
│   │  │                        T_seq=5, step=2)                             │
│   │  ├─ Creates sliding windows: (5, 17, 5)                                │
│   │  ├─ Labels aggregated via majority vote                                │
│   │  └─ Windows only created within sessions                               │
│   │                                                                          │
│   └─ Repeat for val_ds and test_ds                                         │
│                                                                              │
│ STEP 6: Create DataLoaders                                                 │
│   ├─ train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'],     │
│   │                              shuffle=True, num_workers=0)               │
│   ├─ val_loader = DataLoader(val_ds, ...)                                  │
│   └─ test_loader = DataLoader(test_ds, ...)                                │
│   └─ NOTE: num_workers=0 for Jupyter notebook safety                       │
│                                                                              │
│ STEP 7: Create Model                                                       │
│   ├─ model = create_model(model_name, num_outputs=2)                       │
│   ├─ 2 outputs for binary classification                                   │
│   └─ Count parameters                                                      │
│                                                                              │
│ STEP 8: Train Model                                                        │
│   ├─ metrics = train_classification(                                       │
│   │     model, train_loader, val_loader, test_loader, y_train,             │
│   │     num_epochs, lr, weight_decay, checkpoint_path, device)             │
│   │                                                                          │
│   ├─ Best model selected on validation F1                                  │
│   ├─ Final evaluation only on test set                                     │
│   └─ Model saved to checkpoint_path                                        │
│                                                                              │
│ STEP 9: Return Metrics                                                     │
│   └─ Return dict with: accuracy, f1, precision, recall                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ 6. RUN ALL 24 CLASSIFICATION EXPERIMENTS                                    │
│                                                                              │
│ FOR EACH feature_key in [de_LDS, de_movingAve, psd_LDS, psd_movingAve]:   │
│   FOR EACH model_name in [mlp, cnn_single, cnn_multi, eegnet, decnn, trf]: │
│     CALL run_single_classification_experiment(feature_key, model_name)      │
│     STORE results[feature_key][model_name]                                  │
│                                                                              │
│ Total experiments: 4 features × 6 models = 24 experiments                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ 7. SAVE RESULTS                                                             │
│                                                                              │
│ Save to JSON:                                                               │
│   ├─ File: results/baselines/classification_results.json                    │
│   └─ Content: All metrics for all 24 experiments                            │
│                                                                              │
│ Create Summary DataFrame:                                                  │
│   ├─ For each experiment: Feature, Model, Accuracy, F1, Precision, Recall  │
│   └─ Save to CSV: results/baselines/classification_summary.csv              │
│                                                                              │
│ Print Best Models per Feature Type:                                        │
│   ├─ Find model with highest F1 for each feature                           │
│   └─ Display comparison                                                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

NOTEBOOK 3 OUTPUTS:
├─ Models/baselines/classification/{feature_key}/{model_name}_best.pth (24 files)
├─ results/baselines/classification_results.json
└─ results/baselines/classification_summary.csv
```

---

## NOTEBOOK 4: Regression Training Pipeline

### Purpose
Train 24 regression models (4 features × 6 models) to predict continuous PERCLOS values (0-1).

```
NOTEBOOK 4 STRUCTURE
════════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. IMPORT FROM NOTEBOOK 2                                                   │
│ ├─ Use %run 2_baseline_models.ipynb to execute Notebook 2 first             │
│ └─ All functions and classes now available                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. CONFIGURATION                                                            │
│ ├─ Feature types: de_LDS, de_movingAve, psd_LDS, psd_movingAve             │
│ ├─ Models: mlp, cnn_single, cnn_multi, eegnet, decnn, transformer          │
│ ├─ Task: Predict continuous PERCLOS [0, 1]                                 │
│ ├─ Loss function: MSE (Mean Squared Error)                                 │
│ ├─ Hyperparameters per model (same as classification)                      │
│ └─ Paths: Save directory, results directory                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. TRAINING & EVALUATION FUNCTIONS                                          │
│                                                                              │
│ evaluate_regression(model, data_loader, device):                            │
│   ├─ Set model.eval() (disable dropout/batch norm training mode)           │
│   ├─ No gradients computation (torch.no_grad())                            │
│   ├─ For each batch:                                                       │
│   │  ├─ Forward pass through model                                         │
│   │  ├─ Squeeze output to (batch_size,)                                    │
│   │  └─ Collect predictions and labels                                     │
│   └─ Compute metrics:                                                       │
│       ├─ MSE: mean((y_true - y_pred)^2)                                     │
│       ├─ RMSE: sqrt(MSE)                                                    │
│       ├─ MAE: mean(|y_true - y_pred|)                                       │
│       └─ R²: coefficient of determination                                   │
│                                                                              │
│ train_regression(...):                                                      │
│   ├─ Move model to GPU device                                              │
│   ├─ Setup optimizer: Adam with lr and weight_decay                        │
│   ├─ Criterion: MSELoss()                                                   │
│   ├─ Loop for num_epochs:                                                  │
│   │  ├─ Training phase:                                                    │
│   │  │  ├─ model.train() (enable dropout/batch norm training mode)         │
│   │  │  ├─ For each training batch:                                        │
│   │  │  │  ├─ Forward pass                                                 │
│   │  │  │  ├─ Compute MSE loss                                             │
│   │  │  │  ├─ Backward pass                                                │
│   │  │  │  └─ Optimizer step                                               │
│   │  │  └─ Accumulate training loss                                        │
│   │  │                                                                      │
│   │  ├─ Validation phase (for model selection):                            │
│   │  │  ├─ Evaluate on validation set                                      │
│   │  │  ├─ Check if validation MSE < best_mse                             │
│   │  │  ├─ If yes: save model checkpoint (best_mse updated)               │
│   │  │  └─ If no: don't save (keep previous best)                         │
│   │  │  └─ KEY DIFFERENCE: Selection on MSE not F1                        │
│   │  │                                                                      │
│   │  └─ Print progress every 10 epochs                                     │
│   │                                                                          │
│   ├─ After all epochs:                                                     │
│   │  ├─ Load best model from checkpoint                                    │
│   │  ├─ Evaluate on TEST set (final evaluation, only happens once)         │
│   │  └─ Clear CUDA cache                                                   │
│   └─ Return test metrics                                                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. MODEL FACTORY                                                            │
│ ├─ create_model(model_name, num_outputs=1)                                 │
│ │  ├─ Check model_name and instantiate correct class                       │
│ │  ├─ REGRESSION: num_outputs=1 (single continuous value)                  │
│ │  └─ Return model instance                                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ 5. SINGLE EXPERIMENT RUNNER                                                 │
│                                                                              │
│ run_single_regression_experiment(feature_key, model_name)                   │
│                                                                              │
│ STEP 1: Load Data                                                           │
│   ├─ load_seedvig_5band_all_features(BASE_DIR, feature_key)                │
│   └─ Returns: X_all (N, 17, 5), y_all (N,), subj_all (N,)                  │
│                                                                              │
│ STEP 2: Normalize                                                           │
│   ├─ X_norm = normalize_features_subjectwise(X_all, subj_all)              │
│   └─ Each session normalized independently                                 │
│                                                                              │
│ STEP 3: NO Label Binarization                                              │
│   ├─ Keep y_all as continuous [0, 1] PERCLOS values                        │
│   └─ No thresholding here (unlike classification)                          │
│                                                                              │
│ STEP 4: Split at Session Level (80-10-10)                                  │
│   ├─ train_mask, val_mask, test_mask = subject_wise_split(subj_all)       │
│   ├─ 18 sessions → train, 2 sessions → val, 3 sessions → test             │
│   ├─ Split X and y accordingly                                             │
│   └─ Prevents temporal data leakage                                        │
│                                                                              │
│ STEP 5: Create Datasets                                                    │
│   ├─ IF model == 'mlp':                                                    │
│   │  ├─ train_ds = SEEDVIGSingleFrameDataset(X_train, y_train,             │
│   │  │              flatten=True, regression=True)                         │
│   │  └─ Returns flattened (85,) inputs                                      │
│   │                                                                          │
│   ├─ ELIF model in [cnn_single, eegnet, decnn, transformer]:               │
│   │  ├─ train_ds = SEEDVIGSingleFrameDataset(X_train, y_train,             │
│   │  │              flatten=False, regression=True)                        │
│   │  └─ Returns 2D (17, 5) inputs                                           │
│   │                                                                          │
│   ├─ ELIF model == 'cnn_multi':                                            │
│   │  ├─ train_ds = SEEDVIGMultiFrameDataset(X_train, y_train, subj_train,  │
│   │  │                 T_seq=5, step=2, regression=True)                   │
│   │  ├─ Creates sliding windows: (5, 17, 5)                                │
│   │  ├─ Labels aggregated via MEAN (not majority vote for regression)      │
│   │  └─ Windows only created within sessions                               │
│   │                                                                          │
│   └─ regression=True flag passed to dataset class                          │
│                                                                              │
│ STEP 6: Create DataLoaders                                                 │
│   ├─ train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'],     │
│   │                              shuffle=True, num_workers=0)               │
│   ├─ val_loader = DataLoader(val_ds, ...)                                  │
│   └─ test_loader = DataLoader(test_ds, ...)                                │
│   └─ NOTE: num_workers=0 for Jupyter notebook safety                       │
│                                                                              │
│ STEP 7: Create Model                                                       │
│   ├─ model = create_model(model_name, num_outputs=1)                       │
│   ├─ 1 output for continuous regression                                    │
│   └─ Count parameters                                                      │
│                                                                              │
│ STEP 8: Train Model                                                        │
│   ├─ metrics = train_regression(                                           │
│   │     model, train_loader, val_loader, test_loader,                      │
│   │     num_epochs, lr, weight_decay, checkpoint_path, device)             │
│   │                                                                          │
│   ├─ Best model selected on validation MSE                                 │
│   ├─ Final evaluation only on test set                                     │
│   └─ Model saved to checkpoint_path                                        │
│                                                                              │
│ STEP 9: Return Metrics                                                     │
│   └─ Return dict with: mse, rmse, mae, r2                                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ 6. RUN ALL 24 REGRESSION EXPERIMENTS                                        │
│                                                                              │
│ FOR EACH feature_key in [de_LDS, de_movingAve, psd_LDS, psd_movingAve]:   │
│   FOR EACH model_name in [mlp, cnn_single, cnn_multi, eegnet, decnn, trf]: │
│     CALL run_single_regression_experiment(feature_key, model_name)          │
│     STORE results[feature_key][model_name]                                  │
│                                                                              │
│ Total experiments: 4 features × 6 models = 24 experiments                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ 7. SAVE RESULTS                                                             │
│                                                                              │
│ Save to JSON:                                                               │
│   ├─ File: results/baselines/regression_results.json                        │
│   └─ Content: All metrics for all 24 experiments                            │
│                                                                              │
│ Create Summary DataFrame:                                                  │
│   ├─ For each experiment: Feature, Model, Params, MSE, RMSE, MAE, R²       │
│   └─ Save to CSV: results/baselines/regression_summary.csv                  │
│                                                                              │
│ Print Best Models per Feature Type:                                        │
│   ├─ Find model with lowest MSE for each feature                           │
│   └─ Display comparison                                                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

NOTEBOOK 4 OUTPUTS:
├─ Models/baselines/regression/{feature_key}/{model_name}_best.pth (24 files)
├─ results/baselines/regression_results.json
└─ results/baselines/regression_summary.csv
```

---

## COMPLETE PIPELINE FLOW (2→3→4)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ NOTEBOOK 2: Foundation (MUST RUN FIRST)                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│ ✓ Define all reusable components                                           │
│ ✓ Test models with dummy data                                              │
│ ✓ Export classes and functions                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
                        (Import via %run magic)
                                    ↓
┌──────────────────────────────────┬──────────────────────────────────────────┐
│ NOTEBOOK 3 (PARALLEL)            │ NOTEBOOK 4 (PARALLEL)                    │
│ Classification (24 experiments)  │ Regression (24 experiments)              │
├──────────────────────────────────┼──────────────────────────────────────────┤
│                                  │                                          │
│ Load features → Normalize        │ Load features → Normalize                │
│         ↓                        │         ↓                                │
│ Create binary labels             │ Keep continuous labels                   │
│ (PERCLOS >= 0.35)                │ (PERCLOS ∈ [0,1])                        │
│         ↓                        │         ↓                                │
│ 80-10-10 split per session       │ 80-10-10 split per session              │
│         ↓                        │         ↓                                │
│ FOR each of 24 combinations:     │ FOR each of 24 combinations:            │
│   ├─ Create dataset              │   ├─ Create dataset                      │
│   ├─ Train model                 │   ├─ Train model                         │
│   │  ├─ Selection: max F1 (val)  │   │  ├─ Selection: min MSE (val)         │
│   │  └─ Eval: test set           │   │  └─ Eval: test set                   │
│   ├─ Save checkpoint             │   ├─ Save checkpoint                     │
│   └─ Store metrics               │   └─ Store metrics                       │
│         ↓                        │         ↓                                │
│ Save all 24 models               │ Save all 24 models                       │
│ Save classification_results.json │ Save regression_results.json             │
│ Save classification_summary.csv  │ Save regression_summary.csv              │
│                                  │                                          │
└──────────────────────────────────┴──────────────────────────────────────────┘
```

---

## KEY DIFFERENCES BETWEEN CLASSIFICATION (NB3) & REGRESSION (NB4)

| Aspect | Classification (NB3) | Regression (NB4) |
|--------|---------------------|------------------|
| **Target Labels** | Binary: 0 or 1 | Continuous: [0, 1] |
| **Thresholding** | Yes: PERCLOS >= 0.35 | No: Keep raw values |
| **Model Output** | `num_outputs=2` | `num_outputs=1` |
| **Loss Function** | Weighted CrossEntropyLoss | MSELoss |
| **Model Selection** | Best validation F1 | Best validation MSE |
| **Metrics** | Accuracy, F1, Precision, Recall | MSE, RMSE, MAE, R² |
| **Multi-Frame Labels** | Majority vote | Mean aggregate |
| **Expected Performance** | F1: 0.65-0.84 | R²: 0.55-0.82 |

---

## DATA FLOW DIAGRAM

```
RAW DATA
   ├─ SEED-VIG/EEG_Feature_5Bands/{session_id}.mat
   │  └─ Shape: (17 channels, 885 frames, 5 bands)
   │
   └─ SEED-VIG/perclos_labels/{session_id}.mat
      └─ Shape: (885,) continuous [0, 1]
            ↓ load_seedvig_5band_all_features()
            ↓ (23 sessions concatenated)
            ↓
    LOADED DATA
       ├─ X_all: (20,355, 17, 5)
       ├─ y_all: (20,355,) PERCLOS values
       └─ subj_all: (20,355,) session IDs (1-23)
            ↓ normalize_features_subjectwise()
            ↓
    NORMALIZED DATA
       └─ X_norm: (20,355, 17, 5) z-scored per session
            ↓ (Classification) subject_wise_split() + binary labels
            ├─ X_train: (16,280, 17, 5)  y_train: (16,280,) {0,1}
            ├─ X_val: (2,037, 17, 5)     y_val: (2,037,) {0,1}
            └─ X_test: (2,038, 17, 5)    y_test: (2,038,) {0,1}
            ↓
    CLASSIFICATION DATASETS
       ├─ SingleFrame: 1 frame per sample → (17, 5) or (85,)
       └─ MultiFrame: 5 frame windows → (5, 17, 5)
            ↓
    CLASSIFICATION TRAINING (24 experiments)
       ├─ Train each model on train set
       ├─ Select best on validation set (max F1)
       ├─ Evaluate on test set (once)
       └─ Save checkpoints
            ↓
    SAVED MODELS & RESULTS
       ├─ Models/baselines/classification/{feature}/{model}_best.pth (24 files)
       └─ results/baselines/classification_results.json
            ↓
            ↓ (Regression) subject_wise_split() + continuous labels
            ├─ X_train: (16,280, 17, 5)  y_train: (16,280,) ∈ [0,1]
            ├─ X_val: (2,037, 17, 5)     y_val: (2,037,) ∈ [0,1]
            └─ X_test: (2,038, 17, 5)    y_test: (2,038,) ∈ [0,1]
            ↓
    REGRESSION DATASETS
       ├─ SingleFrame: 1 frame per sample → (17, 5) or (85,)
       └─ MultiFrame: 5 frame windows → (5, 17, 5)
            ↓
    REGRESSION TRAINING (24 experiments)
       ├─ Train each model on train set
       ├─ Select best on validation set (min MSE)
       ├─ Evaluate on test set (once)
       └─ Save checkpoints
            ↓
    SAVED MODELS & RESULTS
       ├─ Models/baselines/regression/{feature}/{model}_best.pth (24 files)
       └─ results/baselines/regression_results.json
```

---

## EXECUTION CHECKLIST

```
□ Run Notebook 2
  ├─ Check all imports work
  ├─ Verify GPU is available
  ├─ All 6 models test successfully
  └─ No errors in dummy data tests

□ Run Notebook 3 (Classification)
  ├─ Notebook 2 executed successfully
  ├─ 24 experiments run without errors
  ├─ Check Models/baselines/classification/ for 24 files
  └─ Check results/baselines/classification_results.json exists

□ Run Notebook 4 (Regression)
  ├─ Notebook 2 executed successfully
  ├─ 24 experiments run without errors
  ├─ Check Models/baselines/regression/ for 24 files
  └─ Check results/baselines/regression_results.json exists

□ Run Notebook 5 (Results Analysis)
  ├─ Both NB3 and NB4 completed
  ├─ Visualize results
  └─ Generate comparison plots
```

---

## COMMON PARAMETERS SUMMARY

```
DATASET CONFIGURATION:
├─ 23 recording sessions
├─ ~885 EEG frames per session
├─ 17 EEG channels
├─ 5 frequency bands
├─ 4 feature types: de_LDS, de_movingAve, psd_LDS, psd_movingAve
└─ Continuous PERCLOS labels [0, 1]

SPLITTING:
├─ Train: 18 sessions (80%)
├─ Validation: 2 sessions (10%)
├─ Test: 3 sessions (10%)
├─ Random seed: 42
└─ Critical: Session-wise (no temporal leakage)

NORMALIZATION:
├─ Type: Z-score (mean=0, std=1)
├─ Scope: Per-session independent
├─ Formula: (x - session_mean) / (session_std + 1e-8)
└─ Applied: Before splitting

CLASSIFICATION TASK:
├─ Threshold: PERCLOS >= 0.35
├─ Classes: Alert (1) vs Fatigued (0)
├─ Loss: Weighted CrossEntropyLoss
├─ Selection metric: Validation F1
└─ 24 experiments: 4 features × 6 models

REGRESSION TASK:
├─ Target: Continuous PERCLOS [0, 1]
├─ Loss: MSELoss
├─ Selection metric: Validation MSE
├─ 24 experiments: 4 features × 6 models
└─ Evaluation metrics: MSE, RMSE, MAE, R²

MODELS (6 total):
├─ 1. MLP Baseline (~21K params)
├─ 2. CNN2D Single Frame (~23K params)
├─ 3. CNN2D Multi-Frame (~101K params)
├─ 4. EEGNet (~730 params) ⭐ Most efficient
├─ 5. DE-CNN (~118K params)
└─ 6. Channel Transformer (~70K params)

HYPERPARAMETERS:
├─ Optimizer: Adam
├─ Weight decay: 1e-4
├─ No learning rate scheduler
├─ Batch sizes: 64-256 (model-dependent)
├─ Learning rates: 1e-3 or 5e-4 (model-dependent)
├─ Epochs: 50-100 (model-dependent)
└─ Dropout: 0.25-0.4 (model-dependent)

DATALOADER:
├─ num_workers: 0 (Jupyter safety)
├─ shuffle: True (training only)
├─ pin_memory: False (not needed with num_workers=0)
└─ Multi-frame stride: 2 (within sessions only)
```

---

## TIPS FOR UNDERSTANDING THE FLOW

1. **Start with Notebook 2**: This defines everything. Understand the 6 models and dataset classes first.

2. **Notice the Symmetry**: Notebooks 3 and 4 are nearly identical, only differing in:
   - Label type (binary vs continuous)
   - Loss function (CrossEntropyLoss vs MSELoss)
   - Selection metric (F1 vs MSE)
   - Evaluation metrics (different for each task)

3. **Session-wise is Key**: All operations respect session boundaries:
   - Normalization: per-session
   - Splitting: per-session
   - Multi-frame windows: within-session only

4. **Train/Val/Test is Strict**:
   - Train: Build knowledge
   - Validation: Choose best model
   - Test: Final evaluation (happens only once after loading best model)

5. **Two Parallel Pipelines**:
   - Classification (NB3): Answer "Is the driver alert?"
   - Regression (NB4): Answer "What is the driver's PERCLOS value?"

---

## FILES GENERATED

```
After Notebook 2:
└─ All classes/functions in memory

After Notebook 3:
├─ Models/baselines/classification/
│  ├─ de_LDS/
│  │  ├─ mlp_best.pth
│  │  ├─ cnn_single_best.pth
│  │  ├─ cnn_multi_best.pth
│  │  ├─ eegnet_best.pth
│  │  ├─ decnn_best.pth
│  │  └─ transformer_best.pth
│  ├─ de_movingAve/ (6 files)
│  ├─ psd_LDS/ (6 files)
│  └─ psd_movingAve/ (6 files)
├─ results/baselines/
│  ├─ classification_results.json
│  └─ classification_summary.csv

After Notebook 4:
├─ Models/baselines/regression/
│  ├─ de_LDS/ (6 files)
│  ├─ de_movingAve/ (6 files)
│  ├─ psd_LDS/ (6 files)
│  └─ psd_movingAve/ (6 files)
├─ results/baselines/
│  ├─ regression_results.json
│  └─ regression_summary.csv

Total: 48 trained model checkpoints (24 classification + 24 regression)
```
