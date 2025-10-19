"""
Stacking Meta-Learner Ensemble for Amazon ML Challenge
Combines Model A (LightGBM) and Model B (Qwen2-VL) predictions

IMPORTANT: Price Scale Handling
===============================
Model A (LightGBM):
- Training: Uses log(price+1) as target internally
- OOF Output: Saves BOTH 'oof_price' (normal scale) AND 'oof_log_price' (log scale)
- Test Output: Saves 'price' column in NORMAL PRICE SCALE (already converted via np.expm1)
  
Model B (VLM):
- Training: Directly predicts price (no log transformation)
- OOF Output: Saves 'oof_pred' in NORMAL PRICE SCALE
- Test Output: Saves 'price' column in NORMAL PRICE SCALE

This ensemble script ensures all predictions are in NORMAL PRICE SCALE before meta-learning.
The meta-learner trains on normal price scale and predicts in normal price scale.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error
import json
import os
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

# Paths to OOF predictions (Out-Of-Fold predictions from Model A & B)
MODEL_A_OOF_PATH = "Approach LGBM(with Feat Engg)/output/oof_predictions_train.csv"  # sample_id, fold, oof_pred_log, actual_price
MODEL_B_OOF_PATH = "Approach VLM(with SFT)/DDP Approach/oof_predictions_train.csv"   # sample_id, fold, oof_pred, actual_price

# Paths to test predictions from both models
MODEL_A_TEST_PATH = "Approach LGBM(with Feat Engg)/dataset/sample_test_out.csv"  # sample_id, price (log-transformed)
MODEL_B_TEST_PATH = "Approach VLM(with SFT)/DDP Approach/test_out.csv"            # sample_id, price

# Path to master data for global features
MASTER_TRAIN_PATH = "Approach LGBM(with Feat Engg)/input/master_train.parquet"
MASTER_TEST_PATH = "Approach LGBM(with Feat Engg)/input/master_test.parquet"

# Output paths
OUTPUT_DIR = "ensemble_output"
FINAL_SUBMISSION_PATH = "test_out_ensemble.csv"

# Meta-learner configuration
META_LEARNER_TYPE = "lgbm"  # Options: "lgbm", "ridge", "weighted_avg"
N_FOLDS = 5  # Number of folds for stacking (should match the OOF folds)

# ============================================================================
# SMAPE Metric
# ============================================================================

def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred))
    # Avoid division by zero
    diff = np.abs(y_true - y_pred) / np.where(denominator == 0, 1, denominator)
    return 200.0 * np.mean(diff)

# ============================================================================
# Load OOF Predictions
# ============================================================================

def load_oof_predictions():
    """Load and merge OOF predictions from both models"""
    print("Loading OOF predictions...")
    
    # Load Model A OOF (LightGBM)
    # Expected columns from train_model_a.py: sample_id, oof_price, oof_log_price
    if os.path.exists(MODEL_A_OOF_PATH):
        oof_a = pd.read_csv(MODEL_A_OOF_PATH)
        
        # Model A saves both oof_price (normal scale) and oof_log_price (log scale)
        # We use oof_price which is already in normal price scale
        if 'oof_price' in oof_a.columns:
            oof_a['pred_a'] = oof_a['oof_price']  # Already in price scale
            print("   ✓ Model A: Using 'oof_price' column (normal price scale)")
        elif 'oof_log_price' in oof_a.columns:
            oof_a['pred_a'] = np.expm1(oof_a['oof_log_price'])  # Convert from log to price
            print("   ✓ Model A: Using 'oof_log_price' column (converted to price scale)")
        elif 'oof_pred' in oof_a.columns:
            # Fallback: check if it's in log scale by looking at value ranges
            sample_val = oof_a['oof_pred'].iloc[0]
            if sample_val < 15:  # Likely log scale (log(price+1) typically < 15)
                oof_a['pred_a'] = np.expm1(oof_a['oof_pred'])
                print("   ✓ Model A: Using 'oof_pred' column (detected log scale, converted)")
            else:
                oof_a['pred_a'] = oof_a['oof_pred']
                print("   ✓ Model A: Using 'oof_pred' column (detected normal price scale)")
        else:
            raise ValueError(f"Model A OOF: Cannot find prediction column. Available: {oof_a.columns.tolist()}")
        
        # Check if fold column exists, if not create it
        if 'fold' not in oof_a.columns:
            print("   ⚠ Warning: 'fold' column not found in Model A OOF, creating dummy folds")
            oof_a['fold'] = 0
        
        oof_a = oof_a[['sample_id', 'fold', 'pred_a']]
    else:
        raise FileNotFoundError(f"Model A OOF not found: {MODEL_A_OOF_PATH}")
    
    # Load Model B OOF (VLM - direct price predictions)
    if os.path.exists(MODEL_B_OOF_PATH):
        oof_b = pd.read_csv(MODEL_B_OOF_PATH)
        
        # Model B should output direct price predictions
        if 'oof_pred' in oof_b.columns:
            oof_b['pred_b'] = oof_b['oof_pred']
            print("   ✓ Model B: Using 'oof_pred' column (direct price predictions)")
        elif 'price' in oof_b.columns:
            oof_b['pred_b'] = oof_b['price']
            print("   ✓ Model B: Using 'price' column (direct price predictions)")
        else:
            raise ValueError(f"Model B OOF: Cannot find prediction column. Available: {oof_b.columns.tolist()}")
        
        # Check if fold column exists
        if 'fold' not in oof_b.columns:
            print("   ⚠ Warning: 'fold' column not found in Model B OOF, creating dummy folds")
            oof_b['fold'] = 0
        
        oof_b = oof_b[['sample_id', 'fold', 'pred_b']]
    else:
        raise FileNotFoundError(f"Model B OOF not found: {MODEL_B_OOF_PATH}")
    
    # Merge both OOF predictions
    oof_merged = pd.merge(oof_a, oof_b, on=['sample_id', 'fold'], how='inner')
    
    print(f"\n✓ Loaded {len(oof_merged)} OOF predictions from both models")
    print(f"  Model A pred range: [{oof_merged['pred_a'].min():.2f}, {oof_merged['pred_a'].max():.2f}]")
    print(f"  Model B pred range: [{oof_merged['pred_b'].min():.2f}, {oof_merged['pred_b'].max():.2f}]\n")
    
    return oof_merged

# ============================================================================
# Load Test Predictions
# ============================================================================

def load_test_predictions():
    """Load test predictions from both models"""
    print("Loading test predictions...")
    
    # Load Model A test predictions (LightGBM)
    # predict_model_a.py line 119: preds_price = np.expm1(preds_log)
    # So the saved 'price' column is ALREADY in normal price scale, NOT log scale
    if os.path.exists(MODEL_A_TEST_PATH):
        test_a = pd.read_csv(MODEL_A_TEST_PATH)
        
        if 'price' not in test_a.columns:
            raise ValueError(f"Model A test: 'price' column not found. Available: {test_a.columns.tolist()}")
        
        # Model A's predict_model_a.py already converts log→price via np.expm1
        # So we use it directly WITHOUT another transformation
        test_a['pred_a'] = test_a['price']  # Already in normal price scale
        print("   ✓ Model A: Loaded test predictions (already in normal price scale)")
    else:
        raise FileNotFoundError(f"Model A test predictions not found: {MODEL_A_TEST_PATH}")
    
    # Load Model B test predictions (VLM - direct price predictions)
    if os.path.exists(MODEL_B_TEST_PATH):
        test_b = pd.read_csv(MODEL_B_TEST_PATH)
        
        if 'price' not in test_b.columns:
            raise ValueError(f"Model B test: 'price' column not found. Available: {test_b.columns.tolist()}")
        
        test_b['pred_b'] = test_b['price']  # Direct price predictions
        print("   ✓ Model B: Loaded test predictions (direct price predictions)")
    else:
        raise FileNotFoundError(f"Model B test predictions not found: {MODEL_B_TEST_PATH}")
    
    # Merge both test predictions
    test_merged = pd.merge(
        test_a[['sample_id', 'pred_a']], 
        test_b[['sample_id', 'pred_b']], 
        on='sample_id', 
        how='outer'
    )
    
    # Handle missing predictions
    test_merged['pred_a'].fillna(test_merged['pred_a'].median(), inplace=True)
    test_merged['pred_b'].fillna(test_merged['pred_b'].median(), inplace=True)
    
    print(f"\n✓ Loaded {len(test_merged)} test predictions from both models")
    print(f"  Model A pred range: [{test_merged['pred_a'].min():.2f}, {test_merged['pred_a'].max():.2f}]")
    print(f"  Model B pred range: [{test_merged['pred_b'].min():.2f}, {test_merged['pred_b'].max():.2f}]\n")
    
    return test_merged

# ============================================================================
# Load Global Features
# ============================================================================

def load_global_features():
    """Load global features from master data"""
    print("Loading global features...")
    
    # Load training data
    if MASTER_TRAIN_PATH.endswith('.parquet'):
        train = pd.read_parquet(MASTER_TRAIN_PATH)
    else:
        train = pd.read_csv(MASTER_TRAIN_PATH)
    
    # Load test data
    if MASTER_TEST_PATH.endswith('.parquet'):
        test = pd.read_parquet(MASTER_TEST_PATH)
    else:
        test = pd.read_csv(MASTER_TEST_PATH)
    
    # Select global features (adjust based on what's available in your master data)
    feature_cols = []
    
    # Numeric features
    numeric_candidates = ['mass_g', 'volume_ml', 'pack_count', 'density', 
                         'mass_per_pack', 'volume_per_pack', 'total_flags']
    for col in numeric_candidates:
        if col in train.columns:
            feature_cols.append(col)
    
    # Categorical features (if encoded)
    categorical_candidates = ['category_encoded', 'brand_encoded', 'text_kmeans_cluster_id']
    for col in categorical_candidates:
        if col in train.columns:
            feature_cols.append(col)
    
    if not feature_cols:
        print("Warning: No global features found, using only model predictions")
        train_features = train[['sample_id', 'price']].copy() if 'price' in train.columns else train[['sample_id']].copy()
        test_features = test[['sample_id']].copy()
    else:
        print(f"Using {len(feature_cols)} global features: {feature_cols}")
        train_features = train[['sample_id'] + (['price'] if 'price' in train.columns else []) + feature_cols].copy()
        test_features = test[['sample_id'] + feature_cols].copy()
        
        # Fill missing values
        for col in feature_cols:
            if train_features[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                median_val = train_features[col].median()
                train_features[col].fillna(median_val, inplace=True)
                test_features[col].fillna(median_val, inplace=True)
    
    return train_features, test_features, feature_cols

# ============================================================================
# Meta-Learner Training
# ============================================================================

def train_meta_learner_lgbm(X_train, y_train, X_val, y_val):
    """Train LightGBM meta-learner"""
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42
    }
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    return model

def train_meta_learner_ridge(X_train, y_train):
    """Train Ridge regression meta-learner"""
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    return model

# ============================================================================
# Stacking Ensemble Pipeline
# ============================================================================

def train_stacking_ensemble():
    """Train stacking meta-learner on OOF predictions"""
    print("\n" + "="*80)
    print("STACKING ENSEMBLE TRAINING")
    print("="*80 + "\n")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Step 1: Load OOF predictions
    oof_preds = load_oof_predictions()
    
    # Step 2: Load global features
    train_features, test_features, feature_cols = load_global_features()
    
    # Step 3: Merge OOF predictions with global features and target
    train_data = pd.merge(oof_preds, train_features, on='sample_id', how='left')
    
    # Ensure we have the target
    if 'price' not in train_data.columns:
        raise ValueError("Target variable 'price' not found in training data")
    
    # Step 4: Prepare features for meta-learner
    meta_features = ['pred_a', 'pred_b'] + feature_cols
    
    # Drop rows with missing values
    train_data = train_data.dropna(subset=meta_features + ['price'])
    
    print(f"Training data shape: {train_data.shape}")
    print(f"Meta-features: {meta_features}")
    
    # Step 5: Train meta-learner with fold-wise validation
    meta_models = []
    oof_meta_preds = np.zeros(len(train_data))
    
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_data)):
        print(f"\nTraining meta-learner on Fold {fold + 1}/{N_FOLDS}...")
        
        X_train = train_data.iloc[train_idx][meta_features].values
        y_train = train_data.iloc[train_idx]['price'].values
        
        X_val = train_data.iloc[val_idx][meta_features].values
        y_val = train_data.iloc[val_idx]['price'].values
        
        # Train meta-learner
        if META_LEARNER_TYPE == "lgbm":
            meta_model = train_meta_learner_lgbm(X_train, y_train, X_val, y_val)
            val_preds = meta_model.predict(X_val, num_iteration=meta_model.best_iteration)
        elif META_LEARNER_TYPE == "ridge":
            meta_model = train_meta_learner_ridge(X_train, y_train)
            val_preds = meta_model.predict(X_val)
        else:  # weighted_avg
            # Simple weighted average (weights learned from validation)
            weights = np.array([0.5, 0.5])  # Equal weights as baseline
            meta_model = {'type': 'weighted_avg', 'weights': weights}
            val_preds = X_val[:, 0] * weights[0] + X_val[:, 1] * weights[1]
        
        # Store OOF predictions
        oof_meta_preds[val_idx] = val_preds
        
        # Calculate fold SMAPE
        fold_smape = smape(y_val, val_preds)
        fold_scores.append(fold_smape)
        print(f"Fold {fold + 1} SMAPE: {fold_smape:.4f}")
        
        # Save model
        meta_models.append(meta_model)
    
    # Overall OOF score
    overall_smape = smape(train_data['price'].values, oof_meta_preds)
    print(f"\n{'='*80}")
    print(f"Overall OOF SMAPE: {overall_smape:.4f}")
    print(f"Mean Fold SMAPE: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    print(f"{'='*80}\n")
    
    # Save OOF predictions
    oof_results = train_data[['sample_id', 'price']].copy()
    oof_results['oof_ensemble_pred'] = oof_meta_preds
    oof_results.to_csv(f"{OUTPUT_DIR}/oof_ensemble_predictions.csv", index=False)
    
    # Save metadata
    metadata = {
        'n_folds': N_FOLDS,
        'meta_learner_type': META_LEARNER_TYPE,
        'meta_features': meta_features,
        'overall_oof_smape': float(overall_smape),
        'fold_smapes': [float(s) for s in fold_scores],
        'mean_fold_smape': float(np.mean(fold_scores)),
        'std_fold_smape': float(np.std(fold_scores))
    }
    
    with open(f"{OUTPUT_DIR}/ensemble_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return meta_models, meta_features, train_data

# ============================================================================
# Test Prediction
# ============================================================================

def predict_test(meta_models, meta_features):
    """Generate ensemble predictions for test set"""
    print("\n" + "="*80)
    print("GENERATING TEST PREDICTIONS")
    print("="*80 + "\n")
    
    # Load test predictions from both models
    test_preds = load_test_predictions()
    
    # Load global features for test
    _, test_features, _ = load_global_features()
    
    # Merge test predictions with global features
    test_data = pd.merge(test_preds, test_features, on='sample_id', how='left')
    
    # Prepare features
    for col in meta_features:
        if col not in test_data.columns:
            if col == 'pred_a':
                test_data[col] = test_data['pred_a']
            elif col == 'pred_b':
                test_data[col] = test_data['pred_b']
            else:
                # Fill missing global features with 0 or median
                test_data[col] = 0
    
    X_test = test_data[meta_features].fillna(0).values
    
    # Average predictions from all fold models
    test_ensemble_preds = np.zeros(len(test_data))
    
    for fold, meta_model in enumerate(meta_models):
        print(f"Predicting with meta-model from Fold {fold + 1}/{len(meta_models)}...")
        
        if META_LEARNER_TYPE == "lgbm":
            fold_preds = meta_model.predict(X_test, num_iteration=meta_model.best_iteration)
        elif META_LEARNER_TYPE == "ridge":
            fold_preds = meta_model.predict(X_test)
        else:  # weighted_avg
            weights = meta_model['weights']
            fold_preds = X_test[:, 0] * weights[0] + X_test[:, 1] * weights[1]
        
        test_ensemble_preds += fold_preds
    
    # Average across folds
    test_ensemble_preds /= len(meta_models)
    
    # Ensure predictions are positive
    test_ensemble_preds = np.maximum(test_ensemble_preds, 0.01)
    
    # Create submission
    submission = pd.DataFrame({
        'sample_id': test_data['sample_id'],
        'price': test_ensemble_preds
    })
    
    # Ensure we have all 75,000 test samples
    if len(submission) < 75000:
        print(f"Warning: Only {len(submission)} predictions. Expected 75,000.")
    
    # Save submission
    submission.to_csv(FINAL_SUBMISSION_PATH, index=False)
    print(f"\nFinal submission saved to: {FINAL_SUBMISSION_PATH}")
    print(f"Number of predictions: {len(submission)}")
    print(f"Price range: [{submission['price'].min():.2f}, {submission['price'].max():.2f}]")
    print(f"Mean price: {submission['price'].mean():.2f}")
    
    return submission

# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main ensemble pipeline"""
    print("\n" + "="*80)
    print("STACKING META-LEARNER ENSEMBLE")
    print("Model A: LightGBM (log-price)")
    print("Model B: Qwen2-VL-7B (QLoRA)")
    print("="*80)
    
    # Train stacking ensemble
    meta_models, meta_features, train_data = train_stacking_ensemble()
    
    # Generate test predictions
    submission = predict_test(meta_models, meta_features)
    
    print("\n" + "="*80)
    print("ENSEMBLE COMPLETE!")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
