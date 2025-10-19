# =============================================================================
# 0. SETUP AND IMPORTS
# =============================================================================
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import os
import gc
import json
import joblib
from sklearn.model_selection import train_test_split, KFold
from typing import List

# Ignore Optuna's experimental warning for progress bar
optuna.logging.set_verbosity(optuna.logging.WARNING)


# =============================================================================
# 0.a RUNTIME SWITCHES (set via env for fast runs on deadline)
# =============================================================================
FAST_MODE = os.environ.get('FAST_MODE', '1') == '1'  # default fast for deadline
NUM_THREADS = int(os.environ.get('NUM_THREADS', '64'))
HPO_TRIALS = int(os.environ.get('HPO_TRIALS', '15' if FAST_MODE else '60'))
EARLY_STOP_HPO = int(os.environ.get('EARLY_STOP_HPO', '100' if FAST_MODE else '200'))
EARLY_STOP_FINAL = int(os.environ.get('EARLY_STOP_FINAL', '150' if FAST_MODE else '300'))
MAX_BIN = int(os.environ.get('MAX_BIN', '255' if FAST_MODE else '511'))
BIN_CONSTRUCT = int(os.environ.get('BIN_CONSTRUCT', '200000' if FAST_MODE else '2000000'))
SKIP_OOF = os.environ.get('SKIP_OOF', '1' if FAST_MODE else '0') == '1'
OOF_SPLITS = int(os.environ.get('OOF_SPLITS', '3' if FAST_MODE else '5'))
HPO_SAMPLE_FRAC = float(os.environ.get('HPO_SAMPLE_FRAC', '0.25' if FAST_MODE else '1.0'))
HPO_MIN_ROWS = int(os.environ.get('HPO_MIN_ROWS', '12000' if FAST_MODE else '0'))
HPO_TIMEOUT = int(os.environ.get('HPO_TIMEOUT', '900' if FAST_MODE else '0'))  # seconds, 0=disabled
HPO_N_EST = int(os.environ.get('HPO_N_EST', '3000' if FAST_MODE else '6000'))
HPO_NUM_LEAVES_MAX = int(os.environ.get('HPO_NUM_LEAVES_MAX', '1024' if FAST_MODE else '2048'))


# =============================================================================
# 1. ERROR FUNCTION (SMAPE)
# =============================================================================
def smape(y_true, y_pred):
    """
    Calculates the Symmetric Mean Absolute Percentage Error (SMAPE).
    This error is given as a percentage (0-200%).
    """
    numerator = np.abs(y_pred - y_true)
    # Add a small epsilon to the denominator to handle cases where both true and pred are zero
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2 + 1e-8
    return np.mean(numerator / denominator) * 100

def lgbm_smape_on_log_metric(y_true, y_pred):
    """
    Custom SMAPE on price-space while training on log(price+1).
    LightGBM passes y_true and y_pred as the training target space (log).
    """
    y_true_price = np.expm1(y_true)
    y_pred_price = np.expm1(y_pred)
    score = smape(y_true_price, y_pred_price)
    return 'smape', score, False  # lower is better


# =============================================================================
# 2. DATA LOADING (merge all CSVs from input/)
# =============================================================================
def load_input_data(input_dir: str = 'input') -> pd.DataFrame:
    """
    Preferred: loads 'master_train.parquet' from input_dir if present (fast, typed).
    Fallbacks:
      1) 'master_train.csv' if Parquet missing
      2) Merge all CSV/Parquet files on 'sample_id' if master not found
    - Avoid duplicate columns by dropping overlaps from later files (except 'sample_id').
    - Downcast numerics to save memory.
    """
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Preferred single-file path (Parquet)
    master_parquet = os.path.join(input_dir, 'master_train.parquet')
    if os.path.exists(master_parquet):
        print(f"--- Using master file: {master_parquet} ---")
        try:
            base_df = pd.read_parquet(master_parquet)
        except Exception as e:
            raise RuntimeError(f"Failed to read Parquet: {master_parquet} ({e})")
        if 'sample_id' not in base_df.columns:
            raise KeyError("'sample_id' column missing in master_train.parquet")
        base_df['sample_id'] = base_df['sample_id'].astype(str)

        # Downcast numeric dtypes for memory efficiency
        float_cols = base_df.select_dtypes(include=['float64']).columns
        int_cols = base_df.select_dtypes(include=['int64']).columns
        if len(float_cols) > 0:
            base_df[float_cols] = base_df[float_cols].astype(np.float32)
        if len(int_cols) > 0:
            base_df[int_cols] = base_df[int_cols].astype(np.int32)
        print(f"   -> Loaded master with shape {base_df.shape}")
        return base_df

    # CSV master fallback
    master_path = os.path.join(input_dir, 'master_train.csv')
    if os.path.exists(master_path):
        print(f"--- Using master file: {master_path} ---")
        try:
            base_df = pd.read_csv(master_path)
        except Exception as e:
            raise RuntimeError(f"Failed to read CSV: {master_path} ({e})")
        if 'sample_id' not in base_df.columns:
            raise KeyError("'sample_id' column missing in master_train.csv")
        base_df['sample_id'] = base_df['sample_id'].astype(str)

        # Downcast numeric dtypes for memory efficiency
        float_cols = base_df.select_dtypes(include=['float64']).columns
        int_cols = base_df.select_dtypes(include=['int64']).columns
        if len(float_cols) > 0:
            base_df[float_cols] = base_df[float_cols].astype(np.float32)
        if len(int_cols) > 0:
            base_df[int_cols] = base_df[int_cols].astype(np.int32)
        print(f"   -> Loaded master with shape {base_df.shape}")
        return base_df

    # Fallback: merge any CSV/Parquet files
    csv_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith('.csv')]
    pq_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith('.parquet')]
    all_files = csv_files + pq_files
    if len(all_files) == 0:
        raise FileNotFoundError(f"No CSV/Parquet files found in: {input_dir}")

    loaded = []
    for path in all_files:
        try:
            if path.lower().endswith('.parquet'):
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(path)
            if 'sample_id' not in df.columns:
                continue
            df['sample_id'] = df['sample_id'].astype(str)
            loaded.append((os.path.basename(path), df))
            print(f"   -> Loaded {os.path.basename(path)} with shape {df.shape}")
        except Exception as e:
            print(f"   -> Skipped {os.path.basename(path)} due to error: {e}")

    if len(loaded) == 0:
        raise RuntimeError("No valid CSVs with 'sample_id' column found in input directory")

    # Choose base dataframe
    base_idx = next((i for i, (_, d) in enumerate(loaded) if 'price' in d.columns), None)
    if base_idx is None:
        base_idx = max(range(len(loaded)), key=lambda i: loaded[i][1].shape[0])
    base_name, base_df = loaded[base_idx]
    print(f"--- Using base file: {base_name} ---")

    # Merge others without overlapping columns (except 'sample_id')
    for i, (name, df_other) in enumerate(loaded):
        if i == base_idx:
            continue
        overlap = [c for c in df_other.columns if c in base_df.columns and c != 'sample_id']
        if overlap:
            df_other = df_other.drop(columns=overlap)
        base_df = base_df.merge(df_other, on='sample_id', how='left')
        print(f"   -> Merged {name}; current shape: {base_df.shape}")

    # Downcast numeric dtypes for memory efficiency
    float_cols = base_df.select_dtypes(include=['float64']).columns
    int_cols = base_df.select_dtypes(include=['int64']).columns
    base_df[float_cols] = base_df[float_cols].astype(np.float32)
    base_df[int_cols] = base_df[int_cols].astype(np.int32)

    return base_df


# =============================================================================
# 3. FEATURE ENGINEERING
# =============================================================================
def engineer_features(df):
    """
    Creates new features from the existing data to improve model performance.
    """
    print("--- Stage 1: Engineering new features... ---")

    epsilon = 1e-6
    new_cols = {}
    if 'mass' in df.columns and 'volume' in df.columns:
        new_cols['density'] = df['mass'] / (df['volume'] + epsilon)
    if 'mass' in df.columns and 'pack_count' in df.columns:
        new_cols['mass_per_pack'] = df['mass'] / (df['pack_count'] + epsilon)
    if 'volume' in df.columns and 'pack_count' in df.columns:
        new_cols['volume_per_pack'] = df['volume'] / (df['pack_count'] + epsilon)

    flag_cols = [col for col in df.columns if col.startswith('flag_')]
    if len(flag_cols) > 0:
        new_cols['total_flags'] = df[flag_cols].sum(axis=1)

    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    print(f"   -> Feature engineering complete (density/mass_per_pack/volume_per_pack/total_flags if available)")
    print("--- Feature engineering complete. ---\n")
    return df


# =============================================================================
# 4. PREPROCESSING
# =============================================================================
def prepare_data(X_train, X_val, X_test):
    """
    Prepare data for LightGBM:
    - Cast selected categoricals to 'category' dtype
    - Fill missing numerics with 0.0 and categoricals with 'Unknown'
    """
    print("--- Stage 2: Preparing data (categoricals + NaN handling)... ---")

    # Identify candidate categorical columns
    categorical_cols = [c for c in ['brand', 'category'] if c in X_train.columns]

    for df in (X_train, X_val, X_test):
        # Categoricals
        for col in categorical_cols:
            df[col] = df[col].astype('category')

        # Missing values
        num_cols = df.select_dtypes(include=[np.number]).columns
        obj_cols = [c for c in df.columns if str(df[c].dtype) == 'category']
        df[num_cols] = df[num_cols].fillna(0.0)
        for c in obj_cols:
            df[c] = df[c].cat.add_categories(['Unknown']).fillna('Unknown')

    print("   -> Prepared categoricals and filled missing values.")
    print("--- Preparation complete. ---\n")
    return X_train, X_val, X_test


def build_monotone_constraints(feature_names: List[str]) -> List[int]:
    """
    Build monotone constraints vector aligned with feature order.
    Enforce price increases with: mass (g), volume (ml), pack_count.
    """
    positive = {'mass', 'volume', 'pack_count'}
    return [1 if f in positive else 0 for f in feature_names]


def run_kfold_oof(
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: List[str],
    base_params: dict,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple:
    """
    Perform KFold OOF training with early stopping and SMAPE monitoring on price-space.
    Returns (oof_df, cv_report_dict).
    """
    print(f"--- OOF CV: {n_splits}-fold ---")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    sample_ids = X.index.to_series().copy()  # will map back to df via index

    oof_pred_log = np.full(shape=(len(X),), fill_value=np.nan, dtype=float)
    fold_scores = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        params = base_params.copy()
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric=lgbm_smape_on_log_metric,
            callbacks=[lgb.early_stopping(300, verbose=False)]
        )

        pred_va_log = model.predict(X_va)
        pred_va_price = np.expm1(pred_va_log)
        y_va_price = np.expm1(y_va)
        fold_smape = smape(y_va_price.values, pred_va_price)
        fold_scores.append(float(fold_smape))
        oof_pred_log[va_idx] = pred_va_log
        print(f"   -> Fold {fold}/{n_splits} SMAPE: {fold_smape:.4f}%")

    # Overall OOF score on price-space
    oof_price = np.expm1(oof_pred_log)
    y_price = np.expm1(y)
    overall_smape = smape(y_price.values, oof_price)
    print(f"--- OOF SMAPE: {overall_smape:.4f}% | Folds mean {np.mean(fold_scores):.4f}% ± {np.std(fold_scores):.4f}% ---")

    # Build OOF dataframe aligned to original df rows via index
    oof_df = pd.DataFrame({
        'row_index': X.index,
        'sample_id': X.index,  # index is set to df.index (we'll replace later if needed)
        'oof_price': oof_price,
        'oof_log_price': oof_pred_log,
    })

    cv_report = {
        'n_splits': n_splits,
        'fold_smape_list': fold_scores,
        'fold_smape_mean': float(np.mean(fold_scores)),
        'fold_smape_std': float(np.std(fold_scores)),
        'oof_smape': float(overall_smape),
    }

    return oof_df, cv_report


# =============================================================================
# 5. HYPERPARAMETER TUNING WITH OPTUNA
# =============================================================================
def run_hyperparameter_tuning(X_train, y_train, X_val, y_val, feature_names: List[str]):
    """
    Uses Optuna to find the best hyperparameters for the LGBM Regressor, optimizing for SMAPE (price-space).
    Trains on log(price+1) with Huber objective (compatible with monotone constraints).
    """
    print("--- Stage 3: Hyperparameter Tuning (Optuna, minimize SMAPE on price) ---")

    monotone_vec = build_monotone_constraints(feature_names)
    prefer_gpu = os.environ.get('LGBM_FORCE_CPU', '0') != '1'

    # Subsample for HPO if enabled
    if HPO_SAMPLE_FRAC < 1.0:
        # Build a temporary training/val subset for faster HPO
        n_rows = len(X_train)
        take = max(HPO_MIN_ROWS, int(n_rows * HPO_SAMPLE_FRAC))
        X_train_hpo = X_train.iloc[:take]
        y_train_hpo = y_train.iloc[:take]
        X_val_hpo = X_val
        y_val_hpo = y_val
    else:
        X_train_hpo, y_train_hpo = X_train, y_train
        X_val_hpo, y_val_hpo = X_val, y_val

    def objective(trial):
        params = {
            'objective': 'huber',
            'metric': 'None',
            'n_estimators': HPO_N_EST,
            'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.06),
            'num_leaves': trial.suggest_int('num_leaves', 512, HPO_NUM_LEAVES_MAX),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 0.9),
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 32, 512),
            'max_depth': -1,
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': 42,
            'n_jobs': NUM_THREADS,
            'verbose': -1,
            # GPU settings
            'device': 'gpu' if prefer_gpu else 'cpu',
            'device_type': 'gpu' if prefer_gpu else 'cpu',
            'gpu_device_id': 0,
            'gpu_use_dp': False,
            'max_bin': MAX_BIN,
            'bin_construct_sample_cnt': BIN_CONSTRUCT,
            'feature_pre_filter': False,
            'monotone_constraints': monotone_vec,
        }

        model = lgb.LGBMRegressor(**params)
        try:
            model.fit(
                X_train_hpo, y_train_hpo,
                eval_set=[(X_val_hpo, y_val_hpo)],
                eval_metric=lgbm_smape_on_log_metric,
                callbacks=[lgb.early_stopping(EARLY_STOP_HPO, verbose=False)]
            )
        except Exception as gpu_err:
            if params['device_type'] == 'gpu':
                # Fallback to CPU seamlessly
                params['device_type'] = 'cpu'
                model = lgb.LGBMRegressor(**params)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    eval_metric=lgbm_smape_on_log_metric,
                    callbacks=[lgb.early_stopping(200, verbose=False)]
                )
            else:
                raise gpu_err

        preds_log = model.predict(X_val_hpo)
        preds_price = np.expm1(preds_log)
        y_val_price = np.expm1(y_val_hpo)
        return smape(y_val_price, preds_price)

    study = optuna.create_study(direction='minimize')
    if HPO_TIMEOUT > 0:
        study.optimize(objective, n_trials=HPO_TRIALS, timeout=HPO_TIMEOUT, show_progress_bar=True)
    else:
        study.optimize(objective, n_trials=HPO_TRIALS, show_progress_bar=True)

    print("--- Optuna complete ---")
    print(f"   -> Best validation SMAPE: {study.best_value:.4f}%")
    print("   -> Best parameters:")
    for k, v in study.best_params.items():
        print(f"      {k}: {v}")

    return study.best_params


# =============================================================================
# 6. MAIN EXECUTION SCRIPT
# =============================================================================
if __name__ == "__main__":
    print("--- Loading inputs from 'input/' ---")
    df = load_input_data('input')

    if 'price' not in df.columns:
        raise KeyError("Column 'price' not found in merged input data. Ensure one CSV includes price.")

    # Optional feature engineering
    df = engineer_features(df)

    # Define features and target (log price)
    TARGET = 'price'
    df['log_price'] = np.log1p(df[TARGET].astype(float))
    feature_cols = [c for c in df.columns if c not in ['sample_id', TARGET, 'log_price']]

    X = df[feature_cols]
    y = df['log_price']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    print("--- Data Splitting ---")
    print(f"   -> Training:   {X_train.shape}")
    print(f"   -> Validation: {X_val.shape}")
    print(f"   -> Test:       {X_test.shape}\n")

    # Prepare
    X_train, X_val, X_test = prepare_data(X_train.copy(), X_val.copy(), X_test.copy())

    # Save feature metadata for inference
    os.makedirs('output', exist_ok=True)
    categorical_cols = [c for c in X_train.columns if str(X_train[c].dtype) == 'category']
    categories_map = {c: [str(v) for v in X_train[c].cat.categories.tolist()] for c in categorical_cols}
    metadata = {
        'feature_cols': feature_cols,
        'categorical_cols': categorical_cols,
        'categories_map': categories_map,
    }
    with open(os.path.join('output', 'model_a_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f)
    print("   -> Saved feature metadata to output/model_a_metadata.json")

    # HPO
    best_params = run_hyperparameter_tuning(X_train, y_train, X_val, y_val, feature_cols)

    # Final training
    print("\n--- Stage 4: Training final model with best parameters... ---")
    monotone_vec = build_monotone_constraints(feature_cols)
    prefer_gpu = os.environ.get('LGBM_FORCE_CPU', '0') != '1'

    final_params = best_params.copy()
    final_params.update({
        'objective': 'huber',
        'metric': 'None',
        'n_estimators': max(8000, int(best_params.get('n_estimators', 6000))),
        'random_state': 42,
        'n_jobs': NUM_THREADS,
        'device': 'gpu' if prefer_gpu else 'cpu',
        'device_type': 'gpu' if prefer_gpu else 'cpu',
        'gpu_device_id': 0,
        'gpu_use_dp': False,
        'max_bin': MAX_BIN,
        'bin_construct_sample_cnt': BIN_CONSTRUCT,
        'feature_pre_filter': False,
        'monotone_constraints': monotone_vec,
        'verbose': -1,
    })

    final_model = lgb.LGBMRegressor(**final_params)
    try:
        final_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=lgbm_smape_on_log_metric,
            callbacks=[lgb.early_stopping(EARLY_STOP_FINAL, verbose=True)]
        )
    except Exception as gpu_err:
        if final_params['device_type'] == 'gpu':
            print(f"⚠️ GPU unavailable ({gpu_err}); falling back to CPU.")
            final_params['device_type'] = 'cpu'
            final_model = lgb.LGBMRegressor(**final_params)
            final_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric=lgbm_smape_on_log_metric,
                callbacks=[lgb.early_stopping(EARLY_STOP_FINAL, verbose=True)]
            )
        else:
            raise gpu_err

    # Evaluate on test (price-space SMAPE)
    print("\n--- Stage 5: Final Evaluation ---")
    test_preds_log = final_model.predict(X_test)
    test_preds_price = np.expm1(test_preds_log)
    y_test_price = np.expm1(y_test)
    final_smape = smape(y_test_price.values, test_preds_price)
    print(f"\n   -> Final SMAPE on Test Set: {final_smape:.4f}%\n")

    # OOF CV (using the full available labeled data)
    if SKIP_OOF:
        print("\n--- Stage 6: OOF Cross-Validation (skipped due to FAST_MODE) ---")
    else:
        print("\n--- Stage 6: OOF Cross-Validation ---")
    if not SKIP_OOF:
        # Build a full dataset from training splits
        X_all = pd.concat([X_train, X_val, X_test], axis=0)
        y_all = pd.concat([y_train, y_val, y_test], axis=0)
        # Ensure alignment and consistent order
        X_all = X_all.reindex(columns=feature_cols)
        X_all.index = df.loc[X_all.index, 'sample_id'] if 'sample_id' in df.columns else X_all.index
        base_params_for_oof = final_params.copy()
        # Slightly fewer estimators for faster OOF
        base_params_for_oof['n_estimators'] = max(2000 if FAST_MODE else 4000, int(final_params.get('n_estimators', 6000)))
        oof_df, cv_report = run_kfold_oof(X_all, y_all, feature_cols, base_params_for_oof, n_splits=OOF_SPLITS, random_state=42)
        # Map correct sample_id if present
        if 'sample_id' in df.columns:
            oof_df['sample_id'] = oof_df['row_index'].map(lambda idx: df.iloc[idx]['sample_id'] if isinstance(idx, (int, np.integer)) and idx < len(df) else idx)
        # Save OOF and CV report
        try:
            os.makedirs('output', exist_ok=True)
            oof_path = os.path.join('output', 'oof_predictions.csv')
            oof_df[['sample_id', 'oof_price', 'oof_log_price']].to_csv(oof_path, index=False)
            with open(os.path.join('output', 'cv_report.json'), 'w', encoding='utf-8') as f:
                json.dump(cv_report, f, indent=2)
            print(f"   -> Saved OOF predictions to {oof_path}")
            print(f"   -> Saved CV report to output/cv_report.json")
        except Exception as e:
            print(f"   -> Skipped OOF save: {e}")

    # Feature importance
    try:
        importance_df = pd.DataFrame({
            'feature': final_model.feature_name_,
            'importance_gain': final_model.feature_importances_,
        }).sort_values('importance_gain', ascending=False)
        os.makedirs('output', exist_ok=True)
        importance_path = os.path.join('output', 'feature_importance_report_model_a.xlsx')
        importance_df.to_excel(importance_path, index=False)
        print(f"   -> Saved feature importance to {importance_path}")
        print("   -> Top 10 features:")
        print(importance_df.head(10).to_string(index=False))
    except Exception as e:
        print(f"   -> Skipped importance export: {e}")

    # Save model
    try:
        os.makedirs('output', exist_ok=True)
        # Always save sklearn wrapper for robust inference
        model_pkl = os.path.join('output', 'lgbm_model_a.pkl')
        joblib.dump(final_model, model_pkl)
        print(f"   -> Saved model wrapper to {model_pkl}")
        # Additionally, export raw booster (optional)
        if hasattr(final_model, 'booster_') and final_model.booster_ is not None:
            booster_txt = os.path.join('output', 'lgbm_model_a.txt')
            final_model.booster_.save_model(booster_txt)
            print(f"   -> Saved booster to {booster_txt}")
    except Exception as e:
        print(f"   -> Skipped model save: {e}")

    # Clean up
    gc.collect()
    print("\n--- Script finished successfully. ---")