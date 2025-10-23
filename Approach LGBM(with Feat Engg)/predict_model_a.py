import os
import json
import gc
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print("--- Engineering features on inference data... ---")
    epsilon = 1e-6
    if 'mass' in df.columns and 'volume' in df.columns:
        df['density'] = df['mass'] / (df['volume'] + epsilon)
    if 'mass' in df.columns and 'pack_count' in df.columns:
        df['mass_per_pack'] = df['mass'] / (df['pack_count'] + epsilon)
    if 'volume' in df.columns and 'pack_count' in df.columns:
        df['volume_per_pack'] = df['volume'] / (df['pack_count'] + epsilon)

    flag_cols = [c for c in df.columns if c.startswith('flag_')]
    if len(flag_cols) > 0:
        df['total_flags'] = df[flag_cols].sum(axis=1)

    print("   -> Feature engineering complete.\n")
    return df


def load_test_data(input_dir: str = 'input') -> pd.DataFrame:
    pq_path = os.path.join(input_dir, 'master_test.parquet')
    csv_path = os.path.join(input_dir, 'master_test.csv')
    if os.path.exists(pq_path):
        df = pd.read_parquet(pq_path)
        if 'sample_id' not in df.columns:
            raise KeyError("'sample_id' column missing in master_test.parquet")
    elif os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if 'sample_id' not in df.columns:
            raise KeyError("'sample_id' column missing in master_test.csv")
    else:
        raise FileNotFoundError(f"Test file not found: {pq_path} or {csv_path}")
    df['sample_id'] = df['sample_id'].astype(str)
    # Downcast numerics for memory
    float_cols = df.select_dtypes(include=['float64']).columns
    int_cols = df.select_dtypes(include=['int64']).columns
    if len(float_cols) > 0:
        df[float_cols] = df[float_cols].astype(np.float32)
    if len(int_cols) > 0:
        df[int_cols] = df[int_cols].astype(np.int32)
    print(f"--- Loaded test data: {df.shape} ---")
    return df


def prepare_inference_data(df: pd.DataFrame, feature_cols, categories_map) -> pd.DataFrame:
    print("--- Preparing inference data (align features and categories)... ---")

    # Ensure categorical columns exist and are cast with training categories
    for col, cats in categories_map.items():
        if col not in df.columns:
            df[col] = 'Unknown'
        df[col] = df[col].astype('category')
        # Ensure all saved categories exist; add missing
        current_cats = list(df[col].cat.categories)
        missing = [c for c in cats if c not in current_cats]
        if missing:
            df[col] = df[col].cat.add_categories(missing)
        df[col] = df[col].cat.set_categories(cats)
        df[col] = df[col].fillna('Unknown')

    # Fill missing numerics with 0.0
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0.0)

    # Create any missing features as 0 or 'Unknown' for categoricals
    for col in feature_cols:
        if col not in df.columns:
            if col in categories_map:
                df[col] = pd.Series(['Unknown'] * len(df), dtype='category')
                df[col] = df[col].cat.set_categories(categories_map[col])
            else:
                df[col] = 0.0

    # Drop extras and order columns
    df_out = df[feature_cols].copy()

    print("   -> Inference data prepared.\n")
    return df_out


if __name__ == "__main__":
    print("=== Model A Inference (LightGBM, Pseudo-Huber, price space) ===")

    # Load metadata and model
    meta_path = os.path.join('output', 'model_a_metadata.json')
    model_path = os.path.join('output', 'lgbm_model_a.pkl')
    model_txt_path = os.path.join('output', 'lgbm_model_a.txt')
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata not found: {meta_path}")
    if not (os.path.exists(model_path) or os.path.exists(model_txt_path)):
        raise FileNotFoundError(f"Model not found: {model_path} or {model_txt_path}")

    with open(meta_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    feature_cols = metadata['feature_cols']
    categories_map = metadata.get('categories_map', {})

    model = None
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
        except Exception:
            model = None
    if model is None and os.path.exists(model_txt_path):
        model = lgb.Booster(model_file=model_txt_path)
    print("--- Loaded model and metadata ---")

    # Load test data
    df_test = load_test_data('input')
    df_test = engineer_features(df_test)

    # Keep sample_id for output
    sample_ids = df_test['sample_id'].tolist()

    # Prepare features to align with training
    X_test = prepare_inference_data(df_test, feature_cols, categories_map)

    # Predict directly in price space
    preds_price = model.predict(X_test)
    preds_price = np.clip(preds_price, 0.0, None)

    # Save
    out_dir = 'dataset'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'sample_test_out.csv')
    pd.DataFrame({'sample_id': sample_ids, 'price': preds_price}).to_csv(out_path, index=False)
    print(f"--- Saved predictions to {out_path} ---")

    gc.collect()
    print("=== Inference complete ===")


