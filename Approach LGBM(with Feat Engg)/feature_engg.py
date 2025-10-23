# import pandas as pd
# import numpy as np
# import re
# import os
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.decomposition import TruncatedSVD

# # =============================================================================
# # Helper Function for Cleaning Brand Names
# # =============================================================================
# def clean_brand_name(brand):
#     """Cleans brand names by keeping only lowercase letters."""
#     if not isinstance(brand, str):
#         return ""
#     return re.sub(r'[^a-zA-Z]', '', brand).lower()

# # =============================================================================
# # Main Feature Engineering Pipeline for Training Data
# # =============================================================================
# def feature_engineer_train_data(
#     train_data_path,
#     price_data_path, # Price is a separate file for training
#     kmeans_path,
#     embeddings_path,
#     output_path
# ):
#     """
#     Applies the full feature engineering pipeline to the training data.
#     """
#     print("--- Starting Feature Engineering Pipeline for Training Data ---")

#     # --- 1. Load All Input Files ---
#     print("Step 1: Loading all data files...")
#     try:
#         df_train = pd.read_csv(train_data_path)
#         df_price = pd.read_csv(price_data_path)
#         df_kmeans = pd.read_csv(kmeans_path)

#         with np.load(embeddings_path) as data:
#             embedding_ids = data['sample_id']
#             embeddings = data['embedding']
#         df_embeddings = pd.DataFrame({'sample_id': embedding_ids, 'image_embedding': list(embeddings)})
#     except FileNotFoundError as e:
#         print(f"❌ ERROR: A file was not found. Details: {e}")
#         return
    
#     # --- 2. Merge All Data Sources ---
#     print("Step 2: Merging price, K-Means clusters, and image embeddings...")
#     df_train = pd.merge(df_train, df_price[['sample_id', 'price']], on='sample_id', how='left')
#     df_train = pd.merge(df_train, df_kmeans[['sample_id', 'text_kmeans_cluster_id']], on='sample_id', how='left')
#     df_train = pd.merge(df_train, df_embeddings, on='sample_id', how='left')
#     df_train.dropna(subset=['price'], inplace=True)
    
#     print(f"  -> Merged training data shape: {df_train.shape}")

#     # --- 3. Create Keyword-Based Features ---
#     print("Step 3: Creating keyword features from flags...")
#     flag_columns = [col for col in df_train.columns if col.lower().startswith('flag_')]
#     df_train['keywords'] = [[col.replace('flag_', '') for col in flag_columns if row[col] == 1] for _, row in df_train.iterrows()]
#     df_train['num_keywords'] = df_train['keywords'].apply(len)
    
#     important_keywords = ['organic', 'premium', 'gourmet', 'vegan', 'natural', 'keto']
#     for keyword in important_keywords:
#         df_train[f'is_{keyword}'] = df_train['keywords'].apply(lambda kw_list: 1 if keyword in kw_list else 0)

#     # --- 4. Brand Similarity Encoding (Fit and Transform on Train) ---
#     print("Step 4: Creating brand similarity features...")
#     vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4))
#     svd = TruncatedSVD(n_components=20, random_state=42)

#     train_brands = df_train['brand'].fillna('unknown').astype(str)
#     train_embeddings = svd.fit_transform(vectorizer.fit_transform(train_brands))
#     embedding_df = pd.DataFrame(train_embeddings, columns=[f'brand_sim_{i}' for i in range(20)], index=df_train.index)
#     df_train = pd.concat([df_train, embedding_df], axis=1)

#     # --- 5. Brand & Category Statistics (Calculate and Apply on Train) ---
#     print("Step 5: Calculating and merging brand and category statistics...")
#     df_train['brand_cleaned'] = df_train['brand'].apply(clean_brand_name)

#     brand_stats = df_train.groupby('brand_cleaned')['price'].agg(['mean', 'std']).reset_index()
#     brand_stats.columns = ['brand_cleaned', 'brand_avg_price', 'brand_std_price']
#     brand_stats['brand_std_price'] = brand_stats['brand_std_price'].fillna(0)

#     category_stats = df_train.groupby('category')['price'].agg(['mean', 'std']).reset_index()
#     category_stats.columns = ['category', 'category_avg_price', 'category_std_price']
#     category_stats['category_std_price'] = category_stats['category_std_price'].fillna(0)

#     df_train = pd.merge(df_train, brand_stats, on='brand_cleaned', how='left')
#     df_train = pd.merge(df_train, category_stats, on='category', how='left')

#     # --- 6. Expand Image Embeddings ---
#     print("Step 6: Expanding image embedding vectors into columns...")
#     df_train['image_embedding'] = df_train['image_embedding'].apply(
#         lambda x: [0.0] * 768 if not isinstance(x, list) else x
#     )
#     embedding_df_img = pd.DataFrame(df_train['image_embedding'].tolist(), index=df_train.index)
#     embedding_df_img.columns = [f'img_embedding_{i}' for i in range(768)]
#     df_train = pd.concat([df_train, embedding_df_img], axis=1)

#     # --- 7. Final Cleanup and Save ---
#     print("Step 7: Cleaning up and saving the final file...")
#     columns_to_drop = ['brand', 'category', 'brand_cleaned', 'keywords', 'image_embedding']
#     df_train.drop(columns=columns_to_drop, inplace=True, errors='ignore')

#     # --- THIS LINE IS THE FIX: The os.makedirs line has been removed ---
#     df_train.to_parquet(output_path, index=False)
    
#     print("\n✅ --- Pipeline Complete ---")
#     print(f"   -> Final feature-engineered file saved to: {output_path}")
#     print(f"   -> Final shape: {df_train.shape}")

# # =============================================================================
# # How to Run the Script
# # =============================================================================
# if __name__ == "__main__":
    
#     # ❗ DEFINE YOUR FILE PATHS HERE ❗
#     TRAIN_DATA_CSV = 'output/vlm_structured_data_train.csv'
#     PRICE_DATA_CSV = 'dataset/train.csv'
#     KMEANS_CLUSTERS_CSV = 'output/vlm_structured_data_train.clustered_k300.csv'
#     IMAGE_EMBEDDINGS_NPZ = 'cache_new/clip_vit_l14_image_train.npz'
#     OUTPUT_PARQUET = 'master_train.parquet'
    
#     # Run the full pipeline
#     feature_engineer_train_data(
#         train_data_path=TRAIN_DATA_CSV,
#         price_data_path=PRICE_DATA_CSV,
#         kmeans_path=KMEANS_CLUSTERS_CSV,
#         embeddings_path=IMAGE_EMBEDDINGS_NPZ,
#         output_path=OUTPUT_PARQUET
#     )


import pandas as pd
import numpy as np
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# =============================================================================
# Helper Function for Cleaning Brand Names
# =============================================================================
def clean_brand_name(brand):
    """Cleans brand names by keeping only lowercase letters."""
    if not isinstance(brand, str):
        return ""
    return re.sub(r'[^a-zA-Z]', '', brand).lower()

# =============================================================================
# Main Feature Engineering Pipeline (No Price Data)
# =============================================================================
def feature_engineer_train_data(
    train_data_path,
    kmeans_path,
    embeddings_path,
    output_path
):
    """
    Applies the feature engineering pipeline without using an external price file.
    """
    print("--- Starting Feature Engineering Pipeline for Training Data ---")

    # --- 1. Load All Input Files ---
    print("Step 1: Loading all data files...")
    try:
        df_train = pd.read_csv(train_data_path)
        df_kmeans = pd.read_csv(kmeans_path)

        with np.load(embeddings_path) as data:
            embedding_ids = data['sample_id']
            embeddings = data['embedding']
        # Ensure consistent dtypes and convert embedding rows to Python lists
        emb_dim = int(embeddings.shape[1]) if embeddings.ndim == 2 else 768
        df_embeddings = pd.DataFrame({
            'sample_id': embedding_ids.astype('int64'),
            'image_embedding': [row.astype(np.float32).tolist() for row in embeddings]
        })
    except FileNotFoundError as e:
        print(f"❌ ERROR: A file was not found. Details: {e}")
        return
    
    # --- 2. Merge External Data Sources ---
    print("Step 2: Merging K-Means clusters and image embeddings...")
    # Harmonize key dtype prior to merge to avoid join misses
    df_train['sample_id'] = df_train['sample_id'].astype('int64')
    df_kmeans['sample_id'] = df_kmeans['sample_id'].astype('int64')
    df_train = pd.merge(df_train, df_kmeans[['sample_id', 'text_kmeans_cluster_id']], on='sample_id', how='left')
    df_train = pd.merge(df_train, df_embeddings, on='sample_id', how='left')
    
    print(f"  -> Merged training data shape: {df_train.shape}")

    # --- 3. Create Keyword-Based Features ---
    print("Step 3: Creating keyword features from flags...")
    flag_columns = [col for col in df_train.columns if col.lower().startswith('flag_')]
    df_train['keywords'] = [[col.replace('flag_', '') for col in flag_columns if row[col] == 1] for _, row in df_train.iterrows()]
    df_train['num_keywords'] = df_train['keywords'].apply(len)
    
    important_keywords = ['organic', 'premium', 'gourmet', 'vegan', 'natural', 'keto']
    for keyword in important_keywords:
        df_train[f'is_{keyword}'] = df_train['keywords'].apply(lambda kw_list: 1 if keyword in kw_list else 0)

    # --- 4. Brand Similarity Encoding (Fit and Transform on Train) ---
    print("Step 4: Creating brand similarity features...")
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4))
    svd = TruncatedSVD(n_components=20, random_state=42)

    train_brands = df_train['brand'].fillna('unknown').astype(str)
    train_embeddings = svd.fit_transform(vectorizer.fit_transform(train_brands))
    embedding_df = pd.DataFrame(train_embeddings, columns=[f'brand_sim_{i}' for i in range(20)], index=df_train.index)
    df_train = pd.concat([df_train, embedding_df], axis=1)

    # --- 5. Expand Image Embeddings ---
    print("Step 5: Expanding image embedding vectors into columns...")
    # Convert any numpy arrays to lists; fill missing with zeros of correct dim
    def _to_list_or_zeros(v):
        if isinstance(v, list):
            return v
        if isinstance(v, np.ndarray):
            return v.tolist()
        return [0.0] * emb_dim
    df_train['image_embedding'] = df_train['image_embedding'].apply(_to_list_or_zeros)
    embedding_df_img = pd.DataFrame(df_train['image_embedding'].tolist(), index=df_train.index)
    embedding_df_img.columns = [f'img_embedding_{i}' for i in range(768)]
    df_train = pd.concat([df_train, embedding_df_img], axis=1)

    # --- 6. Final Cleanup and Save ---
    print("Step 6: Cleaning up and saving the final file...")
    # NOTE: brand_cleaned is no longer created, so it's removed from the drop list.
    columns_to_drop = ['brand', 'category', 'keywords', 'image_embedding']
    df_train.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    df_train.to_parquet(output_path, index=False)
    
    print("\n✅ --- Pipeline Complete ---")
    print(f"   -> Final feature-engineered file saved to: {output_path}")
    print(f"   -> Final shape: {df_train.shape}")


# =============================================================================
# How to Run the Script
# =============================================================================
if __name__ == "__main__":
    
    # ❗ DEFINE YOUR FILE PATHS HERE ❗
    TRAIN_DATA_CSV = 'output/vlm_structured_data_test.csv'
    KMEANS_CLUSTERS_CSV = 'output/vlm_structured_data_test.clustered_k300.csv'
    IMAGE_EMBEDDINGS_NPZ = 'cache_new/clip_vit_l14_image_test.npz'
    OUTPUT_PARQUET = 'master_test.parquet'
    
    # Run the full pipeline
    feature_engineer_train_data(
        train_data_path=TRAIN_DATA_CSV,
        kmeans_path=KMEANS_CLUSTERS_CSV,
        embeddings_path=IMAGE_EMBEDDINGS_NPZ,
        output_path=OUTPUT_PARQUET
    )