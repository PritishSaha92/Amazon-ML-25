import pandas as pd
import os

# --- 1. Configuration ---
# The base master file to which we will add data.
base_master_file = 'input/master_train.parquet'

# The file containing the new features to merge into the master file.
new_features_file = 'output/vlm_plus_tfidf_svd_train.parquet'

# The final output path will overwrite the original master file.
output_path = 'input/master_train.parquet'

# --- 2. Main Script ---
try:
    # Load the existing master file and the new features file
    print(f"Reading base master file: {base_master_file}...")
    df_master = pd.read_parquet(base_master_file)

    print(f"Reading new features file: {new_features_file}...")
    df_new_features = pd.read_parquet(new_features_file)
    
    # Merge the new features into the master DataFrame
    print("\nMerging new features into the master file on 'sample_id'...")
    # Using a left merge to ensure all original rows from the master file are kept.
    updated_df = pd.merge(df_master, df_new_features, on='sample_id', how='left')

    # --- Drop specified columns ---
    columns_to_drop = ['brand', 'category']
    print(f"Dropping columns if they exist: {columns_to_drop}...")
    updated_df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    # Save the final combined DataFrame, overwriting the original master file
    print(f"Saving the updated master file to {output_path}...")
    updated_df.to_parquet(output_path, index=False)

    print("\n✅ Success! The master training file has been updated with new features.")
    print(f"   -> Final output saved to: {output_path}")
    print(f"   -> Final shape of file: {updated_df.shape}")

except FileNotFoundError as e:
    print(f"❌ ERROR: An input file was not found.")
    print(f"   -> Details: {e}")
    print("   -> Please ensure 'input/master_train.parquet' and 'output/vlm_plus_tfidf_svd_train.parquet' exist.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")