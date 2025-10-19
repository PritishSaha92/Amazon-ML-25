import os
import pandas as pd
import json
from urllib.parse import urlparse
import argparse

DATASET_DIR = "../dataset"
TRAIN_CSV = os.path.join(DATASET_DIR, "train.csv")
TEST_CSV = os.path.join(DATASET_DIR, "test.csv")
TRAIN_IMAGES_RESIZED = os.path.join(DATASET_DIR, "train_images_resized")
TEST_IMAGES_RESIZED = os.path.join(DATASET_DIR, "test_images_resized")
TRAIN_IMAGES = os.path.join(DATASET_DIR, "train_images")
TEST_IMAGES = os.path.join(DATASET_DIR, "test_images")


def get_filename_from_url(url):
    if not isinstance(url, str):
        return None
    try:
        return os.path.basename(urlparse(url).path)
    except:
        return None


def create_jsonl_from_df(df, image_folder_resized, image_folder_raw, output_path, is_test=False):
    print(f"Processing {len(df)} rows into {os.path.basename(output_path)}...")
    
    system_prompt = (
        "You are a precise e-commerce expert. Your task is to analyze the product "
        "image and its description, then output only the product's price as a numerical "
        "value. Do not add any other text or currency symbols."
    )
    
    written_count = 0
    skipped_count = 0
    
    with open(output_path, "w") as f:
        for _, row in df.iterrows():
            image_filename = get_filename_from_url(row["image_link"])
            if not image_filename:
                skipped_count += 1
                continue
            
            image_path = os.path.join(image_folder_resized, image_filename)
            if not os.path.exists(image_path):
                image_path = os.path.join(image_folder_raw, image_filename)
            
            image_exists = os.path.exists(image_path)
            if not image_exists and not is_test:
                skipped_count += 1
                continue
            
            image_path = os.path.abspath(image_path)
            product_description = str(row["catalog_content"]).strip()
            user_prompt_text = (
                f"Analyze the following product and provide its price.\n\n"
                f"**Product Information:**\n{product_description}"
            )
            
            messages = [{"role": "system", "content": system_prompt}]
            
            if image_exists:
                user_content = [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": user_prompt_text},
                ]
            else:
                user_content = [{"type": "text", "text": user_prompt_text}]
            
            messages.append({"role": "user", "content": user_content})
            
            if not is_test:
                price = f"{round(float(row['price']), 2):.2f}"
                messages.append({"role": "assistant", "content": price})
            
            jsonl_entry = {"sample_id": row["sample_id"], "messages": messages}
            f.write(json.dumps(jsonl_entry) + "\n")
            written_count += 1
    
    print(f"✅ Successfully created {output_path}")
    print(f"   Written: {written_count:,} samples | Skipped: {skipped_count:,} samples")
    return written_count


def main(args):
    print("="*70)
    print("STEP 3: CREATING JSONL FILES")
    print("="*70)
    
    # Clean up old split files if validation ratio changed
    old_files = [
        os.path.join(DATASET_DIR, "train_split.jsonl"),
        os.path.join(DATASET_DIR, "validation_split.jsonl")
    ]
    for old_file in old_files:
        if os.path.exists(old_file):
            os.remove(old_file)
            print(f"🗑️  Removed old split file: {old_file}")
    
    # Process test set (always the same)
    print("\nCreating test.jsonl...")
    test_df = pd.read_csv(TEST_CSV)
    create_jsonl_from_df(
        test_df, 
        TEST_IMAGES_RESIZED, 
        TEST_IMAGES, 
        os.path.join(DATASET_DIR, "test.jsonl"), 
        is_test=True
    )
    
    # Process training data with optional validation split
    print("\nProcessing training data...")
    full_train_df = pd.read_csv(TRAIN_CSV)
    
    if args.validation_split_ratio > 0:
        print(f"Splitting training data with validation ratio: {args.validation_split_ratio}")
        
        shuffled_df = full_train_df.sample(frac=1, random_state=42)
        split_index = int(len(shuffled_df) * (1 - args.validation_split_ratio))
        train_df = shuffled_df[:split_index]
        validation_df = shuffled_df[split_index:]
        
        print(f"Training samples: {len(train_df):,}")
        print(f"Validation samples: {len(validation_df):,}")
        
        train_count = create_jsonl_from_df(
            train_df, 
            TRAIN_IMAGES_RESIZED, 
            TRAIN_IMAGES, 
            os.path.join(DATASET_DIR, "train_split.jsonl")
        )
        val_count = create_jsonl_from_df(
            validation_df, 
            TRAIN_IMAGES_RESIZED, 
            TRAIN_IMAGES, 
            os.path.join(DATASET_DIR, "validation_split.jsonl")
        )
    else:
        print("No validation split. Using full training dataset.")
        train_count = create_jsonl_from_df(
            full_train_df, 
            TRAIN_IMAGES_RESIZED, 
            TRAIN_IMAGES, 
            os.path.join(DATASET_DIR, "train.jsonl")
        )
    
    print("\n✅ JSONL creation complete!")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create JSONL datasets with optional validation split")
    parser.add_argument(
        "--validation-split-ratio",
        type=float,
        default=0.0,
        help="Fraction of training data for validation (e.g., 0.1 for 10%%). Default: 0.0"
    )
    args = parser.parse_args()
    main(args)
