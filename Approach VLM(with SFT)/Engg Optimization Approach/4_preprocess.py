# 4_preprocess.py

import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor
import torch
import argparse

# --- Configuration ---
DATASET_DIR = "../dataset"
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 512 * 28 * 28

def get_jsonl_path(split_name):
    """Determines the correct input JSONL file path based on split name."""
    split_path = os.path.join(DATASET_DIR, f"{split_name}_split.jsonl")
    if os.path.exists(split_path):
        return split_path
    return os.path.join(DATASET_DIR, f"{split_name}.jsonl")

def preprocess_split(split_name, processor):
    jsonl_path = get_jsonl_path(split_name)
    if not os.path.exists(jsonl_path):
        print(f"⚠️ Warning: JSONL file not found for split '{split_name}' at {jsonl_path}. Skipping.")
        return 0, 0
    
    output_dir = f"./preprocessed_{split_name}"
    input_ids_dir = os.path.join(output_dir, "input_ids")
    attention_mask_dir = os.path.join(output_dir, "attention_mask")
    pixel_values_dir = os.path.join(output_dir, "pixel_values")
    meta_dir = os.path.join(output_dir, "meta")
    
    os.makedirs(input_ids_dir, exist_ok=True)
    os.makedirs(attention_mask_dir, exist_ok=True)
    os.makedirs(pixel_values_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)
    
    print(f"\n{'='*70}\nPREPROCESSING {split_name.upper()} SPLIT (WITH ACTUAL TENSORS!)\n{'='*70}")
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    print(f"Processing {total_lines} samples from {os.path.basename(jsonl_path)}...")
    
    success_count, failed_count = 0, 0
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(tqdm(f, total=total_lines, desc=f"Processing {split_name}")):
            try:
                example = json.loads(line)
                messages = example["messages"]
                
                # Find image path
                image_path = next((
                    item["image"]
                    for msg in messages
                    if isinstance(msg["content"], list)
                    for item in msg["content"]
                    if isinstance(item, dict) and item.get("type") == "image"
                ), None)
                
                if not image_path or not os.path.exists(image_path):
                    failed_count += 1
                    continue
                
                # Load image
                with Image.open(image_path).convert("RGB") as image:
                    # Process with processor (THIS IS THE KEY!)
                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                    
                    inputs = processor(
                        text=[text],
                        images=[image],
                        return_tensors="pt",
                        padding=False,
                    )
                    
                    output_key = f"{idx:06d}"
                    
                    # Save ACTUAL TENSORS (not raw data!)
                    np.save(
                        os.path.join(input_ids_dir, f"{output_key}.npy"),
                        inputs["input_ids"][0].numpy()  # [batch=1, seq] → [seq]
                    )
                    np.save(
                        os.path.join(attention_mask_dir, f"{output_key}.npy"),
                        inputs["attention_mask"][0].numpy()  # [batch=1, seq] → [seq]
                    )
                    np.save(
                        os.path.join(pixel_values_dir, f"{output_key}.npy"),
                        inputs["pixel_values"].numpy()  # ✅ NO [0]! Already [patches, dim]
                    )
                    
                    # Save metadata with shapes
                    meta_data = {
                        "sample_id": example["sample_id"],
                        "original_index": idx,
                        "split": split_name,
                        "input_ids_shape": list(inputs["input_ids"][0].shape),      # [seq_len]
                        "pixel_values_shape": list(inputs["pixel_values"].shape),   # ✅ [patches, dim]
                        "image_grid_thw": inputs["image_grid_thw"][0].tolist(),    # [3]
                    }
                    
                    with open(os.path.join(meta_dir, f"{output_key}.json"), 'w') as f_meta:
                        json.dump(meta_data, f_meta)
                    
                    success_count += 1
                    
            except Exception as e:
                tqdm.write(f"Failed to process sample {idx}: {e}")
                failed_count += 1
                continue
    
    print(f"\n--- PREPROCESSING COMPLETE: {split_name.upper()} ---")
    print(f"✅ Success: {success_count:,} | ❌ Failed: {failed_count:,} | 📁 Output: {output_dir}")
    return success_count, failed_count

def main(args):
    print(f"Loading processor from {MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
        trust_remote_code=True
    )
    
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    
    splits_to_process = []
    if args.split == "all":
        splits_to_process = ["train", "validation", "test"]
    elif args.split == "both":
        splits_to_process = ["train", "test"]
    else:
        splits_to_process = [args.split]
    
    summary = {}
    for split in splits_to_process:
        summary[split] = preprocess_split(split, processor)
    
    print(f"\n{'='*70}\nFINAL SUMMARY\n{'='*70}")
    for split, (success, failed) in summary.items():
        if success > 0 or failed > 0:
            print(f"{split.capitalize():<12}: {success:,} success, {failed:,} failed")
    print(f"{'='*70}\n\n✅ Next: Run 5_convert.py")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess VLM data splits WITH TENSORS")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "validation", "test", "both", "all"],
                        help="Which split to process")
    args = parser.parse_args()
    main(args)
