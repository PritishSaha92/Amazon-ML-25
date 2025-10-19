# 5_convert.py

import os
import glob
import json
from tqdm import tqdm
import argparse

try:
    import webdataset as wds
except ImportError:
    print("❌ webdataset not installed! Run: pip install webdataset")
    exit(1)

SAMPLES_PER_SHARD = 1000

def convert_split_to_wds(split_name):
    preprocessed_dir = f"./preprocessed_{split_name}"
    if not os.path.exists(preprocessed_dir):
        print(f"⚠️ Warning: Preprocessed directory not found for split '{split_name}' at {preprocessed_dir}. Skipping.")
        return 0
    
    input_ids_dir = os.path.join(preprocessed_dir, "input_ids")
    attention_mask_dir = os.path.join(preprocessed_dir, "attention_mask")
    pixel_values_dir = os.path.join(preprocessed_dir, "pixel_values")
    meta_dir = os.path.join(preprocessed_dir, "meta")
    
    output_dir = f"./webdataset_{split_name}"
    shard_pattern = os.path.join(output_dir, f"{split_name}-shard-%06d.tar")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*70}\nCONVERTING {split_name.upper()} TO WEBDATASET\n{'='*70}")
    
    input_ids_files = sorted(glob.glob(os.path.join(input_ids_dir, "*.npy")))
    
    if not input_ids_files:
        print(f"❌ No preprocessed files found in {input_ids_dir}. Cannot create WebDataset.")
        return 0
    
    print(f"Found {len(input_ids_files):,} samples. Creating TAR shards with {SAMPLES_PER_SHARD} samples each.")
    
    packed_count = 0
    with wds.ShardWriter(shard_pattern, maxcount=SAMPLES_PER_SHARD) as sink:
        for input_ids_path in tqdm(input_ids_files, desc=f"Packing {split_name}"):
            try:
                key = os.path.basename(input_ids_path).replace(".npy", "")
                attention_mask_path = os.path.join(attention_mask_dir, f"{key}.npy")
                pixel_values_path = os.path.join(pixel_values_dir, f"{key}.npy")
                meta_path = os.path.join(meta_dir, f"{key}.json")
                
                if not all(os.path.exists(p) for p in [attention_mask_path, pixel_values_path, meta_path]):
                    continue
                
                with open(input_ids_path, 'rb') as f_input, \
                     open(attention_mask_path, 'rb') as f_mask, \
                     open(pixel_values_path, 'rb') as f_pix, \
                     open(meta_path, 'r') as f_meta:
                    
                    sample = {
                        "__key__": key,
                        "input_ids.npy": f_input.read(),
                        "attention_mask.npy": f_mask.read(),
                        "pixel_values.npy": f_pix.read(),
                        "metadata.json": f_meta.read().encode('utf-8'),
                    }
                    
                    sink.write(sample)
                    packed_count += 1
            except Exception as e:
                tqdm.write(f"Failed to pack sample {key}: {e}")
                continue
    
    tar_files = sorted(glob.glob(os.path.join(output_dir, "*.tar")))
    print(f"\n--- WEBDATASET COMPLETE: {split_name.upper()} ---")
    print(f"✅ Created {len(tar_files)} TAR shard(s)")
    print(f"📦 Packed {packed_count:,} / {len(input_ids_files):,} samples")
    
    if tar_files:
        pattern = shard_pattern.replace('%06d', '{' + f'000000..{len(tar_files)-1:06d}' + '}')
        print(f"\n📝 Use this pattern in training: {pattern}")
    
    return packed_count

def main(args):
    global SAMPLES_PER_SHARD
    SAMPLES_PER_SHARD = args.samples_per_shard
    
    splits_to_process = []
    if args.split == "all":
        splits_to_process = ["train", "validation", "test"]
    elif args.split == "both":
        splits_to_process = ["train", "test"]
    else:
        splits_to_process = [args.split]
    
    summary = {}
    for split in splits_to_process:
        summary[split] = convert_split_to_wds(split)
    
    print(f"\n{'='*70}\nFINAL SUMMARY\n{'='*70}")
    for split, packed in summary.items():
        if packed > 0:
            print(f"{split.capitalize():<12}: {packed:,} samples packed")
    print(f"{'='*70}\n\n✅ Ready for training! Update paths in 6_main.ipynb.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert preprocessed tensors to WebDataset")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "validation", "test", "both", "all"],
                        help="Which split to convert")
    parser.add_argument("--samples-per-shard", type=int, default=1000,
                        help="Number of samples per TAR shard")
    args = parser.parse_args()
    main(args)
