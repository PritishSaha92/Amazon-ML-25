#!/usr/bin/env python3
"""
Find and re-download all corrupted (0-byte) images with progress tracking
"""

import os
import pandas as pd
import urllib.request
from pathlib import Path
from tqdm import tqdm
import time

DATASET_DIR = "../dataset"
TRAIN_CSV = os.path.join(DATASET_DIR, "train.csv")
TEST_CSV = os.path.join(DATASET_DIR, "test.csv")
TRAIN_IMG_DIR = os.path.join(DATASET_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATASET_DIR, "test_images")
TRAIN_IMG_RESIZED_DIR = os.path.join(DATASET_DIR, "train_images_resized")
TEST_IMG_RESIZED_DIR = os.path.join(DATASET_DIR, "test_images_resized")


def find_corrupted_files(folder):
    """Find all 0-byte files in a folder"""
    corrupted = []
    if not os.path.exists(folder):
        return corrupted
    
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        if os.path.isfile(filepath) and os.path.getsize(filepath) == 0:
            corrupted.append(filename)
    
    return sorted(corrupted)


def get_image_url_from_csv(csv_path, filename):
    """Find the URL for a given filename in CSV"""
    df = pd.read_csv(csv_path)
    
    for _, row in df.iterrows():
        image_link = row.get("image_link", "")
        if isinstance(image_link, str) and filename in image_link:
            return image_link
    
    return None


def download_with_retry(url, save_path, max_retries=3):
    """Download with retries and proper headers"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as response:
                data = response.read()
                
                if len(data) > 0:
                    with open(save_path, 'wb') as f:
                        f.write(data)
                    
                    # Verify file size
                    if os.path.getsize(save_path) > 0:
                        return True, len(data)
                    else:
                        return False, "Saved file is 0 bytes"
            
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False, "404 Not Found"
            return False, f"HTTP Error {e.code}"
        except urllib.error.URLError as e:
            if attempt < max_retries - 1:
                time.sleep(2)
            return False, f"URL Error: {e.reason}"
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
            return False, str(e)
    
    return False, "Max retries exceeded"


def repair_images(corrupted_list, csv_path, img_dir, img_resized_dir, split_name):
    """Repair a list of corrupted images"""
    if not corrupted_list:
        return 0, []
    
    print(f"\n🔄 Repairing {len(corrupted_list)} {split_name} images...")
    
    success_count = 0
    failed = []
    permanent_404 = []
    
    for filename in tqdm(corrupted_list, desc=f"Repairing {split_name}"):
        url = get_image_url_from_csv(csv_path, filename)
        
        if not url:
            tqdm.write(f"   ⚠️  No URL found: {filename}")
            failed.append((filename, "No URL in CSV"))
            continue
        
        # Delete old corrupted files
        corrupted_path = os.path.join(img_dir, filename)
        corrupted_resized_path = os.path.join(img_resized_dir, filename)
        
        if os.path.exists(corrupted_path):
            os.remove(corrupted_path)
        if os.path.exists(corrupted_resized_path):
            os.remove(corrupted_resized_path)
        
        # Re-download
        success, result = download_with_retry(url, corrupted_path)
        
        if success:
            success_count += 1
        else:
            if "404" in str(result):
                permanent_404.append(filename)
            else:
                failed.append((filename, result))
    
    print(f"\n   ✅ Successfully repaired: {success_count}/{len(corrupted_list)}")
    
    if permanent_404:
        print(f"\n   🚫 Permanent 404 errors (image deleted from Amazon): {len(permanent_404)}")
        if len(permanent_404) <= 10:
            for f in permanent_404:
                print(f"      - {f}")
    
    if failed:
        print(f"\n   ❌ Failed (network/timeout): {len(failed)}")
        if len(failed) <= 5:
            for f, err in failed:
                print(f"      - {f}: {err}")
    
    return success_count, permanent_404


def main():
    print("="*70)
    print("CORRUPTED IMAGE REPAIR TOOL")
    print("="*70)
    
    # Find all corrupted files
    print("\n🔍 Scanning for corrupted (0-byte) images...")
    train_corrupted = find_corrupted_files(TRAIN_IMG_DIR)
    test_corrupted = find_corrupted_files(TEST_IMG_DIR)
    
    print(f"\n📊 Found:")
    print(f"   Train: {len(train_corrupted)} corrupted images")
    print(f"   Test:  {len(test_corrupted)} corrupted images")
    print(f"   Total: {len(train_corrupted) + len(test_corrupted)} corrupted images")
    
    if len(train_corrupted) == 0 and len(test_corrupted) == 0:
        print("\n✅ No corrupted images found!")
        return
    
    input("\n⏸️  Press ENTER to start repair process...")
    
    # Repair train images
    train_success = 0
    train_404 = []
    if train_corrupted:
        train_success, train_404 = repair_images(
            train_corrupted, TRAIN_CSV, TRAIN_IMG_DIR, TRAIN_IMG_RESIZED_DIR, "train"
        )
    
    # Repair test images
    test_success = 0
    test_404 = []
    if test_corrupted:
        test_success, test_404 = repair_images(
            test_corrupted, TEST_CSV, TEST_IMG_DIR, TEST_IMG_RESIZED_DIR, "test"
        )
    
    # Final summary
    print("\n" + "="*70)
    print("REPAIR COMPLETE")
    print("="*70)
    print(f"✅ Successfully repaired: {train_success + test_success} images")
    print(f"🚫 Permanent 404s: {len(train_404) + len(test_404)} images")
    print(f"   (These images are deleted from Amazon's servers)")
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Run: python 2_resize.py")
    print("   (to resize the newly downloaded images)")
    print("")
    print("2. Delete old preprocessed data:")
    print("   rm -rf ./preprocessed_train ./preprocessed_validation")
    print("")
    print("3. Re-run preprocessing from step 3:")
    print("   python 3_create_jsonl.py --validation-split-ratio 0.1")
    print("   python 4_preprocess.py --split all")
    print("   python 5_convert.py --split all")
    print("="*70)
    
    # Verify one more time
    print("\n🔍 Final verification...")
    train_still_corrupted = find_corrupted_files(TRAIN_IMG_DIR)
    test_still_corrupted = find_corrupted_files(TEST_IMG_DIR)
    
    if len(train_still_corrupted) == 0 and len(test_still_corrupted) == 0:
        print("✅ All corrupted files fixed!")
    else:
        print(f"⚠️  Still {len(train_still_corrupted) + len(test_still_corrupted)} corrupted files")
        print("   (These are likely permanent 404s)")


if __name__ == "__main__":
    main()
