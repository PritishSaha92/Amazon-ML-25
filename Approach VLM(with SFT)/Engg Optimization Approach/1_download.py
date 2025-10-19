import os
import pandas as pd
from pathlib import Path
from functools import partial
import urllib.request
import multiprocessing
from tqdm import tqdm

DATASET_DIR = "../dataset"
TRAIN_CSV = os.path.join(DATASET_DIR, "train.csv")
TEST_CSV = os.path.join(DATASET_DIR, "test.csv")
TRAIN_IMG_DIR = os.path.join(DATASET_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATASET_DIR, "test_images")


def download_image(image_link, savefolder):
    if isinstance(image_link, str):
        filename = Path(image_link).name
        image_save_path = os.path.join(savefolder, filename)
        if not os.path.exists(image_save_path):
            try:
                urllib.request.urlretrieve(image_link, image_save_path)
            except Exception as ex:
                print(f"Warning: Not able to download - {image_link}\n{ex}")
    return


def download_images(image_links, download_folder):
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)
    download_image_partial = partial(download_image, savefolder=download_folder)
    with multiprocessing.Pool(100) as pool:
        for _ in tqdm(pool.imap(download_image_partial, image_links), total=len(image_links)):
            pass
        pool.close()
        pool.join()


def main():
    print("="*70)
    print("STEP 1: DOWNLOADING IMAGES")
    print("="*70)
    
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    
    train_links = train_df["image_link"].dropna().unique().tolist()
    test_links = test_df["image_link"].dropna().unique().tolist()
    
    print(f"\nDownloading {len(train_links)} unique train images...")
    download_images(train_links, TRAIN_IMG_DIR)
    
    print(f"\nDownloading {len(test_links)} unique test images...")
    download_images(test_links, TEST_IMG_DIR)
    
    print("\n✅ Image download complete!")
    print("="*70)

if __name__ == "__main__":
    main()
