import os
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import time
from urllib.parse import urlparse

# --- Configuration for Local Environment ---
# Assumes your CSV files are in a 'dataset' subfolder
TRAIN_CSV_PATH = 'dataset/train.csv'
TEST_CSV_PATH = 'dataset/test.csv'

# --- MODIFICATION: Define separate subfolders for train and test images ---
TRAIN_IMAGE_FOLDER = 'images/train/'
TEST_IMAGE_FOLDER = 'images/test/'
# --- END MODIFICATION ---

# --- Worker Function to Download a Single Image (No changes needed here) ---
def download_image_worker(args):
    """
    Worker function to download an image and save it with its sample_id.
    Accepts a tuple (sample_id, image_link, save_folder).
    """
    sample_id, image_link, save_folder = args
    max_retries = 3
    retry_delay = 2  # seconds

    if not isinstance(image_link, str) or not image_link.startswith('http'):
        return # Skip invalid links

    try:
        path = urlparse(image_link).path
        extension = os.path.splitext(path)[1]
        if not extension: # Default to .jpg if no extension found
            extension = '.jpg'
        
        filename = f"{sample_id}{extension}"
        save_path = os.path.join(save_folder, filename)

        if os.path.exists(save_path):
            return

        for attempt in range(max_retries):
            try:
                response = requests.get(image_link, stream=True, timeout=15)
                response.raise_for_status()

                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return # Success

            except requests.exceptions.RequestException:
                time.sleep(retry_delay * (attempt + 1)) # Exponential backoff

    except Exception:
        pass


# --- Main Function to Manage Downloads (No changes needed here) ---
def download_all_images(image_data, download_folder):
    """
    Downloads all images from a list of (sample_id, link) tuples
    into a specified folder.
    """
    print(f"Starting download of {len(image_data)} images to '{download_folder}'")

    # Create the target directory if it doesn't exist
    os.makedirs(download_folder, exist_ok=True)

    # Prepare arguments for the worker function
    tasks = [(sample_id, link, download_folder) for sample_id, link in image_data]

    # Use a thread pool to download images in parallel
    with ThreadPoolExecutor(max_workers=32) as executor:
        list(tqdm(executor.map(download_image_worker, tasks), total=len(tasks)))

    print(f"\nFinished downloading to '{download_folder}'.")


# --- Main Execution Block ---
if __name__ == "__main__":
    try:
        train_df = pd.read_csv(TRAIN_CSV_PATH)
        test_df = pd.read_csv(TEST_CSV_PATH)

        # --- MODIFICATION: Process and download train and test data separately ---

        # 1. Process and download TRAINING images
        train_data = train_df[['sample_id', 'image_link']].dropna(subset=['image_link'])
        train_image_list = list(train_data.itertuples(index=False, name=None))
        download_all_images(train_image_list, TRAIN_IMAGE_FOLDER)

        # 2. Process and download TEST images
        test_data = test_df[['sample_id', 'image_link']].dropna(subset=['image_link'])
        test_image_list = list(test_data.itertuples(index=False, name=None))
        download_all_images(test_image_list, TEST_IMAGE_FOLDER)
        
        # --- MODIFICATION: Updated verification step for separate folders ---
        print("\n--- Verification ---")
        if os.path.exists(TRAIN_IMAGE_FOLDER):
            train_count = len(os.listdir(TRAIN_IMAGE_FOLDER))
            print(f"Train: {train_count} / {len(train_image_list)} images downloaded to '{TRAIN_IMAGE_FOLDER}'")
        else:
            print(f"Train folder '{TRAIN_IMAGE_FOLDER}' not found.")

        if os.path.exists(TEST_IMAGE_FOLDER):
            test_count = len(os.listdir(TEST_IMAGE_FOLDER))
            print(f"Test:  {test_count} / {len(test_image_list)} images downloaded to '{TEST_IMAGE_FOLDER}'")
        else:
            print(f"Test folder '{TEST_IMAGE_FOLDER}' not found.")
        # --- END MODIFICATION ---

    except FileNotFoundError:
        print("Error: train.csv or test.csv not found.")
        print("Please make sure your project has a 'dataset' folder containing the CSV files.")
        print(f"Expected paths: '{TRAIN_CSV_PATH}' and '{TEST_CSV_PATH}'")