import os
import multiprocessing
from functools import partial
from PIL import Image, ImageFile
from tqdm import tqdm
import shutil

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASET_DIR = "../dataset"
TRAIN_INPUT_DIR = os.path.join(DATASET_DIR, "train_images")
TEST_INPUT_DIR = os.path.join(DATASET_DIR, "test_images")
TRAIN_OUTPUT_DIR = os.path.join(DATASET_DIR, "train_images_resized")
TEST_OUTPUT_DIR = os.path.join(DATASET_DIR, "test_images_resized")
MAX_SIZE = 512
NUM_WORKERS = 16


def resize_single_image(filename, input_folder, output_folder, max_size):
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)
    
    if os.path.exists(output_path):
        return True
    
    try:
        img = Image.open(input_path).convert("RGB")
        img.thumbnail((max_size, max_size), Image.LANCZOS)
        img.save(output_path, "JPEG", quality=95, optimize=True)
        return True
    except Exception:
        try:
            shutil.copy(input_path, output_path)
        except:
            pass
        return False


def resize_images_in_folder(input_folder, output_folder, max_size, num_workers):
    os.makedirs(output_folder, exist_ok=True)
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Resizing {len(image_files)} images from {input_folder}...")
    print(f"Using {num_workers} parallel workers")
    
    resize_partial = partial(
        resize_single_image,
        input_folder=input_folder,
        output_folder=output_folder,
        max_size=max_size
    )
    
    with multiprocessing.Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(resize_partial, image_files), total=len(image_files), desc="Resizing"))
        pool.close()
        pool.join()
    
    failed_count = len([r for r in results if not r])
    print(f"✅ Resizing complete! Saved to {output_folder}")
    if failed_count > 0:
        print(f"⚠️  {failed_count} images had issues (copied originals)")


def main():
    print("="*70)
    print("STEP 2: RESIZING IMAGES")
    print("="*70)
    
    print(f"\nResizing train images...")
    resize_images_in_folder(TRAIN_INPUT_DIR, TRAIN_OUTPUT_DIR, MAX_SIZE, NUM_WORKERS)
    
    print(f"\nResizing test images...")
    resize_images_in_folder(TEST_INPUT_DIR, TEST_OUTPUT_DIR, MAX_SIZE, NUM_WORKERS)
    
    print("\n✅ Image resizing complete!")
    print("="*70)

if __name__ == "__main__":
    main()
