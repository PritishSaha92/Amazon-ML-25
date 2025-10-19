#!/usr/bin/env python
# coding: utf-8

# ##  Basic Library imports

# In[ ]:


# %pip install -U transformers datasets peft trl bitsandbytes accelerate qwen-vl-utils pillow tensorboard


# In[1]:


import os
import pandas as pd
import json
from urllib.parse import urlparse
import warnings
import torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["OMP_NUM_THREADS"] = "16"

warnings.filterwarnings("ignore")

DATASET_FOLDER = "../dataset/"


# In[2]:


FULL_TRAINING_MODE = False

# Set to True to run inference on test set and generate submission
RUN_INFERENCE = False

# Set to True to skip dataset creation if already created
SKIP_DATASET_CREATION = False

# Set to True to skip image downloading if already done
SKIP_IMAGE_DOWNLOAD = False

INFERENCE_CHECKPOINT = None  # None = use default output_dir

INFERENCE_ONLY = False

RESUME_FROM_CHECKPOINT = None



# Options for resuming:
# RESUME_FROM_CHECKPOINT = "./qwen2-vl-7b-price-predictor-best/periodic_checkpoint"  # Latest periodic
# RESUME_FROM_CHECKPOINT = "./qwen2-vl-7b-price-predictor-best/checkpoint-epoch-1"  # Specific epoch
# RESUME_FROM_CHECKPOINT = True  # Auto-detect latest checkpoint

print("Configuration:")
print(f"  Full Training Mode: {FULL_TRAINING_MODE}")
print(f"  Run Inference: {RUN_INFERENCE}")
print(f"  Skip Dataset Creation: {SKIP_DATASET_CREATION}")
print(f"  Skip Image Download: {SKIP_IMAGE_DOWNLOAD}")



"""
development (validation mode):
FULL_TRAINING_MODE = False
INFERENCE_ONLY = False
RUN_INFERENCE = False
SKIP_DATASET_CREATION = False
SKIP_IMAGE_DOWNLOAD = False

full training:
FULL_TRAINING_MODE = True   # Use all data
INFERENCE_ONLY = False
RUN_INFERENCE = False
SKIP_DATASET_CREATION = True   # Reuse
SKIP_IMAGE_DOWNLOAD = True     # Reuse

generate submission:
FULL_TRAINING_MODE = True   # Doesn't matter
INFERENCE_ONLY = True       # Skip training!
RUN_INFERENCE = True        # Generate predictions
SKIP_DATASET_CREATION = True
SKIP_IMAGE_DOWNLOAD = True

"""


# ##  Download Datasets

# In[3]:


from utils import download_images

if not SKIP_IMAGE_DOWNLOAD:
    print("Downloading training images...")
    train = pd.read_csv(os.path.join(DATASET_FOLDER, "train.csv"))
    download_images(train["image_link"], f"{DATASET_FOLDER}/train_images")
    print(f"Downloaded images for {len(train)} training samples")
else:
    print("Skipping image download (SKIP_IMAGE_DOWNLOAD=True)")


# In[6]:


if not SKIP_IMAGE_DOWNLOAD:
    print("Downloading test images...")
    test = pd.read_csv(os.path.join(DATASET_FOLDER, "test.csv"))
    download_images(test["image_link"], f"{DATASET_FOLDER}/test_images")
    print(f"Downloaded images for {len(test)} test samples")
else:
    print("Skipping test image download (SKIP_IMAGE_DOWNLOAD=True)")


# In[ ]:


from PIL import Image, ImageFile
from tqdm import tqdm
import shutil
import multiprocessing
from functools import partial

# Enable loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

def resize_single_image(filename, input_folder, output_folder, max_size):
    """Resize a single image (for parallel processing)"""
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)
    
    # Skip if already resized
    if os.path.exists(output_path):
        return True
    
    try:
        img = Image.open(input_path).convert("RGB")
        
        # Resize maintaining aspect ratio
        img.thumbnail((max_size, max_size), Image.LANCZOS)
        
        # Save with good quality
        img.save(output_path, "JPEG", quality=95, optimize=True)
        return True
    except Exception as e:
        # If resize fails, copy original
        try:
            shutil.copy(input_path, output_path)
        except:
            pass
        return False

def resize_images_in_folder(input_folder, output_folder, max_size=512, num_workers=16):
    """
    Pre-resizes all images using multiprocessing (10x faster!).
    
    Args:
        input_folder: Source folder with original images
        output_folder: Destination folder for resized images
        max_size: Maximum dimension (width or height)
        num_workers: Number of parallel workers
    """
    os.makedirs(output_folder, exist_ok=True)
    
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Resizing {len(image_files)} images from {input_folder}...")
    print(f"Using {num_workers} parallel workers")
    
    # Create partial function with fixed arguments
    resize_partial = partial(
        resize_single_image,
        input_folder=input_folder,
        output_folder=output_folder,
        max_size=max_size
    )
    
    # Use multiprocessing pool
    with multiprocessing.Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(resize_partial, image_files),
            total=len(image_files),
            desc="Resizing"
        ))
        pool.close()
        pool.join()
    
    failed_count = len([r for r in results if not r])
    
    print(f"✅ All images resized and saved to {output_folder}")
    if failed_count > 0:
        print(f"⚠️  {failed_count} images had issues (copied originals)")

# Resize images after downloading
if not SKIP_IMAGE_DOWNLOAD:
    print("\n" + "="*60)
    print("RESIZING IMAGES FOR FASTER TRAINING")
    print("="*60 + "\n")
    
    # Resize training images (with 16 workers!)
    resize_images_in_folder(
        input_folder=f"{DATASET_FOLDER}/train_images",
        output_folder=f"{DATASET_FOLDER}/train_images_resized",
        max_size=512,
        num_workers=16  # Parallel processing!
    )
    
    # Resize test images
    resize_images_in_folder(
        input_folder=f"{DATASET_FOLDER}/test_images",
        output_folder=f"{DATASET_FOLDER}/test_images_resized",
        max_size=512,
        num_workers=16
    )
    
    print("\n✅ Image resizing complete!")
else:
    print("Skipping image resizing (SKIP_IMAGE_DOWNLOAD=True)")


# ## Create JSONL

# In[ ]:


def get_filename_from_url(url):
    """
    Extracts filename from image URL.

    Example:
    'https://m.media-amazon.com/images/I/51mjZYDYjyL.jpg'
    -> '51mjZYDYjyL.jpg'
    """
    if not isinstance(url, str):
        return None
    try:
        path = urlparse(url).path
        filename = os.path.basename(path)
        return filename
    except Exception:
        return None


# In[ ]:


def create_jsonl_dataset(csv_path, image_folder, output_path, is_test_set=False):
    """
    Converts CSV + local images to JSONL format for Qwen2.5-VL.

    Dataset Format:
    {
        "messages": [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": [
                {"type": "image", "image": "file:///absolute/path.jpg"},
                {"type": "text", "text": "user prompt"}
            ]},
            {"role": "assistant", "content": "price"}
        ]
    }

    Key Points:
    - Uses file:/// (3 slashes) for local absolute paths
    - Content can be string OR list of dicts for multimodal
    - Assistant response only included for training set
    """
    df = pd.read_csv(csv_path)
    print(f"Processing {len(df)} rows from {os.path.basename(csv_path)}...")

    with open(output_path, "w") as f:
        for index, row in df.iterrows():
            image_filename = get_filename_from_url(row["image_link"])
            if not image_filename:
                print(
                    f"Warning: Could not parse filename from URL for sample_id {row['sample_id']}. Skipping."
                )
                continue

            resized_folder = image_folder + "_resized"
            if os.path.exists(resized_folder):
                image_path = os.path.abspath(os.path.join(resized_folder, image_filename))
            else:
                image_path = os.path.abspath(os.path.join(image_folder, image_filename))
            image_exists = os.path.exists(image_path)

            if not image_exists and not is_test_set:
                print(
                    f"Skipping training sample_id {row['sample_id']}: Image not found at {image_path}"
                )
                continue

            product_description = row["catalog_content"].strip()

            system_prompt = (
                "You are a precise e-commerce expert. Your task is to analyze the product "
                "image and its description, then output only the product's price as a numerical "
                "value. Do not add any other text or currency symbols."
            )

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
                print(f"Creating text-only entry for test sample_id {row['sample_id']}")
                user_content = [{"type": "text", "text": user_prompt_text}]

            messages.append({"role": "user", "content": user_content})

            if not is_test_set:
                price_float = float(row["price"])
                price = f"{round(price_float, 2):.2f}"
                messages.append({"role": "assistant", "content": price})

            jsonl_entry = {
                "sample_id": row["sample_id"],
                "messages": messages,
            }
            f.write(json.dumps(jsonl_entry) + "\n")

    print(f"Successfully created dataset at {output_path}")


# In[ ]:


if not SKIP_DATASET_CREATION:
    create_jsonl_dataset(
        csv_path=os.path.join(DATASET_FOLDER, "train.csv"),
        image_folder=f"{DATASET_FOLDER}/train_images",
        output_path=f"{DATASET_FOLDER}/train.jsonl",
        is_test_set=False,
    )

    create_jsonl_dataset(
        csv_path=os.path.join(DATASET_FOLDER, "test.csv"),
        image_folder=f"{DATASET_FOLDER}/test_images",
        output_path=f"{DATASET_FOLDER}/test.jsonl",
        is_test_set=True,
    )
else:
    print("Skipping dataset creation (SKIP_DATASET_CREATION=True)")


# In[ ]:


import os
import json
import pandas as pd
from pathlib import Path

def count_images_in_folder(folder_path):
    """Count total image files in a folder"""
    if not os.path.exists(folder_path):
        return 0
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    return len(image_files)

def count_unique_images_in_csv(csv_path):
    """Count unique image URLs in CSV"""
    df = pd.read_csv(csv_path)
    return df['image_link'].nunique()

def count_jsonl_entries(jsonl_path):
    """Count total entries in JSONL file"""
    if not os.path.exists(jsonl_path):
        return 0
    with open(jsonl_path, 'r') as f:
        return sum(1 for _ in f)

def count_jsonl_with_images(jsonl_path):
    """Count JSONL entries that have images"""
    if not os.path.exists(jsonl_path):
        return 0
    count = 0
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            # Check if any message content has an image
            for msg in data.get('messages', []):
                if isinstance(msg.get('content'), list):
                    for item in msg['content']:
                        if isinstance(item, dict) and item.get('type') == 'image':
                            count += 1
                            break
    return count

print("=" * 70)
print("DATASET STATISTICS & VALIDATION")
print("=" * 70)

# TRAIN DATASET
print("\n📊 TRAINING DATASET:")
print("-" * 70)

train_csv_rows = len(pd.read_csv(f"{DATASET_FOLDER}/train.csv"))
train_unique_urls = count_unique_images_in_csv(f"{DATASET_FOLDER}/train.csv")
train_images = count_images_in_folder(f"{DATASET_FOLDER}/train_images")
train_images_resized = count_images_in_folder(f"{DATASET_FOLDER}/train_images_resized")
train_jsonl_entries = count_jsonl_entries(f"{DATASET_FOLDER}/train.jsonl")
train_jsonl_with_images = count_jsonl_with_images(f"{DATASET_FOLDER}/train.jsonl")

print(f"  CSV rows:                    {train_csv_rows:>6,}")
print(f"  Unique image URLs in CSV:    {train_unique_urls:>6,}")
print(f"  Downloaded images:           {train_images:>6,}")
print(f"  Resized images:              {train_images_resized:>6,}")
print(f"  JSONL total entries:         {train_jsonl_entries:>6,}")
print(f"  JSONL entries with images:   {train_jsonl_with_images:>6,}")
print(f"  JSONL entries text-only:     {train_jsonl_entries - train_jsonl_with_images:>6,}")

# Calculate differences
train_missing_downloads = train_unique_urls - train_images
train_missing_resized = train_images - train_images_resized
train_skipped_jsonl = train_csv_rows - train_jsonl_entries

print(f"\n  Missing from download:       {train_missing_downloads:>6,} ({train_missing_downloads/train_unique_urls*100:.2f}%)")
print(f"  Failed to resize:            {train_missing_resized:>6,}")
print(f"  Skipped in JSONL:            {train_skipped_jsonl:>6,}")

# TEST DATASET
print("\n📊 TEST DATASET:")
print("-" * 70)

test_csv_rows = len(pd.read_csv(f"{DATASET_FOLDER}/test.csv"))
test_unique_urls = count_unique_images_in_csv(f"{DATASET_FOLDER}/test.csv")
test_images = count_images_in_folder(f"{DATASET_FOLDER}/test_images")
test_images_resized = count_images_in_folder(f"{DATASET_FOLDER}/test_images_resized")
test_jsonl_entries = count_jsonl_entries(f"{DATASET_FOLDER}/test.jsonl")
test_jsonl_with_images = count_jsonl_with_images(f"{DATASET_FOLDER}/test.jsonl")

print(f"  CSV rows:                    {test_csv_rows:>6,}")
print(f"  Unique image URLs in CSV:    {test_unique_urls:>6,}")
print(f"  Downloaded images:           {test_images:>6,}")
print(f"  Resized images:              {test_images_resized:>6,}")
print(f"  JSONL total entries:         {test_jsonl_entries:>6,}")
print(f"  JSONL entries with images:   {test_jsonl_with_images:>6,}")
print(f"  JSONL entries text-only:     {test_jsonl_entries - test_jsonl_with_images:>6,}")

# Calculate differences
test_missing_downloads = test_unique_urls - test_images
test_missing_resized = test_images - test_images_resized
test_text_only = test_jsonl_entries - test_jsonl_with_images

print(f"\n  Missing from download:       {test_missing_downloads:>6,} ({test_missing_downloads/test_unique_urls*100:.2f}%)")
print(f"  Failed to resize:            {test_missing_resized:>6,}")
print(f"  Text-only in JSONL:          {test_text_only:>6,} (for missing images)")

# SUMMARY
print("\n" + "=" * 70)
print("SUMMARY:")
print("=" * 70)
print(f"  Total training samples:      {train_jsonl_with_images:>6,} ✅")
print(f"  Total test samples:          {test_jsonl_entries:>6,} ({test_jsonl_with_images:,} with images)")
print(f"  Ready for training:          {'YES ✅' if train_jsonl_with_images > 60000 else 'NO ❌'}")
print(f"  Ready for inference:         {'YES ✅' if test_jsonl_entries == 75000 else 'NO ❌'}")
print("=" * 70)


# In[ ]:


import json
import random

print("Loading training data from JSONL...")
train_data_raw = []
with open(f"{DATASET_FOLDER}/train.jsonl", 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f, 1):
        try:
            train_data_raw.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"Warning: Skipping malformed JSON on line {line_num}: {e}")
            continue

print(f"✅ Loaded {len(train_data_raw)} training samples")


if FULL_TRAINING_MODE:
    print("\n" + "=" * 60)
    print("FULL TRAINING MODE: Using entire dataset for training")
    print("=" * 60)
    train_dataset = train_data_raw
    validation_dataset = None
    print(f"Training samples: {len(train_dataset)}")
    print("Validation: None (full training mode)")

else:
    print("\n" + "=" * 60)
    print("VALIDATION MODE: Splitting dataset for SMAPE monitoring")
    print("=" * 60)
    
    # Manual 90/10 split with shuffling
    random.seed(42)
    random.shuffle(train_data_raw)
    
    split_idx = int(len(train_data_raw) * 0.9)
    train_dataset = train_data_raw[:split_idx]
    validation_dataset = train_data_raw[split_idx:]
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(validation_dataset)}")

print("\n✅ Datasets ready for training!")
print(f"Dataset type: {type(train_dataset)}")  # Will be <class 'list'>


# ## Finetuning

# In[ ]:


import torch
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig,
)
from unsloth import FastVisionModel
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import gc
import numpy as np

torch.cuda.empty_cache()
gc.collect()
torch.cuda.reset_peak_memory_stats()

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# In[ ]:


model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
train_dataset_path = "./train_split.jsonl"
validation_dataset_path = "./validation_split.jsonl"
output_dir = "./qwen2.5-vl-3b-price-predictor-best"

print(f"Model: {model_id}")
print(f"Output directory: {output_dir}")


# In[ ]:


from transformers import TrainerCallback


class SMAPEEvaluationCallback(TrainerCallback):
    """
    Computes SMAPE after each epoch on validation set.
    """

    def __init__(
        self, trainer, validation_dataset, processor, device="cuda", max_samples=500
    ):
        self.trainer = trainer
        self.validation_dataset = validation_dataset
        self.processor = processor
        self.device = device
        self.max_samples = max_samples
        self.best_smape = float("inf")
        self.smape_history = []

    def on_epoch_end(self, args, state, control, **kwargs):
        """Called at end of each epoch."""
        if self.validation_dataset is None:
            print(
                "\nSkipping SMAPE evaluation (no validation set in FULL_TRAINING_MODE)"
            )
            return control
        try:
            print(f"\n{'=' * 60}")
            print(f"Computing SMAPE for Epoch {int(state.epoch)}...")
            print(f"{'=' * 60}")

            self.trainer.model.eval()
            smape_scores = []

            num_samples = min(len(self.validation_dataset), self.max_samples)
            validation_subset = self.validation_dataset[:num_samples]

            for idx, example in enumerate(validation_subset):
                messages = example["messages"][:-1]
                ground_truth = example["messages"][-1]["content"]

                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                images = []
                for msg in messages:
                    if isinstance(msg["content"], list):
                        for item in msg["content"]:
                            if isinstance(item, dict) and item.get("type") == "image":
                                images.append(item["image"])

                inputs = self.processor(
                    text=[text],
                    images=[images] if images else None,
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)

                with torch.no_grad():
                    generated_ids = self.trainer.model.generate(
                        **inputs,
                        max_new_tokens=20,
                        num_beams=1,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        pad_token_id=self.processor.tokenizer.pad_token_id,
                        eos_token_id=self.processor.tokenizer.eos_token_id
                    )

                generated_ids_trimmed = [
                    out_ids[len(in_ids) :]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                prediction = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0].strip()

                try:
                    pred_price = float(prediction)
                    true_price = float(ground_truth)

                    numerator = abs(pred_price - true_price)
                    denominator = (abs(true_price) + abs(pred_price)) / 2

                    if denominator == 0:
                        smape = 0.0 if numerator == 0 else 100.0
                    else:
                        smape = (numerator / denominator) * 100

                    smape_scores.append(smape)

                except (ValueError, TypeError):
                    smape_scores.append(100.0)

                if (idx + 1) % 50 == 0:
                    current_avg = np.mean(smape_scores)
                    print(f"  [{idx + 1}/{num_samples}] Running SMAPE: {current_avg:.2f}%")

            avg_smape = np.mean(smape_scores)
            self.smape_history.append(avg_smape)

            logs = {"eval_smape": avg_smape}
            self.trainer.log(logs)

            if avg_smape < self.best_smape:
                self.best_smape = avg_smape
                print(f"\n✨ NEW BEST SMAPE: {avg_smape:.2f}%")

            print(f"\n{'=' * 60}")
            print(f"Epoch {int(state.epoch)} SMAPE: {avg_smape:.2f}%")
            print(f"Best SMAPE: {self.best_smape:.2f}%")
            print(f"Target: < 44% (leaderboard #1)")
            print(f"{'=' * 60}\n")
        except Exception as e:
            print(f"\n❌ Error during SMAPE evaluation: {e}")
            print("Continuing training despite SMAPE error...")
            import traceback

            traceback.print_exc()
            return control

        self.trainer.model.train()
        return control


print("SMAPE Callback defined!")


# In[ ]:


class EarlyStoppingCallback(TrainerCallback):
    """
    Early stopping based on SMAPE metric.
    Stops training if SMAPE doesn't improve for N epochs.
    """

    def __init__(self, smape_callback, patience=2):
        self.smape_callback = smape_callback
        self.patience = patience
        self.patience_counter = 0
        self.best_smape = float("inf")

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.smape_callback is None:
            return control

        current_smape = (
            self.smape_callback.smape_history[-1]
            if self.smape_callback.smape_history
            else float("inf")
        )

        if current_smape < self.best_smape:
            self.best_smape = current_smape
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            print(f"***  No improvement for {self.patience_counter} epoch(s)")

        if self.patience_counter >= self.patience:
            print(
                f"\n*** Early stopping triggered! No improvement for {self.patience} epochs."
            )
            control.should_training_stop = True

        return control


# In[ ]:


print("🚀 Loading Qwen2.5-VL-3B with Unsloth optimization...")
model, tokenizer = FastVisionModel.from_pretrained(
    model_name=model_id,
    max_seq_length=512,
    load_in_4bit=True,
    dtype=torch.bfloat16,
    # use_gradient_checkpointing=True,
    use_gradient_checkpointing=False,
    trust_remote_code=True,
)

# Enable training mode
model = FastVisionModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0.,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                   "gate_proj", "up_proj", "down_proj"],
    bias="none",
    # use_gradient_checkpointing="unsloth",
    use_gradient_checkpointing=False,
    random_state=42,
)


# In[ ]:


print("Loading processor...")
min_pix = 256 * 28 * 28
max_pix = 512 * 28 * 28

processor = AutoProcessor.from_pretrained(model_id, min_pixels=min_pix, max_pixels=max_pix, trust_remote_code=True)

if processor.tokenizer.pad_token is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
processor.tokenizer.padding_side = "right"
print(f"   EOS token: '{processor.tokenizer.eos_token}' (ID: {processor.tokenizer.eos_token_id})")
print(f"   PAD token: '{processor.tokenizer.pad_token}' (ID: {processor.tokenizer.pad_token_id})")
print("Processor loaded successfully!")
print(f"Vocab size: {len(processor.tokenizer)}")
print(f"   Min pixels: {min_pix:,} (~256x256 after resize)")
print(f"   Max pixels: {max_pix:,} (~512x512 after resize)")


# In[ ]:


def collate_fn(examples):
    """
    Custom data collator for Qwen2-VL multimodal inputs.
    
    What it does:
    1. Applies chat template to convert messages to text format
    2. Extracts image paths from message structure
    3. Processes both text and images together with the processor
    4. Creates labels for causal language modeling
    """
    texts = [
        processor.apply_chat_template(
            example["messages"], 
            tokenize=False,
            add_generation_prompt=False
        )
        for example in examples
    ]

    image_inputs = []
    for example in examples:
        images = []
        for msg in example["messages"]:
            if isinstance(msg["content"], list):
                for item in msg["content"]:
                    if isinstance(item, dict) and item.get("type") == "image":
                        images.append(item["image"])
        image_inputs.append(images if images else None)

    batch = processor(
        text=texts,
        images=image_inputs,
        return_tensors="pt",
        padding=True,
    )

    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels
    
    return batch

print("Custom collator function defined!")


# In[ ]:


# Training configuration using SFTConfig
training_args = SFTConfig(
    # Basic settings
    output_dir=output_dir,
    num_train_epochs=3,
    # Batch size (A100-optimized)
    per_device_train_batch_size=2,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=1,  # Effective batch = 16
    # Memory optimization
    gradient_checkpointing=False,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    # A100-specific optimizations
    optim="adamw_torch_fused",
    tf32=True,  # A100 TensorFloat-32
    bf16=True,
    # Training hyperparameters
    learning_rate=2e-4,
    weight_decay=0.001,
    max_grad_norm=0.3,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    # Logging
    logging_steps=25,
    logging_first_step=True,
    # Evaluation & checkpointing
    eval_strategy="no",
    save_strategy="epoch",
    save_total_limit=3,
    load_best_model_at_end=False,
    # VLM-specific (CRITICAL!)
    remove_unused_columns=False,
    dataset_text_field="",
    dataset_kwargs={"skip_prepare_dataset": True},
    max_length=384,
    packing=False,
    # A100 performance
    dataloader_num_workers=16,
    dataloader_pin_memory=True,
    dataloader_persistent_workers=True,
    dataloader_prefetch_factor=8,
    # Reproducibility
    torch_empty_cache_steps=20,
    seed=42,
    data_seed=42,
)

print("Training configuration:")
print(f"  Epochs: {training_args.num_train_epochs}")
print(
    f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}"
)
print(f"  Learning rate: {training_args.learning_rate}")
print(f"  Mixed precision: bf16={training_args.bf16}")


# In[ ]:


class MemoryCleanupCallback(TrainerCallback):
    """Clean up GPU memory between epochs"""
    def on_epoch_end(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()
        gc.collect()
        print("\n🧹 GPU memory cleaned up")
        return control


# In[ ]:


import os
import shutil
import json
from transformers import TrainerCallback

class PeriodicCheckpointCallback(TrainerCallback):
    """
    Saves a rolling checkpoint every N steps (keeps only 1 copy).
    This is separate from epoch-based checkpoints.
    
    Purpose: Resume training if crash happens mid-epoch.
    """
    def __init__(self, save_steps=10, checkpoint_dir="periodic_checkpoint"):
        self.save_steps = save_steps
        self.checkpoint_dir = checkpoint_dir
        
    def on_step_end(self, args, state, control, **kwargs):
        # Save every N steps
        if state.global_step % self.save_steps == 0:
            checkpoint_path = os.path.join(args.output_dir, self.checkpoint_dir)
            
            # Remove old periodic checkpoint
            if os.path.exists(checkpoint_path):
                print(f"\n🔄 Replacing periodic checkpoint at step {state.global_step}")
                shutil.rmtree(checkpoint_path)
            else:
                print(f"\n💾 Creating first periodic checkpoint at step {state.global_step}")
            
            # Get model from kwargs (passed by callback handler)
            model = kwargs.get("model")
            if model is None:
                print("⚠️  Warning: Model not found in callback kwargs, skipping checkpoint")
                return control
            
            # Save model checkpoint
            model.save_pretrained(checkpoint_path)
            
            # Save trainer state (properly serialize it)
            state_path = os.path.join(checkpoint_path, "trainer_state.json")
            with open(state_path, 'w') as f:
                # Convert TrainerState to dict manually
                state_dict = {
                    'epoch': state.epoch,
                    'global_step': state.global_step,
                    'max_steps': state.max_steps,
                    'num_train_epochs': state.num_train_epochs,
                    'log_history': state.log_history,
                    'best_metric': state.best_metric,
                    'best_model_checkpoint': state.best_model_checkpoint,
                    'is_local_process_zero': state.is_local_process_zero,
                    'is_world_process_zero': state.is_world_process_zero,
                }
                json.dump(state_dict, f, indent=2)
            
            print(f"✅ Periodic checkpoint saved (step {state.global_step})")
        
        return control

print("✅ PeriodicCheckpointCallback defined!")


# In[ ]:


import os
kernel_pid = os.getpid()
print(f"Your notebook's kernel PID is: {kernel_pid}")


# In[ ]:


print("Loading datasets...")
if FULL_TRAINING_MODE:
    eval_dataset_for_trainer = None
else:
    eval_dataset_for_trainer = validation_dataset

print(f"Training samples: {len(train_dataset)}")
if eval_dataset_for_trainer:
    print(f"Validation samples: {len(eval_dataset_for_trainer)}")

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset_for_trainer,
    data_collator=collate_fn,
    processing_class=processor.tokenizer,
)
trainer.add_callback(MemoryCleanupCallback())

if not FULL_TRAINING_MODE and eval_dataset_for_trainer is not None:
    # SMAPE callback
    smape_eval_samples = min(
        len(eval_dataset_for_trainer), 
        max(100, int(len(eval_dataset_for_trainer) * 0.2))
    )
    smape_eval_samples = min(smape_eval_samples, 1000)
    
    smape_callback = SMAPEEvaluationCallback(
        trainer=trainer,
        validation_dataset=eval_dataset_for_trainer,
        processor=processor,
        device="cuda",
        max_samples=smape_eval_samples,
    )
    trainer.add_callback(smape_callback)
    print("\n✅ SMAPE callback added")
    
    # Early stopping
    early_stopping = EarlyStoppingCallback(smape_callback, patience=2)
    trainer.add_callback(early_stopping)
    print("✅ Early stopping added (patience=2)")
else:
    smape_callback = None
    print("\n⚠️  No SMAPE evaluation (FULL_TRAINING_MODE)")

# ADD PERIODIC CHECKPOINT CALLBACK (Always enabled!)
periodic_checkpoint = PeriodicCheckpointCallback(
    save_steps=30,  # Save every 10 steps
    checkpoint_dir="periodic_checkpoint"  # Single rolling checkpoint
)
trainer.add_callback(periodic_checkpoint)
print("✅ Periodic checkpoint callback added (every 10 steps)")

print("\n" + "=" * 50)
print("Trainer initialized successfully!")
print("=" * 50)


# In[ ]:


import os
kernel_pid = os.getpid()
print(f"Your notebook's kernel PID is: {kernel_pid}")


# In[ ]:


if not INFERENCE_ONLY:    
    print("\n" + "=" * 50)
    print("Starting Fine-Tuning...")
    print("=" * 50 + "\n")

    if RESUME_FROM_CHECKPOINT:
        print(f"🔄 Resuming from checkpoint: {RESUME_FROM_CHECKPOINT}")
    
    trainer.train(resume_from_checkpoint=RESUME_FROM_CHECKPOINT)

    print("\n" + "=" * 50)
    print("Training Complete!")
    if smape_callback:
        print(f"Best SMAPE achieved: {smape_callback.best_smape:.2f}%")
        print(f"SMAPE history: {smape_callback.smape_history}")
    print("=" * 50)

    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Total epochs: {trainer.state.epoch}")
    print(f"Total steps: {trainer.state.global_step}")
    print(f"Best checkpoint: {trainer.state.best_model_checkpoint}")

    if smape_callback:
        print(f"\nSMAPE Performance:")
        print(f"  Initial SMAPE: {smape_callback.smape_history[0]:.2f}%")
        print(f"  Final SMAPE: {smape_callback.smape_history[-1]:.2f}%")
        print(f"  Best SMAPE: {smape_callback.best_smape:.2f}%")
        print(
            f"  Improvement: {smape_callback.smape_history[0] - smape_callback.best_smape:.2f}%"
        )
        print(f"\n  Epoch-by-epoch SMAPE:")
        for i, smape in enumerate(smape_callback.smape_history, 1):
            print(f"    Epoch {i}: {smape:.2f}%")

    print("=" * 60)

    print("\n" + "=" * 50)
    print("Saving the Best Model...")
    print("=" * 50)

    trainer.save_model(output_dir)

    processor.save_pretrained(output_dir)

    print(f"\n*** Model and processor saved to: {output_dir}")
    print("\nSaved files:")
    print("  - adapter_config.json (LoRA configuration)")
    print("  - adapter_model.safetensors (LoRA weights)")
    print("  - tokenizer files")
    print("  - processor configuration")
else:
    print("\n***  Skipping training (INFERENCE_ONLY=True)")


# In[ ]:


from datasets import load_dataset
from tqdm import tqdm
if RUN_INFERENCE:
    print("\n" + "=" * 60)
    print("Running Inference on Test Set...")
    print("=" * 60)

    test_dataset = load_dataset(
        "json", data_files=f"{DATASET_FOLDER}/test.jsonl", split="train"
    )
    print(f"Test samples: {len(test_dataset)}")

    checkpoint_path = INFERENCE_CHECKPOINT if INFERENCE_CHECKPOINT else output_dir
    if os.path.exists(f"{checkpoint_path}/adapter_model.safetensors"):
        print(f"\n*** Loading adapter from: {checkpoint_path}")
        from peft import PeftModel

        try:
            _ = model.device
        except:
            print("Loading base model...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )

        model = PeftModel.from_pretrained(model, checkpoint_path)
        model.eval()
        print("*** Fine-tuned adapter loaded!")
    else:
        print("***  Using model from training session (no saved adapter found)")
        model.eval()

    predictions = []
    sample_ids = []
    batch_size = 16

    for batch_start in tqdm(range(0, len(test_dataset), batch_size), desc="Inference"):
        batch_end = min(batch_start + batch_size, len(test_dataset))
        batch_examples = test_dataset[batch_start:batch_end]

        if not isinstance(batch_examples["sample_id"], list):
            batch_examples = {k: [v] for k, v in batch_examples.items()}

        batch_sample_ids = batch_examples["sample_id"]
        batch_messages = batch_examples["messages"]

        batch_texts = []
        batch_images = []

        for messages in batch_messages:
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            batch_texts.append(text)
            images = []
            for msg in messages:
                if isinstance(msg["content"], list):
                    for item in msg["content"]:
                        if isinstance(item, dict) and item.get("type") == "image":
                            images.append(item["image"])
            batch_images.append(images if images else None)

        inputs = processor(
            text=batch_texts,
            images=batch_images,
            return_tensors="pt",
            padding=True,
        ).to("cuda")

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=20,
                num_beams=1,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )

        for i, (in_ids, out_ids) in enumerate(zip(inputs.input_ids, generated_ids)):
            generated_ids_trimmed = out_ids[len(in_ids) :]
            prediction = processor.decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()

            try:
                predicted_price = float(prediction)
            except (ValueError, TypeError):
                print(
                    f"Warning: Non-numeric prediction '{prediction}' for sample {batch_sample_ids[i]}, using 0.0"
                )
                predicted_price = 0.0

            predictions.append(predicted_price)
            sample_ids.append(batch_sample_ids[i])

        if (batch_end) % 500 == 0 or batch_end == len(test_dataset):
            print(f"  Processed {batch_end}/{len(test_dataset)} samples...")

    submission_df = pd.DataFrame({"sample_id": sample_ids, "price": predictions})
    submission_path = "test_out.csv"

    print("\nValidating submission format...")
    assert "sample_id" in submission_df.columns, "Missing sample_id column"
    assert "price" in submission_df.columns, "Missing price column"
    assert len(submission_df) == len(test_dataset), (
        f"Mismatch: {len(submission_df)} predictions vs {len(test_dataset)} test samples"
    )
    assert submission_df["sample_id"].duplicated().sum() == 0, (
        "Duplicate sample_ids found!"
    )
    print("*** Submission format validated")

    failed_predictions = sum(1 for p in predictions if p == 0.0)
    if failed_predictions > len(predictions) * 0.1:
        print(
            f"\n***  WARNING: {failed_predictions}/{len(predictions)} predictions failed (returned 0.0)"
        )
        print("This suggests the model may need more training or better prompts")

    submission_df.to_csv(submission_path, index=False)

    print(f"\n{'=' * 60}")
    print(f"*** Submission saved to: {submission_path}")
    print(f"Total predictions: {len(predictions)}")
    print(f"Sample predictions:")
    print(submission_df.head(10))
    print(f"{'=' * 60}")

    del inputs
    torch.cuda.empty_cache()
    gc.collect()
    print("\n*** GPU memory cleaned up")

else:
    print("\n***  Skipping inference (RUN_INFERENCE=False)")

