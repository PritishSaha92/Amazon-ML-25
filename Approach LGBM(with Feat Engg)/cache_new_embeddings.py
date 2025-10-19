"""
Cache DeBERTa-v3-base text embeddings and CLIP ViT-L/14 image embeddings
for both train and test splits. Outputs compressed NPZ files under
cache_new/ for reuse in Model A (LightGBM) and Model B (Two-Tower).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset

from PIL import Image, ImageOps

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from transformers import (
    AutoTokenizer,
    AutoModel,
    CLIPModel,
    CLIPProcessor,
)


# ------------------------------
# Utility and dataset classes
# ------------------------------


@dataclass
class EmbeddingConfig:
    dataset_dir: str
    images_dir: str
    cache_dir: str
    splits: Tuple[str, ...]
    text_model_name: str
    image_model_name: str
    max_length_text: int
    batch_size_text: int
    batch_size_image: int
    num_workers: int
    prefetch_factor: int
    pin_memory: bool
    persistent_workers: bool
    device: str
    fp16: bool
    overwrite: bool


class TextDataset(Dataset):
    def __init__(self, sample_ids: np.ndarray, texts: List[str]):
        self.sample_ids = sample_ids
        self.texts = texts

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int):
        return {
            "sample_id": int(self.sample_ids[index]),
            "text": self.texts[index],
        }


class ImageDataset(Dataset):
    def __init__(self, sample_ids: np.ndarray, image_dir: str):
        self.sample_ids = sample_ids
        self.image_dir = image_dir

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int):
        sample_id = int(self.sample_ids[index])
        image_path = os.path.join(self.image_dir, f"{sample_id}.jpg")
        image = None
        error = None
        try:
            with Image.open(image_path) as pil_img:
                image = ImageOps.exif_transpose(pil_img).convert("RGB")
        except Exception as ex:  # noqa: BLE001
            # --- DEBUG LINE ADDED ---
            # This will print the path of any image that fails to load.
            print(f"DEBUG: Failed to load image at path: {image_path} | Error: {ex}")
            error = str(ex)

        return {
            "sample_id": sample_id,
            "image": image,
            "error": error,
        }


# ------------------------------
# Embedding helpers
# ------------------------------


def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = (last_hidden_state * input_mask_expanded).sum(dim=1)
    sum_mask = input_mask_expanded.sum(dim=1).clamp(min=1e-9)
    return sum_embeddings / sum_mask


def compute_text_embeddings(
    df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: str,
    max_length: int,
    batch_size: int,
    num_workers: int,
    use_fp16: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    sample_ids = df["sample_id"].to_numpy()
    texts = df["descriptive_text"].fillna("").astype(str).tolist()

    dataset = TextDataset(sample_ids=sample_ids, texts=texts)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=1 if num_workers > 0 else None,
    )

    all_embeddings: List[np.ndarray] = []
    model.eval()
    autocast_dtype = torch.float16 if use_fp16 else None
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Text embeddings", leave=False):
            tokenized = tokenizer(
                batch["text"], padding=True, truncation=True, max_length=max_length, return_tensors="pt"
            )
            tokenized = {k: v.to(device) for k, v in tokenized.items()}
            with torch.autocast(device_type="cuda", dtype=autocast_dtype) if use_fp16 else torch.no_grad():
                outputs = model(**tokenized)
                pooled = mean_pooling(outputs.last_hidden_state, tokenized["attention_mask"])
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            all_embeddings.append(pooled.detach().cpu().numpy())
    embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    return sample_ids, embeddings


def compute_image_embeddings(
    df: pd.DataFrame,
    images_split_dir: str,
    processor: CLIPProcessor,
    model: CLIPModel,
    device: str,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    pin_memory: bool,
    persistent_workers: bool,
    use_fp16: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sample_ids = df["sample_id"].to_numpy()
    dataset = ImageDataset(sample_ids=sample_ids, image_dir=images_split_dir)
    dl_kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate_images,
    )
    if num_workers > 0:
        dl_kwargs.update(dict(prefetch_factor=prefetch_factor, persistent_workers=persistent_workers))
    dataloader = DataLoader(dataset, **dl_kwargs)

    all_embeddings: List[np.ndarray] = []
    all_missing_mask: List[np.ndarray] = []
    model.eval()
    autocast_dtype = torch.float16 if use_fp16 else None
    with torch.no_grad():
        for step_idx, batch in enumerate(tqdm(dataloader, desc="Image embeddings", leave=False), start=1):
            images: List[Image.Image] = batch["images"]
            missing_mask: np.ndarray = batch["missing_mask"]
            inputs = processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            with torch.autocast(device_type="cuda", dtype=autocast_dtype) if use_fp16 else torch.no_grad():
                image_features = model.get_image_features(pixel_values=pixel_values)
                image_features = torch.nn.functional.normalize(image_features, p=2, dim=1)
            all_embeddings.append(image_features.detach().cpu().numpy())
            all_missing_mask.append(missing_mask)
            if step_idx % 10 == 0:
                torch.cuda.empty_cache()
    embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    missing = np.concatenate(all_missing_mask, axis=0).astype(np.uint8)
    return sample_ids, embeddings, missing


def _collate_images(batch: List[dict]) -> dict:
    images: List[Image.Image] = []
    missing_mask: List[int] = []
    for item in batch:
        if item["image"] is None:
            images.append(Image.new("RGB", (1, 1), color=(0, 0, 0)))
            missing_mask.append(1)
        else:
            images.append(item["image"])
            missing_mask.append(0)
    return {
        "images": images,
        "missing_mask": np.array(missing_mask, dtype=np.uint8),
    }


# ------------------------------
# Main entry point
# ------------------------------

def _create_descriptive_text(row: pd.Series) -> str:
    parts = []
    if "brand" in row and pd.notna(row["brand"]):
        parts.append(f"Brand: {row['brand']}.")
    if "category" in row and pd.notna(row["category"]):
        parts.append(f"Category: {row['category']}.")
    if "pack_count" in row and pd.notna(row["pack_count"]) and row["pack_count"] > 1:
        parts.append(f"Pack of {int(row['pack_count'])}.")
    if "mass_g" in row and pd.notna(row["mass_g"]) and row["mass_g"] > 0:
        parts.append(f"Weight: {row['mass_g']:.2f}g.")
    if "volume_ml" in row and pd.notna(row["volume_ml"]) and row["volume_ml"] > 0:
        parts.append(f"Volume: {row['volume_ml']:.2f}ml.")
    if "keywords" in row and isinstance(row["keywords"], list) and row["keywords"]:
        parts.append(f"Features: {', '.join(row['keywords'])}.")
    return " ".join(parts) if parts else "No description available."


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache text and image embeddings for train/test splits.")
    parser.add_argument("--dataset_dir", type=str, default="data", help="Directory containing train.json and test.json")
    parser.add_argument("--images_dir", type=str, default="images", help="Directory containing images/train and images/test")
    parser.add_argument("--cache_dir", type=str, default="cache_new", help="Output directory for cached NPZ files")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "test"], choices=["train", "test"], help="Dataset splits to process")
    parser.add_argument("--text_model", dest="text_model_name", type=str, default="microsoft/deberta-v3-base", help="Hugging Face text model name")
    parser.add_argument("--image_model", dest="image_model_name", type=str, default="openai/clip-vit-large-patch14", help="Hugging Face image model name")
    parser.add_argument("--max_length_text", type=int, default=256, help="Max sequence length for text tokenization")
    parser.add_argument("--batch_size_text", type=int, default=256, help="Batch size for text embedding")
    parser.add_argument("--batch_size_image", type=int, default=128, help="Batch size for image embedding")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--prefetch_factor", type=int, default=1, help="DataLoader prefetch_factor for image loader (workers>0)")
    parser.add_argument("--pin_memory", action="store_true", help="Use pinned memory for DataLoaders")
    parser.add_argument("--persistent_workers", action="store_true", help="Keep workers alive between iterations (image loader)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on")
    parser.add_argument("--fp16", action="store_true", help="Use float16 mixed precision (recommended on A100/H100)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing cache files if present")

    args = parser.parse_args()
    config = EmbeddingConfig(**vars(args))
    os.makedirs(config.cache_dir, exist_ok=True)

    print(f"Loading text model: {config.text_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.text_model_name)
    text_model = AutoModel.from_pretrained(config.text_model_name).to(config.device)

    print(f"Loading image model: {config.image_model_name}")
    image_processor = CLIPProcessor.from_pretrained(config.image_model_name, use_fast=True)
    image_model = CLIPModel.from_pretrained(config.image_model_name).to(config.device)

    for split in config.splits:
        print(f"\nProcessing split: {split}")
        json_path = os.path.join(config.dataset_dir, f"{split}.json")
        if not os.path.exists(json_path):
            print(f"ERROR: JSON file not found at {json_path}")
            sys.exit(1)
        try:
            df = pd.read_json(json_path)
        except Exception as e:
            print(f"ERROR: Failed to read or parse JSON file at {json_path}. Details: {e}")
            continue
        if "Sample_id" in df.columns:
            df.rename(columns={"Sample_id": "sample_id"}, inplace=True)
        elif "sample_id" not in df.columns:
            print(f"ERROR: 'Sample_id' or 'sample_id' column not found in {json_path}")
            continue
        df = df.dropna(subset=["sample_id"]).copy()
        df["sample_id"] = df["sample_id"].astype(np.int64)

        print("Generating descriptive text from features...")
        df["descriptive_text"] = df.apply(_create_descriptive_text, axis=1)

        text_out = os.path.join(config.cache_dir, f"deberta_v3_base_text_{split}.npz")
        img_out = os.path.join(config.cache_dir, f"clip_vit_l14_image_{split}.npz")

        if config.overwrite or not os.path.exists(text_out):
            print("Creating text embeddings...")
            sample_ids_text, text_embeds = compute_text_embeddings(
                df=df, tokenizer=tokenizer, model=text_model, device=config.device,
                max_length=config.max_length_text, batch_size=config.batch_size_text,
                num_workers=config.num_workers, use_fp16=config.fp16,
            )
            np.savez_compressed(text_out, sample_id=sample_ids_text, embedding=text_embeds.astype(np.float32))
            print(f"Saved: {text_out} [shape={text_embeds.shape}]")
        else:
            print(f"Skipping text embeddings (exists): {text_out}")
        if config.overwrite or not os.path.exists(img_out):
            print("Creating image embeddings...")
            images_split_dir = os.path.join(config.images_dir, split)
            sample_ids_img, img_embeds, missing_mask = compute_image_embeddings(
                df=df, images_split_dir=images_split_dir, processor=image_processor, model=image_model,
                device=config.device, batch_size=config.batch_size_image, num_workers=config.num_workers,
                prefetch_factor=config.prefetch_factor, pin_memory=config.pin_memory,
                persistent_workers=config.persistent_workers if config.num_workers > 0 else False,
                use_fp16=config.fp16,
            )
            np.savez_compressed(
                img_out, sample_id=sample_ids_img, embedding=img_embeds.astype(np.float32),
                missing_image=missing_mask.astype(np.uint8),
            )
            missing_count = int(missing_mask.sum())
            print(f"Saved: {img_out} [shape={img_embeds.shape}, missing_images={missing_count}]")
        else:
            print(f"Skipping image embeddings (exists): {img_out}")

    print("\nDone. Embeddings cached in:", config.cache_dir)


if __name__ == "__main__":
    main()