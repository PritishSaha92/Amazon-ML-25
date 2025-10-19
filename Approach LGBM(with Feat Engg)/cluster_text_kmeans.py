#!/usr/bin/env python3
"""
Train-only MiniBatchKMeans clustering of DeBERTa-v3-base sentence embeddings
and attach the resulting cluster id as a pseudo-category feature to the
structured CSVs for train and test.

Key properties:
- No leakage: K selection and final model fit use only TRAIN embeddings
- Automatic K selection over a grid (default: 300..600 step 50) using
  Calinski-Harabasz on a held-out train subset
- Efficient: uses MiniBatchKMeans; memory-friendly sampling for K selection
- Saves artifacts (model, metadata, assignments) under cache directory

Outputs:
- New CSVs with an added column `text_kmeans_cluster_id`:
  `output/vlm_structured_data_train.clustered_k{K}.csv`
  `output/vlm_structured_data_test.clustered_k{K}.csv`
- Clustering artifacts in `cache_new/` by default
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import calinski_harabasz_score


DEFAULT_CACHE_DIR = "cache_new"
DEFAULT_OUTPUT_DIR = "output"


@dataclass
class ClusteringConfig:
    cache_dir: str
    output_dir: str
    train_npz: str
    test_npz: str
    structured_train_csv: str
    structured_test_csv: str
    select_k: bool
    k: int
    k_min: int
    k_max: int
    k_step: int
    select_k_train_samples: int
    select_k_val_samples: int
    batch_size: int
    max_iter: int
    n_init: int
    random_state: int
    overwrite_structured: bool


def _ensure_exists(path: str, description: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{description} not found at: {path}")


def _load_npz_embeddings(npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(npz_path) as data:
        sample_ids = data["sample_id"].astype(np.int64)
        embeddings = data["embedding"].astype(np.float32)
    return sample_ids, embeddings


def _choose_k_via_ch(
    embeddings: np.ndarray,
    k_min: int,
    k_max: int,
    k_step: int,
    select_k_train_samples: int,
    select_k_val_samples: int,
    batch_size: int,
    max_iter: int,
    n_init: int,
    random_state: int,
) -> Tuple[int, Dict[int, float]]:
    num_rows = embeddings.shape[0]
    rng = np.random.default_rng(seed=random_state)

    # Disjoint sampling for train/val within TRAIN to avoid optimism.
    permuted_indices = rng.permutation(num_rows)
    train_count = min(num_rows, int(select_k_train_samples))
    train_indices = permuted_indices[:train_count]
    remaining_indices = permuted_indices[train_count:]
    val_count = min(remaining_indices.shape[0], int(select_k_val_samples))
    if val_count == 0:
        # Fallback: reuse the same pool if dataset is too small
        val_indices = train_indices
    else:
        val_indices = remaining_indices[:val_count]

    X_train = embeddings[train_indices]
    X_val = embeddings[val_indices]

    candidate_ks = list(range(int(k_min), int(k_max) + 1, int(k_step)))
    ch_scores: Dict[int, float] = {}

    print(
        f"Selecting K via Calinski-Harabasz on train-only subset: "
        f"train_samples={X_train.shape[0]}, val_samples={X_val.shape[0]}, "
        f"grid={candidate_ks}"
    )

    for k in candidate_ks:
        print(f"  Fitting MiniBatchKMeans for k={k} ...")
        model = MiniBatchKMeans(
            n_clusters=k,
            init="k-means++",
            n_init=n_init,
            max_iter=max_iter,
            batch_size=batch_size,
            random_state=random_state,
            compute_labels=True,
            reassignment_ratio=0.01,
            verbose=0,
        )
        model.fit(X_train)
        try:
            val_labels = model.predict(X_val)
            # CH requires at least two distinct labels in the validation set
            if np.unique(val_labels).size < 2:
                raise ValueError("Only one cluster present in validation assignments")
            ch = float(calinski_harabasz_score(X_val, val_labels))
        except Exception as ex:  # noqa: BLE001
            print(f"    Warning: CH score failed for k={k}: {ex}")
            ch = float("-inf")
        ch_scores[k] = ch
        print(f"    CH(k={k}) = {ch:.2f}")
        del model
        gc.collect()

    # Choose the best K by highest CH score; tie broken by smaller K
    best_k = max(ch_scores.items(), key=lambda kv: (kv[1], -kv[0]))[0]
    print(f"Best K by CH: k={best_k}, score={ch_scores[best_k]:.2f}")
    return best_k, ch_scores


def _fit_final_model(
    train_embeddings: np.ndarray,
    k: int,
    batch_size: int,
    max_iter: int,
    n_init: int,
    random_state: int,
) -> MiniBatchKMeans:
    print(
        f"Fitting final MiniBatchKMeans on full train: n={train_embeddings.shape[0]}, "
        f"d={train_embeddings.shape[1]}, k={k}"
    )
    model = MiniBatchKMeans(
        n_clusters=k,
        init="k-means++",
        n_init=n_init,
        max_iter=max_iter,
        batch_size=batch_size,
        random_state=random_state,
        compute_labels=True,
        reassignment_ratio=0.01,
        verbose=0,
    )
    model.fit(train_embeddings)
    return model


def _merge_cluster_feature(
    structured_csv_path: str,
    assignments: pd.DataFrame,
    out_dir: str,
    k: int,
    overwrite: bool,
) -> str:
    _ensure_exists(structured_csv_path, "Structured CSV")
    df = pd.read_csv(structured_csv_path)
    if "Sample_id" in df.columns and "sample_id" not in df.columns:
        df = df.rename(columns={"Sample_id": "sample_id"})
    if "sample_id" not in df.columns:
        raise KeyError(
            f"Column 'sample_id' not found in structured CSV: {structured_csv_path}"
        )
    df["sample_id"] = df["sample_id"].astype(np.int64)

    merged = df.merge(assignments, on="sample_id", how="left", validate="one_to_one")
    if overwrite:
        out_path = structured_csv_path
    else:
        base = os.path.basename(structured_csv_path)
        name, ext = os.path.splitext(base)
        out_path = os.path.join(out_dir, f"{name}.clustered_k{k}{ext}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    merged.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Cluster DeBERTa text embeddings with MiniBatchKMeans (train-only fit) "
            "and append cluster id to structured CSVs."
        )
    )
    parser.add_argument("--cache_dir", type=str, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--train_npz",
        type=str,
        default=os.path.join(DEFAULT_CACHE_DIR, "deberta_v3_base_text_train.npz"),
    )
    parser.add_argument(
        "--test_npz",
        type=str,
        default=os.path.join(DEFAULT_CACHE_DIR, "deberta_v3_base_text_test.npz"),
    )
    parser.add_argument(
        "--structured_train_csv",
        type=str,
        default=os.path.join(DEFAULT_OUTPUT_DIR, "vlm_structured_data_train.csv"),
    )
    parser.add_argument(
        "--structured_test_csv",
        type=str,
        default=os.path.join(DEFAULT_OUTPUT_DIR, "vlm_structured_data_test.csv"),
    )
    parser.add_argument("--k", type=int, default=-1, help="If >0, skip selection and use this K")
    parser.add_argument("--select_k", action="store_true", help="Enable automatic K selection")
    parser.add_argument("--k_min", type=int, default=300)
    parser.add_argument("--k_max", type=int, default=600)
    parser.add_argument("--k_step", type=int, default=50)
    parser.add_argument("--select_k_train_samples", type=int, default=200_000)
    parser.add_argument("--select_k_val_samples", type=int, default=100_000)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument("--n_init", type=int, default=10)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument(
        "--overwrite_structured",
        action="store_true",
        help="Overwrite structured CSVs in-place instead of writing new files",
    )

    args = parser.parse_args()
    config = ClusteringConfig(**vars(args))

    # Load embeddings (DeBERTa vectors are already L2-normalized in the source script)
    _ensure_exists(config.train_npz, "Train text embeddings NPZ")
    _ensure_exists(config.test_npz, "Test text embeddings NPZ")
    train_sample_ids, train_embeddings = _load_npz_embeddings(config.train_npz)
    test_sample_ids, test_embeddings = _load_npz_embeddings(config.test_npz)

    print(
        f"Loaded TRAIN embeddings: shape={train_embeddings.shape}, TEST embeddings: shape={test_embeddings.shape}"
    )

    # Select K on train-only subset if requested and not forced
    if (config.k is None or config.k <= 0) and config.select_k:
        best_k, ch_scores = _choose_k_via_ch(
            embeddings=train_embeddings,
            k_min=config.k_min,
            k_max=config.k_max,
            k_step=config.k_step,
            select_k_train_samples=config.select_k_train_samples,
            select_k_val_samples=config.select_k_val_samples,
            batch_size=config.batch_size,
            max_iter=config.max_iter,
            n_init=max(1, config.n_init // 2),  # a bit faster for selection
            random_state=config.random_state,
        )
        chosen_k = best_k
    elif config.k and config.k > 0:
        chosen_k = int(config.k)
        ch_scores = {}
        print(f"Skipping selection; using k={chosen_k}")
    else:
        # Default if neither k nor select_k was provided
        chosen_k = 500
        ch_scores = {}
        print(f"No k provided and selection disabled. Defaulting to k={chosen_k}")

    # Fit final model on FULL train embeddings
    model = _fit_final_model(
        train_embeddings=train_embeddings,
        k=chosen_k,
        batch_size=config.batch_size,
        max_iter=config.max_iter,
        n_init=config.n_init,
        random_state=config.random_state,
    )

    # Predict clusters for train/test (no leakage)
    if hasattr(model, "labels_") and model.labels_ is not None and model.labels_.shape[0] == train_embeddings.shape[0]:
        train_clusters = model.labels_.astype(np.int32)
    else:
        train_clusters = model.predict(train_embeddings).astype(np.int32)
    test_clusters = model.predict(test_embeddings).astype(np.int32)

    # Create assignment DataFrames
    train_assignments = pd.DataFrame(
        {"sample_id": train_sample_ids, "text_kmeans_cluster_id": train_clusters}
    )
    test_assignments = pd.DataFrame(
        {"sample_id": test_sample_ids, "text_kmeans_cluster_id": test_clusters}
    )

    # Save artifacts
    os.makedirs(config.cache_dir, exist_ok=True)
    model_path = os.path.join(config.cache_dir, f"text_kmeans_mbkmeans_k{chosen_k}.joblib")
    joblib.dump(model, model_path)

    meta = {
        "k": chosen_k,
        "k_grid": list(range(int(config.k_min), int(config.k_max) + 1, int(config.k_step)))
        if args.select_k
        else [],
        "ch_scores": ch_scores,
        "batch_size": config.batch_size,
        "max_iter": config.max_iter,
        "n_init": config.n_init,
        "random_state": config.random_state,
        "train_embeddings_shape": tuple(train_embeddings.shape),
        "test_embeddings_shape": tuple(test_embeddings.shape),
        "note": "Embeddings were L2-normalized upstream (cache_new_embeddings.py).",
    }
    meta_path = os.path.join(config.cache_dir, f"text_kmeans_metadata_k{chosen_k}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    assign_train_path = os.path.join(
        config.cache_dir, f"text_kmeans_assignments_train_k{chosen_k}.csv"
    )
    assign_test_path = os.path.join(
        config.cache_dir, f"text_kmeans_assignments_test_k{chosen_k}.csv"
    )
    train_assignments.to_csv(assign_train_path, index=False)
    test_assignments.to_csv(assign_test_path, index=False)

    # Merge cluster id into structured CSVs
    train_out_csv = _merge_cluster_feature(
        structured_csv_path=config.structured_train_csv,
        assignments=train_assignments,
        out_dir=config.output_dir,
        k=chosen_k,
        overwrite=config.overwrite_structured,
    )
    test_out_csv = _merge_cluster_feature(
        structured_csv_path=config.structured_test_csv,
        assignments=test_assignments,
        out_dir=config.output_dir,
        k=chosen_k,
        overwrite=config.overwrite_structured,
    )

    # Basic reporting
    print("\n=== Clustering complete ===")
    print(f"Model saved: {model_path}")
    print(f"Metadata saved: {meta_path}")
    print(f"Assignments saved: {assign_train_path} | {assign_test_path}")
    print(
        f"Structured outputs: train -> {train_out_csv}, test -> {test_out_csv}; "
        f"added column 'text_kmeans_cluster_id' (k={chosen_k})"
    )


if __name__ == "__main__":
    main()


