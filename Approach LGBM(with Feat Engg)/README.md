# ML Challenge 2025: Smart Product Pricing — Final Report

**Team Name:** Asi baat hai kya  
**Team Members:** Ankit Meda · Pritish Saha · Atul Singh · Abhranil Mondal  
**Submission Date:** 13 Oct 2025

---

## 1. Executive Summary
We build a multimodal, feature‑engineered pricing system that converts raw product images and catalog text into structured attributes and dense features, then learns a price model with gradient boosting. Images are parsed with the vision–language model Qwen2.5‑VL and CLIP; text is represented via TF‑IDF+SVD and DeBERTa embeddings clustered into pseudo‑categories. A LightGBM regressor trained in original price space using a custom Pseudo‑Huber objective (delta estimated from the price IQR) with monotonic constraints delivers robust performance and fast inference while aligning optimization with SMAPE evaluation.

---

## 2. Methodology Overview

### 2.1 Problem Analysis
Price is driven by brand, quantity (pack count, mass/volume), and product category, with additional weak signals in descriptive text and imagery. Images often contain salient brand cues and quantity indicators; the provided `catalog_content` mixes titles, bullet points, and numeric units. We therefore extract: (a) reliable structured fields from images and text; (b) dense text/image embeddings; and (c) simple interaction features (e.g., density, per‑pack values).

Key observations from EDA:
- Catalog text is noisy but informative; char n‑grams help capture units and variants.  
- Image availability is high but not perfect; we handle missing images explicitly.  
- Price relates monotonically to mass, volume, and pack_count.

### 2.2 Solution Strategy
Approach Type: Hybrid multimodal feature‑engineering + gradient boosting  
Core Innovations:
- High‑throughput VLM extraction using Qwen2.5‑VL with robust parsing and text fallbacks, producing structured fields: brand, pack_count, mass_g, volume_ml, category, and keyword flags.  
- Complementary representations: TF‑IDF+SVD on `catalog_content`, CLIP image embeddings, and DeBERTa text embeddings clustered via MiniBatchKMeans into pseudo‑categories.  
- Training in original price space with a custom Pseudo‑Huber objective (robust to outliers, smooth gradients); delta is computed from the Inter‑Quartile Range (IQR) of the price distribution; SMAPE is used as a validation metric (feval) directly in price space; monotonic constraints enforce sensible relationships.

---

## 3. Model Architecture

### 3.1 End‑to‑End Flow
1) Data & images: download images from `image_link` (see `download_script.py`).  
2) VLM features (`feat_optimized.py`): Qwen2.5‑VL converts image+text to structured fields; responses are parsed with rule‑based guards and catalog fallbacks; keyword flags are one‑hot; a small text‑only classifier refines categories when the VLM is uncertain.  
3) Text TF‑IDF (`build_tfidf_features.py`): word (1–2) and char (3–5) n‑grams → TruncatedSVD (default 4096 dims), scaled and L2‑normalized; merged onto the VLM CSV.  
4) Embeddings cache (`cache_new_embeddings.py`): DeBERTa‑v3‑base sentence embeddings for text and CLIP ViT‑L/14 embeddings for images (L2‑normalized).  
5) Text clusters (`cluster_text_kmeans.py`): MiniBatchKMeans on train‑only DeBERTa embeddings yields `text_kmeans_cluster_id` for train/test (no leakage) and merges into the structured CSVs.  
6) Master assembly (`feature_engg.py`, `feat_cleanup.py`):
   - Train master: use `feature_engg.py` with train paths to combine VLM CSV + KMeans clusters + CLIP embeddings into `input/master_train.parquet`. Then run `feat_cleanup.py` to merge TF‑IDF+SVD features (e.g., `output/vlm_plus_tfidf_svd_train.parquet`) into the master.  
   - Test master: run `feature_engg.py` again with test paths (replace "train" with "test" in file names) to produce `input/master_test.parquet`. Optionally mirror the cleanup step by swapping paths to merge TF‑IDF+SVD test features.  
7) Training (`train_model_a.py`): LightGBM on merged features; Optuna tunes hyperparameters while minimizing SMAPE; we train in price space with a custom Pseudo‑Huber objective (delta from IQR), early stopping on SMAPE; optional OOF CV.  
8) Inference (`predict_model_a.py`): loads metadata to align categories and feature order, loads the trained LightGBM booster, and outputs `sample_id,price` in original units.

Reference for Qwen2.5‑VL: [Hugging Face model card](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct).

### 3.2 Model Components

Text Processing Pipeline
- Input: `catalog_content` from `dataset/train.csv` and `dataset/test.csv`  
- TF‑IDF: word 1–2, char 3–5, sublinear TF, up to 200k/300k features respectively; TruncatedSVD to 4096; StandardScaler + L2 normalization.  
- DeBERTa‑v3‑base sentence embeddings (L2‑normalized); MiniBatchKMeans selects K (default grid 300–600) via Calinski‑Harabasz on train‑only subsets to create `text_kmeans_cluster_id`.

Image Processing Pipeline
- Qwen2.5‑VL (3B Instruct) extracts brand, pack_count, mass_g, volume_ml, keywords list, and category from image + text prompts. The parser enforces strict brand hygiene and unit validation with catalog fallbacks.  
- CLIP ViT‑L/14 image embeddings (L2‑normalized); a missing‑image mask is stored and used during training.  
- High‑throughput implementation with async preloading, large micro‑batches, and GPU‑friendly settings.

Feature Engineering (prior to modeling)
- density = mass / (volume + ε)  
- mass_per_pack = mass / (pack_count + ε)  
- volume_per_pack = volume / (pack_count + ε)  
- total_flags = sum of keyword flags

Learning Algorithm
- Model: LightGBM regressor; target is price (no log transform).  
- Objective/metrics: Custom Pseudo‑Huber objective (delta from price IQR) for robustness; validation metric is SMAPE computed in price space.  
- Monotonic constraints: positive for `mass`, `volume`, `pack_count`.  
- HPO: Optuna tunes learning rate, num_leaves, min_data_in_leaf, regularization, and feature_fraction with early stopping, using Pseudo‑Huber objective and SMAPE feval.  
- Optional OOF K‑fold to estimate stability; feature importances are exported.

---

## 4. Model Performance
- Primary metric: SMAPE, computed on price space.  
- Validation protocol: train/validation split with early stopping on SMAPE using the custom Pseudo‑Huber objective; optional OOF CV (see `output/cv_report.json`).  
- The training script prints the final validation SMAPE; export artifacts include model, feature metadata, and importance report.  
- Public leaderboard SMAPE is produced from `sample_test_out.csv` generated by `predict_model_a.py`.

---

## 5. Conclusion
We combine complementary visual and textual signals to form a compact, high‑quality feature set and train a monotone‑aware gradient boosting model targeted directly at SMAPE. The approach is efficient (GPU‑accelerated extraction, streaming, and batched inference), avoids external price lookups, and yields stable generalization through diverse features and careful validation.

---

## Appendix

### A. Reproducibility (condensed)
1) Download images using `download_script.py` (retry logic recommended).  
2) Run `feat_optimized.py` to create `output/vlm_structured_data_{train,test}.csv`.  
3) Run `build_tfidf_features.py` to generate TF‑IDF+SVD features: saves `output/vlm_plus_tfidf_svd_{train,test}.parquet`.  
4) (Optional) Run `cache_new_embeddings.py` to create DeBERTa/CLIP NPZ caches.  
5) Run `cluster_text_kmeans.py` to append `text_kmeans_cluster_id` to the structured CSVs (produces `output/vlm_structured_data_{train,test}.clustered_k{K}.csv`).  
6) Assemble masters with `feature_engg.py`:
   - Train: set `TRAIN_DATA_CSV='output/vlm_structured_data_train.csv'`, `KMEANS_CLUSTERS_CSV='output/vlm_structured_data_train.clustered_k{K}.csv'`, `IMAGE_EMBEDDINGS_NPZ='cache_new/clip_vit_l14_image_train.npz'`, `OUTPUT_PARQUET='input/master_train.parquet'`.  
   - Test: set `TRAIN_DATA_CSV='output/vlm_structured_data_test.csv'`, `KMEANS_CLUSTERS_CSV='output/vlm_structured_data_test.clustered_k{K}.csv'`, `IMAGE_EMBEDDINGS_NPZ='cache_new/clip_vit_l14_image_test.npz'`, `OUTPUT_PARQUET='input/master_test.parquet'`.  
7) Merge TF‑IDF+SVD into train master via `feat_cleanup.py` (paths default to `input/master_train.parquet` and `output/vlm_plus_tfidf_svd_train.parquet`; adjust similarly for test if desired).  
8) Train with `train_model_a.py` (reads `input/master_train.{parquet,csv}`): the script trains in original price space with a custom Pseudo‑Huber objective and early stopping on SMAPE.  
9) Predict with `predict_model_a.py` (reads `input/master_test.{parquet,csv}`) → outputs `dataset/sample_test_out.csv` in original price units.

### B. Compliance
No external price lookup or internet augmentation is used. All features are derived from provided images and text as per the challenge rules (see `README.md`).

### C. References
- Qwen/Qwen2.5‑VL‑3B‑Instruct model card: https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct

### D. Detailed Rationale for Pseudo‑Huber in Price Space (Highly Detailed)

Motivation: severe skew and objective mismatch  
The training labels span multiple orders of magnitude (e.g., $0.13–$2,796) with high skew. Our earlier pipeline applied a log transform and optimized MSE/Huber in log space while evaluating SMAPE in price space. This created an optimization–evaluation mismatch: gradients were shaped by log‑scale errors, but metric sensitivity (SMAPE) depended on absolute price space. As a result, improvements in log space did not consistently translate into SMAPE gains.

Why not optimize SMAPE directly?  
SMAPE is scale‑normalized and symmetric, which is desirable for evaluation. However, its gradient has problematic behavior near zero: small denominators cause very large gradients for low‑price items, destabilizing training and over‑emphasizing the cheapest products. Empirically this led to noisy updates and degraded validation performance.

Pseudo‑Huber: smooth and robust  
Pseudo‑Huber loss bridges L2 (for small residuals) and L1 (for large residuals) with continuously differentiable gradients:
L(r) = δ² (√(1 + (r/δ)²) − 1), with r = ŷ − y.  
Gradients: r / √(1 + (r/δ)²); Hessian: 1 / (1 + (r/δ)²)^{3/2}.  
For |r| ≪ δ, the loss behaves like L2, encouraging tight fits; for |r| ≫ δ, it behaves like L1, limiting outlier influence. This gives stable updates and resilience to extreme prices.

Data‑driven δ from IQR  
Choosing δ is critical: too small behaves like L1 (under‑fits), too large like L2 (outlier‑sensitive). We estimate δ from the Inter‑Quartile Range (IQR) of prices to reflect dispersion robustly. Specifically, δ ≈ IQR / 1.349 (a robust std proxy), optionally scaled via environment variable `PH_DELTA_MULT`. We compute δ per training set (and per fold for OOF), adapting to distribution shifts.

Train in price space with SMAPE feval  
We remove the log transform and train directly on price. Optimization uses the Pseudo‑Huber objective with the data‑driven δ. Early stopping monitors SMAPE computed in price space via a custom evaluation function, ensuring that optimization progress correlates with the leaderboard metric.

Practical effects  
- Stable gradients across the entire price range (no explosion near zero).  
- Robustness to extreme outliers without discarding them.  
- Alignment between training objective and evaluation metric, improving convergence behavior.  
- Maintains monotonicity constraints on quantity features, preserving sensible price trends.  
- Compatible with Optuna HPO by keeping Pseudo‑Huber fixed and tuning tree parameters; early stopping uses SMAPE.

Operational notes  
- We expose `PH_DELTA_MULT` and `PH_MIN_DELTA` to adjust δ behavior if needed.  
- Models are saved as LightGBM boosters (`.txt`, `.pkl`) and inference predicts prices directly (no exp/log).  
- OOF CV computes fold‑specific δ from the training fold to avoid leakage and to reflect local dispersion.
