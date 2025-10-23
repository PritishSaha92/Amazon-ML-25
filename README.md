# ML Challenge 2025: Smart Product Pricing — Master Report

**Team Name:** Asi baat hai kya  
**Team Members:** Ankit Meda · Pritish Saha · Atul Singh · Abhranil Mondal  
**Submission Date:** 13 Oct 2025

---

## 1. Executive Summary
We develop a multimodal pricing system combining a feature‑engineered LightGBM model (Model A) with a vision‑language model fine‑tuned under a high‑throughput DDP/WebDataset pipeline (Model B). Model A trains in original price space using a custom Pseudo‑Huber objective (delta from price IQR) with SMAPE validation and monotonic constraints. Final predictions are produced by a stacking meta‑learner trained on OOF predictions plus global features, yielding robust generalization and competitive SMAPE.

---

## 2. Implementation‑Level Engineering Diagram

![Implementation-Level Engineering Diagram](Implementation-Level%20Engineering%20Diagram.png)

## 3. Methodology Overview

### 3.1 Problem Analysis
Price depends on brand, quantity (mass/volume/pack), and category, with additional weak signals spread across noisy catalog text and product images. Images encode brand cues and quantities; text often contains units and variants. Our strategy extracts structured attributes and dense representations from both modalities, then learns price with models targeted directly at minimizing SMAPE in price space.

**Key Observations:**
- Catalog text is noisy but informative; char n‑grams help capture units and variants.
- Image availability is high but not perfect; missing‑image handling is necessary.
- Price relates monotonically to mass, volume, and pack_count.

### 3.2 Solution Strategy
**Approach Type:** Hybrid, two strong base models + stacking meta‑learner  
**Core Innovation:**
- High‑fidelity VLM data pipeline (offline preprocessing → WebDataset shards → DDP streaming) enabling stable, high‑throughput fine‑tuning of Qwen2.5‑VL for price extraction from image+text.
- Rich feature‑engineered LightGBM with monotonic constraints on quantity features, TF‑IDF+SVD, CLIP and DeBERTa signals, and KMeans pseudo‑categories.
- Stacking meta‑learner trained on OOF predictions plus global features to blend modalities effectively without leakage.

---

## 4. Model Architecture

### 4.1 Architecture Overview

For a deeper, step‑by‑step theoretical explanation of every component and training phase, see [Detailed Explanation (PDF)](Detailed%20Explanation.pdf).

- Data sources: `dataset/train.csv`, `dataset/test.csv`, `images/train|test` from `image_link`.
- Model A (LGBM): feature‑engineered multimodal tabular model trained in original price space with a custom Pseudo‑Huber objective (delta from price IQR), SMAPE validation, and monotonic constraints.
 - Model A (LGBM): feature‑engineered multimodal tabular model trained in original price space with a custom Pseudo‑Huber objective (delta from price IQR), validated by SMAPE, and constrained monotonically on quantity features.
- Model B (VLM DDP): Qwen2.5‑VL‑3B fine‑tuned via Unsloth LoRA using DDP and WebDataset streaming of preprocessed tensors.
- Meta‑learner: trained on OOF predictions from Models A and B + global features; blends fold‑wise test predictions to produce final `test_out.csv`.

### 4.2 Model Components

**Text Processing Pipeline:**
- Preprocessing: TF‑IDF (word 1–2, char 3–5, sublinear TF), TruncatedSVD→4096 dims, StandardScaler + L2 normalization.
- Embeddings: DeBERTa‑v3‑base sentence embeddings (L2‑normalized).
- Clustering: MiniBatchKMeans on train‑only embeddings to produce `text_kmeans_cluster_id` (no leakage).

**Image Processing Pipeline:**
- VLM extraction (primary): Qwen2.5‑VL (3B Instruct) takes image + catalog text via a chat prompt and returns structured fields (brand, pack_count, mass_g, volume_ml, category, keywords). Outputs are validated and normalized (units/ranges); no OCR or heavy rule‑based parsing is used.
- CLIP: ViT‑L/14 embeddings (L2‑normalized) for images; missing‑image mask preserved.
- Offline VLM tensorization (for Model B): `AutoProcessor` converts chat examples into `input_ids`, `attention_mask`, and patch embeddings `pixel_values` saved as `.npy`; metadata `image_grid_thw` saved alongside.

**Tabular Feature Engineering (for Model A):**
- density = mass / (volume + ε)
- mass_per_pack = mass / (pack_count + ε)
- volume_per_pack = volume / (pack_count + ε)
- total_flags = sum(keyword flags)

**Learning:**
- Model A: LightGBM regressor in original price space with a custom Pseudo‑Huber objective (delta from price IQR), SMAPE validation, monotonic constraints on mass/volume/pack_count, Optuna HPO; end‑to‑end training and inference remain in price space (no log/expm1 conversions).
- Model B: Qwen2.5‑VL‑3B with Unsloth LoRA (4‑bit base, bf16); DDP/WebDataset streaming; periodic checkpointing; SMAPE monitoring utility.
- Meta‑learner: regression model trained on OOF predictions [A,B] + global features (e.g., unit totals), validated fold‑wise; test predictions are blended per fold. The `ensemble.py` script has been updated to operate fully in price space and read Model A OOF from `Approach LGBM(with Feat Engg)/output/oof_predictions.csv`.

## 5. Conclusion
A hybrid, production‑minded pipeline combines a feature‑engineered LightGBM with a high‑throughput VLM fine‑tune and a stacking meta‑learner. Offline preprocessing and WebDataset streaming remove I/O bottlenecks; monotonic constraints and careful validation stabilize generalization; stacking integrates complementary signals to achieve low SMAPE.

---

## Appendix

### A. Code Artifacts and Directory Map
- LGBM (Model A): `Approach LGBM(with Feat Engg)/`
  - Data assembly: `feature_engg.py`, `feat_cleanup.py`
  - Text: `build_tfidf_features.py`
  - Embeddings: `cache_new_embeddings.py`
  - Clustering: `cluster_text_kmeans.py`
  - Training: `train_model_a.py`
  - Inference: `predict_model_a.py`
  - Readme: `Approach LGBM(with Feat Engg)/README.md`
- VLM DDP (Model B): `Approach VLM(with SFT)/DDP Approach/`
  - Image I/O: `1_download.py`, `2_resize.py`
  - JSONL: `3_create_jsonl.py`
  - Offline preprocess + shards: see DDP/Engg readmes; WebDataset consumed by `6_main.ipynb`
  - Training/Inference: `6_main.ipynb`, `example.py`
  - Utilities: `run.sh`, `test2.py`, `utils.py`
  - Readme: `Approach VLM(with SFT)/DDP Approach/README.md`
- Meta‑learner: `ensemble.py` (stacking on OOF predictions + global features; produces final `test_out.csv`)

### B. Reproducibility (High‑level)
1) Prepare LGBM masters and train Model A  
   - Build structured CSVs and TF‑IDF+SVD → `input/master_train.parquet`/`input/master_test.parquet`  
   - Train `train_model_a.py` (Pseudo‑Huber in price space) → saves booster + metadata; infer `predict_model_a.py` (price space) → OOF + test predictions
2) Prepare VLM DDP and train Model B  
   - Download/resize images; create JSONL  
   - Offline preprocess and shard (or use existing shards)  
   - Train with `6_main.ipynb` → save checkpoints; generate OOF (if applicable) + test predictions
3) Train stacking meta‑learner  
   - Collect OOF predictions from A and B, plus global features  
   - Train `ensemble.py` with K‑fold; validate on folds; blend test predictions fold‑wise  
   - Write final `dataset/sample_test_out.csv` (or `test_out.csv`)

### C. Implementation Notes
- Price space throughout: Model A, inference, and ensemble operate entirely in price space (no log/expm1), eliminating objective–metric mismatch and instability near zero.
- Ensemble input expectations: Model A OOF at `Approach LGBM(with Feat Engg)/output/oof_predictions.csv` (column `oof_price`), Model A test at `Approach LGBM(with Feat Engg)/dataset/sample_test_out.csv` (column `price`).
- VLM tokens: collator masks all special tokens `>= vocab_size` to `-100` to prevent OOB labels during loss.
- Throughput: prefer WebDataset streaming (sequential TAR I/O) with tuned dataloader workers; disable gradient checkpointing when VRAM headroom exists; raise batch size to ~90–95% VRAM.
- Monotonic constraints: enforce positive monotonicity on quantity features in LGBM to reflect real‑world trends.

### D. References
- Qwen/Qwen2.5‑VL‑3B‑Instruct: https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct

---

**Note:** This report summarizes and aligns details from the three component READMEs: LGBM (Model A), VLM DDP (Model B), and this master ensemble description. Refer to component READMEs for full scripts and commands.
