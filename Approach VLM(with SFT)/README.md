# VLM Fine-Tuning with Supervised Fine-Tuning (SFT)

**Team Name:** Asi baat hai kya  
**Team Members:** Ankit Meda · Pritish Saha · Atul Singh · Abhranil Mondal

---

## Overview

This directory contains our Vision-Language Model (VLM) fine-tuning experiments using **Qwen2.5-VL-3B-Instruct** for price prediction. Due to severe time constraints during the challenge, we explored two different optimization approaches to maximize training throughput and model performance.

---

## Two Optimization Approaches

### 1. **DDP Approach** (`DDP Approach/`)
Focused on distributed training with Distributed Data Parallel (DDP) and WebDataset streaming for high I/O throughput. This approach prioritizes scaling across multiple GPUs and efficient data loading through preprocessed TAR shards.

👉 **See detailed documentation:** [`DDP Approach/README.md`](DDP%20Approach/README.md)

### 2. **Engg Optimization Approach** (`Engg Optimization Approach/`)
Emphasizes engineering optimizations for training on limited resources (single GPU / notebook environments). Includes offline preprocessing, memory-efficient techniques, and practical workarounds for resource-constrained setups.

👉 **See detailed documentation:** [`Engg Optimization Approach/README.md`](Engg%20Optimization%20Approach/README.md)

---

## Why Two Approaches?

We had very limited time during the challenge and needed to experiment quickly with different optimization strategies:
- **DDP Approach**: For scenarios where we had access to multiple GPUs or cloud resources
- **Engg Optimization Approach**: For local development and rapid iteration on single-GPU setups

Both approaches share the same core idea (fine-tuning Qwen2.5-VL for price prediction) but differ in their data pipeline and training configurations.

---

## Quick Start

Choose the approach that matches your hardware setup:

- **If you have multiple GPUs or cloud resources:** Use the **DDP Approach**
- **If you have a single GPU or limited resources:** Use the **Engg Optimization Approach**

Both approaches include:
- Image download and preprocessing scripts
- JSONL dataset creation
- Training notebooks and scripts
- Inference pipelines

Refer to the respective README files for detailed setup and execution instructions.

---

## Model Details

- **Base Model:** Qwen/Qwen2.5-VL-3B-Instruct
- **Training Method:** LoRA (Low-Rank Adaptation) with Unsloth
- **Task:** Multimodal price prediction from product images + catalog text
- **Input Format:** Chat-style messages with image and text
- **Output:** Numeric price prediction

---

## Note

Due to time constraints during the challenge, both approaches contain experimental code and various optimization attempts. We recommend starting with the detailed READMEs in each subdirectory to understand the specific workflow and requirements.

