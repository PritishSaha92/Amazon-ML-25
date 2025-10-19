## VLM Price Prediction — Engineering Optimization Approach

### 1. Executive Summary
We fine‑tune Qwen2.5‑VL‑3B to predict product price from catalog text and images using two practical paths:
- A lean JSONL path (download → resize → chat JSONL → Unsloth SFT → inference)
- An advanced WebDataset streaming path (consumes preprocessed TAR shards)

The modeling principle: format each example as a multimodal chat (image + text) with the numeric price as the assistant response. Training uses Unsloth’s LoRA over a 4‑bit base with bf16 compute.

---

### 2. Directory Contents
- `1_download.py`, `2_resize.py`, `3_create_jsonl.py`: Data prep for the JSONL flow
- `4_preprocess.py`, `5_convert.py`: Offline preprocessing + WebDataset conversion (if maintained in this repo; in some setups these live in the sibling `DDP Approach` directory)
- `6_main.ipynb`: WebDataset streaming training/inference (consumes shards)
- `example.py`, `example.ipynb`, `example_commented.ipynb`: JSONL training flows with callbacks
- `run.sh`: Reference shell for preprocessing + sharding (adjust paths to where scripts live)
- `Optimizing VLM Training on Limited Notebook.pdf`: Notes

---

### 3. Two Execution Modes
1) JSONL (lean path):
```bash
# 1) Download images:
python 1_download.py
python 2_resize.py

# 2) Create JSONL:
python 3_create_jsonl.py --validation-split-ratio 0.1

# 3) Train/Infer:
python example.py
```

2) WebDataset (advanced path):
Prerequisites: Preprocessed shards must exist locally, e.g.:
- Train shards: `./webdataset_train/train-shard-{000000..000067}.tar`
- Validation shards (optional): `./webdataset_validation/validation-shard-{000000..000007}.tar`
- Test shards: `./webdataset_test/test-shard-{000000..000074}.tar`

How to obtain shards: Run `4_preprocess.py` and `5_convert.py` (if present here), or use the sibling `DDP Approach` which provides those scripts, to create `.npy`+metadata and pack to TAR shards.

```bash
# 1) Download images:
python 1_download.py
python 2_resize.py

# 2) Create JSONL:
python 3_create_jsonl.py --validation-split-ratio 0.1

# 3) Offline tensorization & sharding
python 4_preprocess.py --split all
python 5_convert.py --split all --samples-per-shard 1000

# 4) Open 6_main.ipynb → set WEBDATASET_*_URL → train & infer
```

---

### 4. Data and Formats
#### 4.1 Chat JSONL schema (`3_create_jsonl.py`)
Each line:
```json
{
  "sample_id": 123,
  "messages": [
    {"role": "system", "content": "You are a precise e-commerce expert..."},
    {"role": "user", "content": [
      {"type": "image", "image": "/abs/path/to/img.jpg"},
      {"type": "text",  "text":  "Analyze the following product..."}
    ]},
    {"role": "assistant", "content": "199.99"}
  ]
}
```
- Local images are absolute paths; resized copy is used if present. For test, the assistant turn is omitted.

#### 4.2 Preprocessed tensors (`4_preprocess.py`)
- `input_ids.npy`: int64 `[seq_len]`
- `attention_mask.npy`: int64 `[seq_len]`
- `pixel_values.npy`: float32 `[num_patches, hidden_dim]` (already vision‑embedded)
- `meta.json`: `{ sample_id, original_index, split, input_ids_shape, pixel_values_shape, image_grid_thw }`

#### 4.3 WebDataset sample (`5_convert.py`)
Keys per TAR sample: `input_ids.npy`, `attention_mask.npy`, `pixel_values.npy`, `metadata.json`.
In `6_main.ipynb`, samples are decoded with `np.load(BytesIO(...))`, and `image_grid_thw` is used to validate `num_patches == t*h*w`.

---

### 5. Training (WebDataset streaming)
- Model: `FastVisionModel.from_pretrained(MODEL_ID, load_in_4bit=True, dtype=bf16, use_gradient_checkpointing=False, trust_remote_code=True)`.
- LoRA: `FastVisionModel.get_peft_model(...)` on attention/MLP modules.
- Processor: `AutoProcessor.from_pretrained(..., min_pixels=256*28*28, max_pixels=512*28*28)`; ensure `pad_token` and set padding side.
- Collator fix: pad sequences, clone labels, set pad tokens to `-100`, and mask ALL special tokens `>= tokenizer.vocab_size` to `-100` to avoid out‑of‑bounds indices.
- Data: `WebDataset` → `IndexableWebDataset` shim for Unsloth; shuffle buffer for train.
- SFT config (typical): batch=4, grad‑accum=4 (effective 16); `bf16=True`, `tf32=True`, `optim="adamw_torch_fused"`; tune `dataloader_num_workers`.
- Checkpointing: `save_strategy="steps"` (e.g., every `EVAL_STEPS`) + `PeriodicCheckpointCallback`; memory cleanup callback.
- DDP: Use `torchrun` to launch multi‑GPU. The DDP‑aware SMAPE callback in `example.py` runs only on rank 0 and syncs others with `dist.barrier()`.

---

### 6. Inference and Submission
- `6_main.ipynb` streams `WEBDATASET_TEST_URL`, batches via the collator, and generates outputs; the numeric price is parsed from the generated tail (after prompt).
- Post‑processing fills any missing predictions with the training median to guarantee 75,000 rows.
- Output file: `test_out.csv` with `sample_id,price`.

---

### 7. Performance Engineering Rationale (maps to the optimization plan)
- Diagnose GPU starvation: low avg GPU util with spikes + idle CPU indicates an I/O‑bound loader.
- Quick wins: increase batch toward 90–95% VRAM, disable gradient checkpointing when VRAM headroom exists, reduce save/eval frequency, enable `pin_memory`/`persistent_workers`.
- Architectural shift: move deterministic processing offline (`4_preprocess.py`) and shard to WebDataset (`5_convert.py`) to convert random tiny file reads into sequential TAR streaming.
- Chat correctness: always use `apply_chat_template` before the processor to match training format.
- Compute: 4‑bit base + bf16 compute via Unsloth; SDPA/FlashAttention may vary by backend (Qwen2.5‑VL is handled by Unsloth’s optimized path).

---

### 8. Quickstart
```bash
# 1) (Optional) Download & resize
python 1_download.py
python 2_resize.py

# 2) JSONL (optionally with validation)
python 3_create_jsonl.py --validation-split-ratio 0.1

# 3) Offline tensorization & sharding
python 4_preprocess.py --split all
python 5_convert.py --split all --samples-per-shard 1000

# 4) Open 6_main.ipynb → set WEBDATASET_*_URL → train & infer
```

---

### 9. Troubleshooting
- Missing images: train samples without images are skipped; test samples use text‑only.
- Token/OOM: ensure collator masks tokens `>= vocab_size` to `-100`; tune batch size per VRAM; reduce if OOM.
- Slow steps/sec: confirm training from WebDataset shards; consider disabling gradient checkpointing when VRAM allows and tuning `num_workers`.

---

### 10. Compliance and References
- Compliance: No external price lookup; only provided text/images are used.
- Reference: Qwen/Qwen2.5‑VL‑3B‑Instruct (https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct).
