## VLM Price Prediction — DDP Approach (WebDataset Streaming)

### 1. Executive Summary
This approach provides two practical, end‑to‑end paths to fine‑tune `Qwen/Qwen2.5‑VL‑3B‑Instruct` for price prediction using the provided catalog text and images:
- A lean JSONL path (download → resize → chat JSONL → Unsloth SFT → inference)
- An advanced WebDataset streaming path that consumes preprocessed TAR shards (for maximum throughput)

Both paths follow the same modeling principle: format each example as a multimodal chat (image + text) where the assistant response is the numeric price string. Training uses Unsloth’s LoRA over a 4‑bit base with bf16 compute. Utilities are included to fix image data issues and to validate dataset integrity before training.

---

### 2. Directory Contents
- `1_download.py`: Parallel downloader to `../dataset/{train,test}_images` (multiprocessing pool size 100).
- `2_resize.py`: Parallel resize to long‑side 512 px → `../dataset/{train,test}_images_resized`.
- `3_create_jsonl.py`: Builds chat JSONL files from CSVs. Supports optional validation split via `--validation-split-ratio`.
- `utils.py`: Shared helpers for robust downloading (multiprocessing + progress).
- `example.py`: Full JSONL pipeline (download/resize/JSONL/stats/train/infer). Includes SMAPE callback + optional early stopping variant.
- `6_main.ipynb`: WebDataset streaming training/inference (consumes preprocessed TAR shards). Includes decoding, special‑token masking fix, periodic checkpointing, and post‑processing to complete submission.
- `test2.py`: Corrupted image repair tool (re‑download with headers and retries; prompts for confirmation; prints next steps).
- `run.sh`: Reference shell for the offline preprocessing + sharding flow (see WebDataset path prerequisites below).

Input CSVs expected in `../dataset/`:
- `train.csv`: `sample_id`, `price`, `catalog_content`, `image_link`
- `test.csv`: `sample_id`, `catalog_content`, `image_link`

---

### 3. Two Execution Modes

#### A) Direct JSONL Training (simple, self‑contained)
1) Download images
```bash
python 1_download.py
```
2) Resize images
```bash
python 2_resize.py
```
3) Create chat JSONL (optionally with a validation split)
```bash
python 3_create_jsonl.py --validation-split-ratio 0.1
```
4) Train and/or infer
- Open and run `example.py` (or `example.ipynb`)
- Training uses Unsloth SFT with LoRA on the 4‑bit base model and bf16 compute
- Optional: enable SMAPE monitoring and early stopping when running with a held‑out validation set
5) Produce submission: `test_out.csv` with `sample_id,price`

When to use: Fast iteration on a single machine, minimal setup, easy debugging.

#### B) WebDataset Streaming Training (throughput‑oriented)
This path streams preprocessed tensors from TAR shards (WebDataset) for high I/O efficiency.

Prerequisites: Preprocessed shards must exist locally, e.g.:
- Train shards: `./webdataset_train/train-shard-{000000..000067}.tar`
- Validation shards (optional): `./webdataset_validation/validation-shard-{000000..000007}.tar`
- Test shards: `./webdataset_test/test-shard-{000000..000074}.tar`

How to obtain shards: Run offline preprocessing + conversion (see the sibling “Engg Optimization Approach” or equivalent scripts) to create `.npy` and metadata per sample and pack them into TAR shards. The `run.sh` here shows the intended sequence, but the actual `4_preprocess.py` and `5_convert.py` may live in the Engg Optimization Approach directory after your rename.

Steps:
1) Ensure shards exist at the URLs used inside `6_main.ipynb` (set the patterns at the top of the notebook)
2) Open `6_main.ipynb` and run cells to:
   - Load processor with pixel bounds (`min_pixels=256*28*28`, `max_pixels=512*28*28`)
   - Create WebDataset datasets (`create_webdataset`) and wrap train with `IndexableWebDataset` (adds minimal `__len__`/`__getitem__` caching for Unsloth)
   - Train with Unsloth SFT (LoRA over 4‑bit base, bf16 compute)
   - Run streaming inference over test shards and write `test_out.csv`
3) Post‑processing fills any missing predictions with the training median to guarantee 75,000 rows

When to use: Max throughput, stable feeding of the GPU once shards have been prepared.

---

### 4. Data Formats and Shapes

#### 4.1 Chat JSONL (used by JSONL path)
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
- For test, the assistant turn is omitted. If an image is missing, test entries become text‑only.

#### 4.2 WebDataset sample (used by streaming path)
Each TAR sample contains:
- `input_ids.npy` (int64, shape `[seq_len]`)
- `attention_mask.npy` (int64, `[seq_len]`)
- `pixel_values.npy` (float32, `[num_patches, hidden_dim]`)
- `metadata.json` with `{ sample_id, input_ids_shape, pixel_values_shape, image_grid_thw }`

`6_main.ipynb` decodes with `np.load(BytesIO(...))` and verifies `num_patches == (t*h*w)` using `image_grid_thw`.

---

### 5. Training Details

#### JSONL path (example.py / example.ipynb)
- Model loading (Unsloth): 4‑bit base + bf16 compute; LoRA on attention/MLP modules; gradient checkpointing default OFF (enable only if VRAM constrained)
- Processor: ensures `pad_token`, sets right padding; applies `tokenizer.apply_chat_template` per example
- Collator: batches formatted texts + lazily loaded images via `AutoProcessor`, then builds labels; masks PAD tokens to `-100`
- SFTConfig: typical settings
  - `per_device_train_batch_size=2..8`, `gradient_accumulation_steps=1..4`
  - `optim="adamw_torch_fused"`, `bf16=True`, `tf32=True`
  - Dataloader: `dataloader_num_workers=8..16`, `pin_memory=True`, `persistent_workers=True`
  - Checkpoints: `save_strategy="epoch"` or with a periodic callback (keeps one rolling checkpoint)
- Optional SMAPE evaluation: Callback computes SMAPE on a subset of the validation set and can drive early stopping

#### WebDataset path (6_main.ipynb)
- Decoding: `decode_sample` reads `.npy` tensors; wrapper `IndexableWebDataset` provides minimal indexability for Unsloth initialization
- Special‑token masking fix: collator pads and builds labels, then masks ANY token `>= tokenizer.vocab_size` to `-100` to avoid out‑of‑bounds label indices
- Unsloth patch: notebook patches `unsloth_zoo.tokenizer_utils.fix_untrained_tokens` to a no‑op to avoid startup scans over the dataset
- SFTConfig (typical in the notebook):
  - `per_device_train_batch_size=4`, `gradient_accumulation_steps=4` (effective 16)
  - `bf16=True`, `optim="adamw_torch_fused"`, `tf32=True`
  - Dataloader often starts with `num_workers=0`, `pin_memory=False` (tune per system; increasing workers may help once storage isn’t the bottleneck)
  - Periodic checkpoint callback (keeps a rolling checkpoint every N steps)
- Debug utilities (cells):
  - Inspect a raw sample’s shapes (`input_ids`, `pixel_values`, `image_grid_thw`)
  - Confirm special‑token masking (vision tokens not left in labels)
  - Validate `image_grid_thw` vs `pixel_values` patch count over several samples

---

### 6. Inference and Submission
- JSONL path: batches test JSONL; applies chat template; generates; strips prompt; parses float; writes `test_out.csv`
- WebDataset path: streams test shards; collates preprocessed tensors; generates; strips prompt; parses float; writes `test_out.csv`
- Post‑processing in the WebDataset notebook: fills any missing predictions with the training median to guarantee 75,000 rows

---

### 7. Data Quality Toolkit
- `test2.py`: Scans for 0‑byte images; re‑downloads with proper user‑agent headers and retries; deletes corrupted originals and resized copies before repairing; prints permanent 404s and recommended next steps (re‑resize, rebuild JSONL, re‑preprocess/re‑shard if using WebDataset)
- Dataset statistics (in `example.py`): counts of downloaded/resized images, JSONL entries with/without images, and gaps (missed downloads, resize failures, skipped JSONL rows)

---

### 8. Performance Playbook (ties to the optimization plan)
- Diagnose GPU starvation: use quick step‑timing or watch utilization; if GPU idles while CPU is ~idle, you’re I/O‑bound
- Quick wins: raise batch size toward ~90–95% VRAM; disable gradient checkpointing when headroom exists; reduce checkpoint/eval frequency; set `pin_memory=True` and `persistent_workers=True` for JSONL path
- Architect for throughput: prefer WebDataset shards from offline preprocessing for larger runs; stream them with WebDataset to convert random small reads into sequential tar I/O
- Chat correctness: always use the model’s chat template before processor calls

---

### 9. Troubleshooting
- Corrupted or missing images: run `python test2.py`, then re‑run resize and JSONL; if using WebDataset shards, re‑preprocess/re‑convert afterwards
- CUDA OOM: reduce per‑device batch or accumulation; keep gradient checkpointing OFF when you can afford memory for better throughput
- Label index errors: ensure the special‑token masking fix is present (mask tokens `>= vocab_size` to `-100`)
- Slow steps/sec: verify you are training from shards (for streaming path) and increase `num_workers` if CPU is under‑utilized

---

### 10. Compliance and References
- Compliance: No external price lookup; only provided images and text are used
- Reference: Qwen/Qwen2.5‑VL‑3B‑Instruct (https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
