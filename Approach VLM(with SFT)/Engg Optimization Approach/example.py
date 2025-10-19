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
from PIL import Image
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["UNSLOTH_DISABLE_TRAINER_PATCHING"] = "1"  
os.environ["UNSLOTH_NO_CUDA_EXTENSIONS"] = "1"
os.environ["TORCH_DISTRIBUTED_STATIC_GRAPH"] = "1"

warnings.filterwarnings("ignore")

DATASET_FOLDER = "../dataset/"


# In[2]:


FULL_TRAINING_MODE = True

# Set to True to run inference on test set and generate submission
RUN_INFERENCE = True

# Set to True to skip dataset creation if already created
SKIP_DATASET_CREATION = False

# Set to True to skip image downloading if already done
SKIP_IMAGE_DOWNLOAD = False

# INFERENCE_CHECKPOINT = "ama-3/src/qwen2.5-vl-3b-price-predictor-best/checkpoint-15"  # None = use default output_dir
INFERENCE_CHECKPOINT = None

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


# In[4]:


import json
import random

print("=" * 60)
print("Loading Pre-Split Datasets...")
print("=" * 60)

# Load training split
print("\n📂 Loading training data from train_split.jsonl...")
train_dataset = []
with open(f"{DATASET_FOLDER}/train_split.jsonl", 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f, 1):
        try:
            train_dataset.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"⚠️ Warning: Skipping malformed JSON on line {line_num}: {e}")
            continue

print(f"✅ Loaded {len(train_dataset)} training samples")

# Load validation split
print("\n📂 Loading validation data from validation_split.jsonl...")
validation_dataset = []
with open(f"{DATASET_FOLDER}/validation_split.jsonl", 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f, 1):
        try:
            validation_dataset.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"⚠️ Warning: Skipping malformed JSON on line {line_num}: {e}")
            continue

print(f"✅ Loaded {len(validation_dataset)} validation samples")

print("\n" + "=" * 60)
print("✅ Datasets ready for DDP training!")
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(validation_dataset)}")
print("=" * 60)


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


import torch
import torch.distributed as dist
from transformers import TrainerCallback
from PIL import Image
import numpy as np
import gc

class SMAPEEvaluationCallback(TrainerCallback):
    """
    DDP-aware SMAPE evaluation callback on a SUBSET of data.
    - Runs SMAPE computation ONLY on rank 0 (main GPU) on the first 500 samples.
    - All other GPUs wait at a barrier.
    - Evaluates every `eval_steps` steps for a quick performance pulse.
    """
    def __init__(self, validation_dataset, processor, eval_steps=150, eval_subset_size=50):
        self.validation_dataset = validation_dataset
        self.processor = processor
        self.eval_steps = eval_steps
        self.eval_subset_size = eval_subset_size
        self.smape_history = []
        self.best_smape = float('inf')
        
    def on_step_end(self, args, state, control, **kwargs):
        # Only evaluate at specified intervals and if global_step is not 0
        if state.global_step == 0 or state.global_step % self.eval_steps != 0:
            return control
        
        is_distributed = dist.is_initialized()
        is_main_process = not is_distributed or dist.get_rank() == 0
        
        if is_main_process:
            print(f"\n{'=' * 60}")
            print(f"🎯 SMAPE Quick Evaluation at Step {state.global_step} (on first {self.eval_subset_size} samples)")
            print(f"{'=' * 60}")
            
            model = kwargs.get("model")
            if model is None:
                print("⚠️ Warning: Model not found in callback kwargs")
                if is_distributed:
                    dist.barrier()
                return control
            
            model.eval()
            
            predictions = []
            actuals = []
            
            # HERE'S THE MAGIC: We slice the dataset!
            eval_data = self.validation_dataset[:self.eval_subset_size]
            
            with torch.no_grad():
                for i, example in enumerate(eval_data):
                    try:
                        messages = example["messages"]
                        text = self.processor.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                        
                        images = []
                        for msg in messages:
                            if isinstance(msg["content"], list):
                                for item in msg["content"]:
                                    if isinstance(item, dict) and item.get("type") == "image":
                                        img_path = item["image"]
                                        try:
                                            img = Image.open(img_path).convert("RGB")
                                            images.append(img)
                                        except Exception:
                                            continue
                        
                        inputs = self.processor(
                            text=[text],
                            images=[images if images else None],
                            return_tensors="pt",
                            padding=True,
                        ).to(model.device)
                        
                        generated_ids = model.generate(
                            **inputs, max_new_tokens=20, num_beams=1, do_sample=False,
                            pad_token_id=self.processor.tokenizer.pad_token_id,
                            eos_token_id=self.processor.tokenizer.eos_token_id,
                        )
                        
                        generated_ids_trimmed = generated_ids[0][len(inputs.input_ids[0]):]
                        prediction = self.processor.decode(
                            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                        ).strip()
                        
                        try:
                            pred_price = float(prediction)
                        except (ValueError, TypeError):
                            pred_price = 0.0
                        
                        actual_price = None
                        for msg in messages:
                            if msg["role"] == "assistant":
                                try:
                                    actual_price = float(msg["content"])
                                except (ValueError, TypeError):
                                    continue
                        
                        if actual_price is not None:
                            predictions.append(pred_price)
                            actuals.append(actual_price)
                        
                        del inputs, generated_ids
                        
                        if (i + 1) % 100 == 0:
                            print(f"  Evaluated {i + 1}/{len(eval_data)} samples...")
                            
                    except Exception as e:
                        print(f"⚠️ Error processing sample {i}: {e}")
                        continue
            
            if len(predictions) > 0:
                smape = self._calculate_smape(actuals, predictions)
                self.smape_history.append(smape)
                
                if smape < self.best_smape:
                    self.best_smape = smape
                    print(f"\n🎉 New best SMAPE (on subset): {smape:.2f}% (improved!)")
                else:
                    print(f"\n📊 Current SMAPE (on subset): {smape:.2f}% (best: {self.best_smape:.2f}%)")
                
                print(f"Samples evaluated: {len(predictions)}")
            else:
                print("\n⚠️ No valid predictions - SMAPE not calculated")
            
            torch.cuda.empty_cache()
            gc.collect()
            model.train()
            print(f"{'=' * 60}\n")
        
        if is_distributed:
            dist.barrier()
            if not is_main_process:
                print(f"[Rank {dist.get_rank()}] ⏸️  Waited for SMAPE evaluation to complete")
        
        return control
    
    def _calculate_smape(self, actual, predicted):
        actual = np.array(actual)
        predicted = np.array(predicted)
        numerator = np.abs(predicted - actual)
        denominator = (np.abs(actual) + np.abs(predicted)) / 2
        mask = denominator != 0
        smape_values = np.zeros_like(numerator)
        smape_values[mask] = numerator[mask] / denominator[mask]
        return np.mean(smape_values) * 100




# In[ ]:


from transformers import TrainerCallback
import torch

print("🚀 Loading Qwen2.5-VL-3B with Unsloth optimization...")
model, tokenizer = FastVisionModel.from_pretrained(
    model_name=model_id,
    max_seq_length=384,
    load_in_4bit=True,
    dtype=torch.bfloat16,
    # use_gradient_checkpointing=True,
    use_gradient_checkpointing=True,
    trust_remote_code=True,
    device_map=f"cuda:{os.environ.get('LOCAL_RANK', 0)}",
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
    use_gradient_checkpointing="unsloth",
    # use_gradient_checkpointing=True,
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
processor.tokenizer.padding_side = "left"
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
                        try:
                            img = Image.open(item["image"]).convert("RGB")
                            images.append(img)
                        except Exception as e:
                            print(f"⚠️ Error loading image: {e}")
                            continue
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
    per_device_train_batch_size=16,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,  # Effective batch = 16
    # Memory optimization
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant': False},
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
    logging_steps=5,
    logging_first_step=True,
    # Evaluation & checkpointing
    eval_strategy="no",
    save_strategy="steps",
    save_steps=15,
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
    ddp_find_unused_parameters=True,
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
import signal
import torch.distributed as dist
from transformers import TrainerCallback

class PeriodicCheckpointCallback(TrainerCallback):
    """
    DDP-safe periodic checkpoint callback.
    - Saves every N steps (step-based)
    - Only rank 0 saves the checkpoint
    - Other ranks wait at barrier
    - Handles keyboard interrupt gracefully
    """
    def __init__(self, save_steps=250, checkpoint_dir="periodic_checkpoint"):
        self.save_steps = save_steps
        self.checkpoint_dir = checkpoint_dir
        self.last_save_step = 0
        self.saving_in_progress = False
        
    def on_step_end(self, args, state, control, **kwargs):
        # Check if it's time to save
        if state.global_step - self.last_save_step < self.save_steps:
            return control
        
        # Check if we're in DDP mode
        is_distributed = dist.is_initialized()
        is_main_process = not is_distributed or dist.get_rank() == 0
        
        if is_main_process:
            self.saving_in_progress = True  # Flag to prevent Ctrl+C during save
            
            checkpoint_path = os.path.join(args.output_dir, self.checkpoint_dir)
            
            # Remove old periodic checkpoint
            if os.path.exists(checkpoint_path):
                print(f"\n🔄 Replacing periodic checkpoint at step {state.global_step}")
                shutil.rmtree(checkpoint_path)
            else:
                print(f"\n💾 Creating periodic checkpoint at step {state.global_step}")
            
            # Get model from kwargs
            model = kwargs.get("model")
            if model is None:
                print("⚠️ Warning: Model not found in callback kwargs, skipping checkpoint")
                self.saving_in_progress = False
                if is_distributed:
                    dist.barrier()
                return control
            
            try:
                # Save model checkpoint
                model.save_pretrained(checkpoint_path)
                
                # Save trainer state
                state_path = os.path.join(checkpoint_path, "trainer_state.json")
                with open(state_path, 'w') as f:
                    state_dict = {
                        'epoch': state.epoch,
                        'global_step': state.global_step,
                        'max_steps': state.max_steps,
                        'num_train_epochs': state.num_train_epochs,
                        'log_history': state.log_history[-10:],  # Keep last 10 entries
                        'best_metric': state.best_metric,
                        'best_model_checkpoint': state.best_model_checkpoint,
                    }
                    json.dump(state_dict, f, indent=2)
                
                print(f"✅ Periodic checkpoint saved (step {state.global_step})")
                self.last_save_step = state.global_step
                
            except Exception as e:
                print(f"❌ Error saving checkpoint: {e}")
            finally:
                self.saving_in_progress = False
        
        # CRITICAL: Synchronize all processes
        if is_distributed:
            dist.barrier()  # All GPUs wait until rank 0 finishes saving
            if not is_main_process:
                print(f"[Rank {dist.get_rank()}] ⏸️  Waited for checkpoint save to complete")
        
        return control


# In[ ]:


import signal
import sys

class GracefulInterruptHandler:
    """
    Handles Ctrl+C gracefully - waits if checkpoint is being saved
    """
    def __init__(self, checkpoint_callback):
        self.checkpoint_callback = checkpoint_callback
        self.interrupted = False
        signal.signal(signal.SIGINT, self.handle_interrupt)
    
    def handle_interrupt(self, signum, frame):
        if self.checkpoint_callback.saving_in_progress:
            print("\n⚠️  Checkpoint save in progress... Please wait!")
            print("(Forcing exit may corrupt the checkpoint)")
            return
        
        self.interrupted = True
        print("\n🛑 Keyboard interrupt received. Stopping training gracefully...")
        print("(Last periodic checkpoint is safe to use)")
        sys.exit(0)


# In[ ]:


trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=None,
    data_collator=collate_fn,
    processing_class=processor.tokenizer,
)
trainer.add_callback(MemoryCleanupCallback())

# ADD PERIODIC CHECKPOINT CALLBACK (Always enabled!)
periodic_checkpoint = PeriodicCheckpointCallback(
    save_steps=1000,  # Save every 10 steps
    checkpoint_dir="periodic_checkpoint"  # Single rolling checkpoint
)
trainer.add_callback(periodic_checkpoint)
interrupt_handler = GracefulInterruptHandler(periodic_checkpoint)

# if trainer.args.distributed_state.num_processes > 1:
#     print("🔧 Enabling static graph for DDP...")
#     trainer.model._set_static_graph()
#     print("✅ Static graph enabled!")

# trainer.accelerator.ddp_handler.ddp_kwargs['static_graph'] = True

smape_callback = SMAPEEvaluationCallback(
    validation_dataset=validation_dataset,
    processor=processor,
    eval_steps=2000  # Evaluate every 150 steps
)
trainer.add_callback(smape_callback)


print("✅ Periodic checkpoint callback added (every 10 steps)")

print("\n" + "=" * 50)
print("Trainer initialized successfully!")
print("=" * 50)


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
from tqdm import tqdm
import torch.distributed as dist
from accelerate import Accelerator
from peft import PeftModel

if RUN_INFERENCE:
    accelerator = Accelerator()

    accelerator.print("\n" + "=" * 60)
    accelerator.print("🚀 Running DISTRIBUTED Inference on Test Set...")
    accelerator.print(f"🔥 Unleashing the power of {accelerator.num_processes} GPUs!")
    accelerator.print("=" * 60)

    test_dataset = []
    if accelerator.is_main_process:
        print(f"\n📂 Main process loading test data from {DATASET_FOLDER}/test.jsonl...")
    with open(f"{DATASET_FOLDER}/test.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            try: test_dataset.append(json.loads(line))
            except json.JSONDecodeError: continue
    
    accelerator.print(f"✅ Full test dataset with {len(test_dataset)} samples loaded into memory.")

    checkpoint_path = INFERENCE_CHECKPOINT if INFERENCE_CHECKPOINT else output_dir
    
    # Load base model ONLY if necessary (not already in memory from training)
    try:
        _ = model.device
    except NameError:
        accelerator.print("Base model not in memory. Loading fresh for inference...")
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, quantization_config=bnb_config, trust_remote_code=True,
            # device_map handled by accelerator
        )

    # Apply PEFT adapter and prepare with accelerator
    if os.path.exists(f"{checkpoint_path}/adapter_model.safetensors"):
        accelerator.print(f"*** Loading adapter from: {checkpoint_path}")
        model = PeftModel.from_pretrained(model, checkpoint_path)
    else:
        accelerator.print(f"*** Could not find adapter at {checkpoint_path}, using base model.")

    model.eval()
    model, processor = accelerator.prepare(model, processor)

    predictions, sample_ids = [], []
    batch_size = 16

    with accelerator.split_between_processes(test_dataset) as sharded_dataset:
        for batch_start in tqdm(range(0, len(sharded_dataset), batch_size), 
                                desc=f"Inference on GPU {accelerator.process_index}", 
                                disable=not accelerator.is_main_process):
            
            batch_examples = sharded_dataset[batch_start : batch_start + batch_size]
            batch_sample_ids = [ex["sample_id"] for ex in batch_examples]
            batch_messages = [ex["messages"] for ex in batch_examples]
    
            batch_texts, batch_images = [], []
            for messages in batch_messages:
                batch_texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
                images = []
                for msg in messages:
                    if isinstance(msg["content"], list):
                        for item in msg["content"]:
                            if item.get("type") == "image":
                                try: images.append(Image.open(item["image"]).convert("RGB"))
                                except: continue
                batch_images.append(images or None)
    
            inputs = processor(text=batch_texts, images=batch_images, return_tensors="pt", padding=True)
            
            # 💖💖💖 FIX #2: EXPLICITLY MOVE ALL INPUT TENSORS TO THE CORRECT GPU 💖💖💖
            inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
            # 💖💖💖💖💖💖💖💖💖💖💖💖💖💖💖💖💖💖💖💖💖💖💖💖💖💖💖💖💖💖💖💖💖💖💖

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs, max_new_tokens=20, pad_token_id=processor.tokenizer.pad_token_id,
                )
    
            decoded_preds = processor.batch_decode(generated_ids, skip_special_tokens=True)
            
            for i, full_pred in enumerate(decoded_preds):
                # We need to strip the prompt from the prediction
                prompt_len = len(processor.decode(inputs['input_ids'][i], skip_special_tokens=True))
                prediction_text = full_pred[prompt_len:].strip()

                try:
                    predicted_price = float(prediction_text)
                except (ValueError, TypeError):
                    predicted_price = 0.0
    
                predictions.append(predicted_price)
                sample_ids.append(batch_sample_ids[i])
            
    gathered_predictions = accelerator.gather_for_metrics(predictions)
    gathered_sample_ids = accelerator.gather_for_metrics(sample_ids)

    if accelerator.is_main_process:
        submission_df = pd.DataFrame({"sample_id": gathered_sample_ids, "price": gathered_predictions})
        submission_path = "test_out.csv"
        submission_df.to_csv(submission_path, index=False)
        print(f"\n{'=' * 60}\n✅ Submission saved to: {submission_path}\n")

