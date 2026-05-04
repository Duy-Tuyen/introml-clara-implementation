"""
scripts/finetune_apple.py — Fine-tune CLaRa using Apple's native modeling_clara.py.

Fine-tunes query_reasoner_adapter + decoder_adapter LoRA weights (Stage II)
starting from the Apple CLaRa-7B-E2E pretrained checkpoint.

The encoder_adapter is FROZEN (no gradients) — matching the paper's Stage II
design where the compressor is fixed after SCP pretraining.

Environment variables (set by the notebook):
    CLARA_CKPT_PATH      : Path to Apple checkpoint dir (required)
    CLARA_DATASET        : 'squad' | 'triviaqa' | 'hotpotqa' | 'nq'
    CLARA_N_TRAIN        : Number of training samples (default: 2000)
    CLARA_N_VAL          : Number of validation samples (default: 200)
    CLARA_FT_LR          : Learning rate (default: 5e-6)
    CLARA_FT_EPOCHS      : Number of epochs (default: 1)
    CLARA_FT_GRAD_ACC    : Gradient accumulation steps (default: 8)
    CLARA_FT_MAX_DEC_LEN : Max decoder sequence length (default: 128)
    CLARA_OUTPUT_DIR      : Where to save fine-tuned checkpoint
    CLARA_MODEL_VERSION   : Version string for logging
"""

from __future__ import annotations

import gc
import math
import os
import copy
import shutil
import json
import random
from datetime import datetime
from typing import List, Dict, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModel, AutoConfig, get_cosine_schedule_with_warmup


# ═══════════════════════════════════════════════════════════════════════════════
# 1. WORKDIR ASSEMBLY  (reuse logic from evaluate_apple.py)
# ═══════════════════════════════════════════════════════════════════════════════

def assemble_workdir(ckpt_path: str) -> str:
    """Create a writable working copy of the Apple checkpoint."""
    work_dir = "/kaggle/working/apple-ft-workdir"

    if os.path.isdir(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir, exist_ok=True)

    for item in os.listdir(ckpt_path):
        src = os.path.join(ckpt_path, item)
        dst = os.path.join(work_dir, item)
        if os.path.isdir(src):
            os.symlink(src, dst, target_is_directory=True)
        else:
            os.symlink(src, dst)

    # Patch config.json: set quantization=int4, fix model names
    config_path = os.path.join(work_dir, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)

    cfg["quantization"] = "int4"
    if cfg.get("compr_base_model_name", "").startswith("/"):
        cfg["compr_base_model_name"] = "mistralai/Mistral-7B-Instruct-v0.2"
    if cfg.get("decoder_model_name", "").startswith("/"):
        cfg["decoder_model_name"] = "mistralai/Mistral-7B-Instruct-v0.2"

    # Force stage2 for training
    cfg["training_stage"] = "stage2"
    cfg["load_adapters"] = False
    cfg["pure_inference"] = False

    # Oracle fine-tuning: 1 document per question → top_k must be 1
    cfg["generation_top_k"] = 1

    # Disable memory token embedding optimization — we only train LoRA adapters.
    # This prevents the in-place autograd error in _replace_embeddings where
    # Apple's code modifies a view of the embedding weight in-place.
    cfg["optimize_mem_tokens"] = False

    # Remove symlink and write patched config
    os.remove(config_path)
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)

    return work_dir


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DATASET LOADING (train/val splits for fine-tuning)
# ═══════════════════════════════════════════════════════════════════════════════

def load_finetune_dataset(dataset_name: str, split: str, n_samples: int) -> List[Dict]:
    """
    Load QA dataset for fine-tuning.

    Returns list of {'question': str, 'answer': str, 'document': str}.
    For oracle mode, document = gold context passage.
    """
    from datasets import load_dataset

    if dataset_name == 'squad':
        ds = load_dataset('rajpurkar/squad', split=split)
        ds = ds.filter(lambda x: len(x['answers']['text']) > 0)
        ds = ds.shuffle(seed=42)
        if n_samples:
            ds = ds.select(range(min(n_samples, len(ds))))
        samples = []
        for row in ds:
            samples.append({
                'question': row['question'],
                'answer': row['answers']['text'][0],
                'document': row['context'],
            })

    elif dataset_name == 'triviaqa':
        ds = load_dataset('trivia_qa', 'rc.nocontext', split=split)
        ds = ds.filter(lambda x: len(x['answer']['aliases']) > 0)
        ds = ds.shuffle(seed=42)
        if n_samples:
            ds = ds.select(range(min(n_samples, len(ds))))
        samples = []
        for row in ds:
            samples.append({
                'question': row['question'],
                'answer': row['answer']['value'],
                'document': row['question'],  # nocontext: question as doc
            })

    elif dataset_name == 'hotpotqa':
        ds = load_dataset('hotpot_qa', 'distractor', split=split)
        ds = ds.shuffle(seed=42)
        if n_samples:
            ds = ds.select(range(min(n_samples, len(ds))))
        samples = []
        for row in ds:
            # Use supporting facts as oracle document
            sf_titles = set(row.get('supporting_facts', {}).get('title', []))
            sents = []
            for t, s in zip(row['context']['title'], row['context']['sentences']):
                if t in sf_titles:
                    sents.extend(s)
            doc = ' '.join(sents) if sents else row['question']
            samples.append({
                'question': row['question'],
                'answer': row['answer'],
                'document': doc,
            })

    elif dataset_name == 'nq':
        ds = load_dataset('nq_open', split=split)
        ds = ds.shuffle(seed=42)
        if n_samples:
            ds = ds.select(range(min(n_samples, len(ds))))
        samples = []
        for row in ds:
            a = row['answer'][0] if row['answer'] else ''
            samples.append({
                'question': row['question'],
                'answer': a,
                'document': row['question'],
            })

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    print(f"  ✓ Loaded {len(samples)} {split} samples from {dataset_name}")
    import sys; sys.stdout.flush()
    return samples


# ═══════════════════════════════════════════════════════════════════════════════
# 3. BATCH PREPARATION (uses Apple model's tokenizer and methods)
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_batch(model, sample: Dict, max_dec_len: int = 512) -> Dict[str, torch.Tensor]:
    """
    Prepare a single training batch from a raw sample.

    Uses the Apple model's own tokenizer and methods to create inputs
    matching _forward_stage2_batch expectations:
      - query_input_ids/mask: query with memory tokens (for query_reasoner)
      - enc_input_ids/mask:   document with memory tokens (for encoder)
      - dec_input_ids/mask:   decoder prompt with memory token placeholders
      - labels:               -100 for prompt, token ids for answer
    """
    tokenizer = model.decoder_tokenizer

    # 1. Encode query with memory tokens (for query_reasoner compression)
    q_enc = model._prepare_encoder_inputs(
        [sample['question']], max_length=model.doc_max_length
    )

    # 2. Encode document with memory tokens (for encoder compression)
    doc_enc = model._prepare_encoder_inputs(
        [sample['document']], max_length=model.doc_max_length
    )

    # 3. Build decoder prompt + answer using Apple's chat template
    #    _blend_prompt_and_selected_memory_tokens returns (prompt_len, full_text)
    prompt_len, response_text = model._blend_prompt_and_selected_memory_tokens(
        query=sample['question'], answer=sample['answer']
    )

    # 4. Tokenize the full response (prompt + answer)
    dec_tokens = tokenizer(
        response_text,
        return_tensors='pt',
        padding='max_length',
        max_length=max_dec_len,
        truncation=True,
        add_special_tokens=False,
    )

    # 5. Create labels: -100 for prompt tokens, actual ids for answer
    labels = dec_tokens['input_ids'].clone()
    labels[:, :prompt_len] = -100
    labels[dec_tokens['attention_mask'] == 0] = -100  # Mask padding

    return {
        'query_input_ids': q_enc['input_ids'],
        'query_attention_mask': q_enc['attention_mask'],
        'enc_input_ids': doc_enc['input_ids'],
        'enc_attention_mask': doc_enc['attention_mask'],
        'dec_input_ids': dec_tokens['input_ids'],
        'dec_attention_mask': dec_tokens['attention_mask'],
        'labels': labels,
        'stage': 'stage2',
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TRAINING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def _vram() -> str:
    u = torch.cuda.memory_allocated() / 1e9
    t = torch.cuda.get_device_properties(0).total_memory / 1e9
    return f"{u:.1f}/{t:.1f}GB"


def freeze_encoder_adapter(model):
    """
    Freeze encoder_adapter LoRA params so only query_reasoner + decoder
    adapters are trained (matching paper Stage II).
    """
    frozen_count = 0
    for name, param in model.named_parameters():
        if 'encoder_adapter' in name and param.requires_grad:
            param.requires_grad_(False)
            frozen_count += 1

    # ALWAYS freeze the embedding weight — we never train embeddings during
    # fine-tuning, and Apple's _replace_embeddings does in-place ops on
    # the embedding output which crashes autograd if weight requires grad.
    emb = model.decoder.get_input_embeddings()
    print(f"  [debug] Embedding weight requires_grad BEFORE freeze: {emb.weight.requires_grad}")
    emb.weight.requires_grad_(False)
    # Remove any gradient hooks left from optimize_mem_tokens
    if hasattr(emb.weight, '_backward_hooks'):
        emb.weight._backward_hooks = {}
    frozen_count += 1
    print(f"  [debug] Embedding weight requires_grad AFTER  freeze: {emb.weight.requires_grad}")

    print(f"  ✓ Froze {frozen_count} encoder_adapter/embedding params")


def count_trainable_params(model) -> Tuple[int, int]:
    """Return (trainable, total) parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def validate(model, val_samples: List[Dict], max_dec_len: int,
             max_batches: int = 40) -> float:
    """Run validation and return average loss."""
    model.eval()
    total_loss, n = 0.0, 0

    with torch.no_grad():
        for i, sample in enumerate(val_samples):
            if i >= max_batches:
                break
            try:
                batch = prepare_batch(model, sample, max_dec_len)
                loss, _ = model(batch=batch)
                total_loss += loss.item()
                n += 1
            except Exception as e:
                if i == 0:
                    print(f"  ⚠ Val batch {i} error: {e}")
                continue

    model.train()
    return total_loss / max(n, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. CHECKPOINT SAVING
# ═══════════════════════════════════════════════════════════════════════════════

def save_finetuned_checkpoint(model, ckpt_path: str, output_dir: str,
                               epoch: int, val_loss: float):
    """
    Save fine-tuned checkpoint in Apple format.

    Copies the original checkpoint structure and replaces adapters.pth
    with the fine-tuned adapter weights.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Copy all files from original checkpoint (except adapters.pth)
    for item in os.listdir(ckpt_path):
        src = os.path.join(ckpt_path, item)
        dst = os.path.join(output_dir, item)
        if os.path.exists(dst):
            continue
        if item == 'adapters.pth':
            continue  # Will save new version below
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)

    # Extract adapter state dicts in Apple's format: {adapter_name: state_dict}
    adapter_state = {}
    for adapter_name in model.adapter_keys:
        adapter_sd = {}
        model.decoder.set_adapter(adapter_name)
        for name, param in model.decoder.named_parameters():
            if 'lora_' in name and adapter_name in name:
                # Strip the base model prefix to match loading expectations
                adapter_sd[name] = param.data.cpu().clone()
        if adapter_sd:
            adapter_state[adapter_name] = adapter_sd

    torch.save(adapter_state, os.path.join(output_dir, 'adapters.pth'))

    # Save training metadata
    meta = {
        'epoch': epoch,
        'val_loss': val_loss,
        'timestamp': datetime.now().isoformat(),
    }
    with open(os.path.join(output_dir, 'finetune_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    # Restore all adapters active
    model.decoder.set_adapter(model.adapter_keys)
    print(f"  ✓ Saved fine-tuned checkpoint → {output_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. MAIN TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Parse config from env ─────────────────────────────────────────────────
    ckpt_path     = os.environ.get('CLARA_CKPT_PATH',
                                   '/kaggle/input/datasets/tokiggle/clara-7b-e2e-4q')
    dataset_name  = os.environ.get('CLARA_DATASET', 'squad')
    n_train       = int(os.environ.get('CLARA_N_TRAIN', '2000'))
    n_val         = int(os.environ.get('CLARA_N_VAL', '200'))
    lr            = float(os.environ.get('CLARA_FT_LR', '5e-6'))
    num_epochs    = int(os.environ.get('CLARA_FT_EPOCHS', '1'))
    grad_accum    = int(os.environ.get('CLARA_FT_GRAD_ACC', '8'))
    max_dec_len   = int(os.environ.get('CLARA_FT_MAX_DEC_LEN', '128'))
    output_dir    = os.environ.get('CLARA_OUTPUT_DIR',
                                   f'/kaggle/working/clara-ft-{dataset_name}')
    model_version = os.environ.get('CLARA_MODEL_VERSION',
                                   f'FT_{dataset_name}')

    print("=" * 60)
    print("CLaRa Fine-Tuning — Apple Native Pipeline")
    print("=" * 60)
    print(f"  Checkpoint  : {ckpt_path}")
    print(f"  Dataset     : {dataset_name}")
    print(f"  Train/Val   : {n_train}/{n_val}")
    print(f"  LR          : {lr}")
    print(f"  Epochs      : {num_epochs}")
    print(f"  Grad accum  : {grad_accum}")
    print(f"  Max dec len : {max_dec_len}")
    print(f"  Output      : {output_dir}")
    print("=" * 60)

    # ── Step 1: Assemble workdir ──────────────────────────────────────────────
    print("\n[1/5] Assembling workdir...")
    work_dir = assemble_workdir(ckpt_path)
    print(f"  ✓ Workdir: {work_dir}")

    # ── Step 2: Clear stale HF cache & load model ─────────────────────────────
    print("\n[2/5] Loading Apple CLaRa model (4-bit NF4)...")

    # Clear stale HF module cache
    hf_cache = os.path.expanduser(
        "~/.cache/huggingface/modules/transformers_modules/apple-ft-workdir")
    if os.path.isdir(hf_cache):
        shutil.rmtree(hf_cache)
        print("  ✓ Cleared stale HF module cache")

    model = AutoModel.from_pretrained(
        work_dir, trust_remote_code=True, load_pretrained_checkpoint=True,
    )

    # NOTE: gradient checkpointing is INCOMPATIBLE with Apple's adapter switching.
    # The model calls set_adapter() during forward (encoder→query→decoder), so
    # recomputation during backward produces different tensor shapes → crash.
    # Instead we save VRAM via: mixed precision autocast + reduced seq len.
    model.decoder.config.use_cache = False
    print("  ✓ KV cache disabled for training")

    # ── Monkey-patch _replace_embeddings to fix in-place autograd error ────
    # Apple's original code: inputs_embeds = embedding(ids) then in-place
    # assignment. The embedding output is a view of the weight tensor,
    # so in-place ops crash autograd. Fix: .clone() before modifying.
    import types

    def _replace_embeddings_safe(self, compressed_embs, dec_input_ids, indices):
        inputs_embeds = self.decoder.get_input_embeddings()(dec_input_ids).clone()
        num_embs = compressed_embs.size(1)
        slot_len = num_embs + (1 if self.sep else 0)
        first_mem_token_indices = torch.argmax(
            (dec_input_ids == self.decoder_tokenizer.mem_token_ids[0]).int(), dim=1
        )
        batch_size = inputs_embeds.size(0)
        for i in range(batch_size):
            for j in range(indices[i], indices[i + 1]):
                start_idx = first_mem_token_indices[i].item() + (j - indices[i]) * slot_len
                inputs_embeds[i, start_idx:start_idx + num_embs, :] = compressed_embs[j]
        return inputs_embeds

    model._replace_embeddings = types.MethodType(_replace_embeddings_safe, model)
    print("  ✓ Patched _replace_embeddings (.clone() for autograd safety)")

    vram = torch.cuda.memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  ✓ Model loaded. VRAM: {vram:.1f}/{total:.1f} GB")

    # ── Step 3: Freeze encoder, set up optimizer ──────────────────────────────
    print("\n[3/5] Configuring training...")

    # Freeze encoder adapter (only train query_reasoner + decoder)
    freeze_encoder_adapter(model)

    trainable, total_params = count_trainable_params(model)
    print(f"  Trainable params: {trainable:,} / {total_params:,} "
          f"({100*trainable/total_params:.2f}%)")

    # Optimizer: only trainable params
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01,
    )

    # ── Step 4: Load data ─────────────────────────────────────────────────────
    print(f"\n[4/5] Loading '{dataset_name}' dataset...")
    train_samples = load_finetune_dataset(dataset_name, 'train', n_train)
    val_samples   = load_finetune_dataset(dataset_name, 'validation', n_val)

    steps_per_epoch = math.ceil(len(train_samples) / grad_accum)
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = int(total_steps * 0.03)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )

    print(f"  Steps/epoch: {steps_per_epoch}  "
          f"Total: {total_steps}  Warmup: {warmup_steps}")

    # ── Step 5: Training loop ─────────────────────────────────────────────────
    import sys, time as _time
    print(f"\n[5/5] Training for {num_epochs} epoch(s)...\n")
    sys.stdout.flush()

    best_val = float('inf')
    global_step = 0
    model.train()
    optimizer.zero_grad()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_samples = 0
        errors = 0

        # Shuffle training data each epoch
        random.shuffle(train_samples)

        for step, sample in enumerate(train_samples):
            try:
                # Clear cache before each step to prevent fragmentation OOM
                torch.cuda.empty_cache()

                batch = prepare_batch(model, sample, max_dec_len)
                if step == 0:
                    print(f"  [timing] Batch prepared, starting forward..."); sys.stdout.flush()
                    _t0 = _time.time()

                # No autocast — model is already 4-bit quantized with
                # float32 LoRA. Adding autocast creates redundant casts.
                loss, info = model(batch=batch)

                if step == 0:
                    print(f"  [timing] Forward done in {_time.time()-_t0:.1f}s, backward..."); sys.stdout.flush()
                    _t1 = _time.time()

                loss = loss / grad_accum
                loss.backward()

                if step == 0:
                    print(f"  [timing] Backward done in {_time.time()-_t1:.1f}s | {_vram()}"); sys.stdout.flush()

                epoch_loss += loss.item() * grad_accum
                epoch_samples += 1
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    # Free the failed computation graph
                    torch.cuda.empty_cache()
                    gc.collect()
                errors += 1
                if errors <= 3:
                    import traceback
                    print(f"  ⚠ Step {step} error: {e}")
                    if errors == 1:
                        traceback.print_exc()
                if errors >= 10:
                    print(f"  ❌ Too many errors ({errors}), aborting epoch")
                    break
                continue

            # Gradient accumulation step
            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    1.0,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 5 == 0 or global_step == 1:
                    avg_loss = epoch_loss / max(epoch_samples, 1)
                    print(f"  Ep{epoch+1} step{global_step:4d}/{total_steps} | "
                          f"loss {avg_loss:.4f} | "
                          f"lr {scheduler.get_last_lr()[0]:.1e} | {_vram()}")
                    sys.stdout.flush()

            # VRAM management
            if (step + 1) % grad_accum == 0:
                gc.collect()

        # Flush remaining gradients
        if (step + 1) % grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

        # ── Validation ────────────────────────────────────────────────────
        val_loss = validate(model, val_samples, max_dec_len)
        train_loss = epoch_loss / max(epoch_samples, 1)
        print(f"\n  Epoch {epoch+1}/{num_epochs}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  "
              f"errors={errors}  {_vram()}")

        # Save best checkpoint
        if val_loss < best_val:
            best_val = val_loss
            save_finetuned_checkpoint(
                model, ckpt_path, output_dir, epoch + 1, val_loss)

        model.train()

    # ── Cleanup ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Best val loss: {best_val:.4f}")
    print(f"  Checkpoint   : {output_dir}")
    print(f"{'='*60}")

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
