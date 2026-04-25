"""
scripts/evaluate.py — Evaluation Pipeline for CLaRa.

Calculates Exact Match (EM) and F1 Score on any supported dataset
using the SQuAD-style normalisation pipeline, exactly as done in the
original CLaRa paper and standard QA benchmarks.

Usage:
    # From repo root:
    python -m scripts.evaluate

    # Override dataset or checkpoint via env vars (or edit cfg below):
    CLARA_DATASET=2wikimultihop CLARA_CKPT_EPOCH=2 python -m scripts.evaluate
"""

from __future__ import annotations

import os
import re
import string
import collections
from typing import List, Tuple

import torch
from peft import load_peft_weights, set_peft_model_state_dict
from tqdm import tqdm

from configs.config import CLaRaConfig
from models.clara_model import build_clara_model
from models.utils import print_vram_usage
from data.dataset import get_eval_loader


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ANSWER NORMALISATION  (SQuAD-style, identical to the reference scorer)
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_answer(s: str) -> str:
    """
    Canonical answer normalisation used by SQuAD / most QA leaderboards.

    Steps (order matters):
        1. Lowercase
        2. Remove punctuation
        3. Remove articles (a, an, the)
        4. Collapse whitespace
    """
    def _lower(text: str) -> str:
        return text.lower()

    def _remove_punctuation(text: str) -> str:
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def _remove_articles(text: str) -> str:
        # Only strip leading/trailing/standalone articles, not mid-word
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def _fix_whitespace(text: str) -> str:
        return ' '.join(text.split())

    return _fix_whitespace(_remove_articles(_remove_punctuation(_lower(s))))


# ═══════════════════════════════════════════════════════════════════════════════
# 2. METRIC FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_exact_match(prediction: str, ground_truth: str) -> float:
    """
    Returns 1.0 if the normalised prediction equals the normalised ground truth,
    else 0.0.
    """
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def calculate_f1(prediction: str, ground_truth: str) -> float:
    """
    Token-level F1 between prediction and ground_truth after normalisation.
    Identical to the SQuAD official evaluation script.

    F1 = 2 * precision * recall / (precision + recall)
    where precision = common_tokens / prediction_tokens
          recall    = common_tokens / ground_truth_tokens
    """
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    # Edge case: both empty → perfect match
    if not pred_tokens and not gold_tokens:
        return 1.0
    # One of them is empty
    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counter = collections.Counter(pred_tokens)
    gold_counter = collections.Counter(gold_tokens)

    # Intersection: sum of min counts for each token
    common = sum((pred_counter & gold_counter).values())

    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall    = common / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def score_batch(
    predictions: List[str],
    ground_truths: List[str],
) -> Tuple[List[float], List[float]]:
    """
    Score a batch and return per-example (em, f1) lists.
    Ground truth can be a single string or a list of acceptable answers —
    we take the max score across all acceptable answers (SQuAD convention).
    """
    em_scores, f1_scores = [], []
    for pred, golds in zip(predictions, ground_truths):
        # golds may be a single string or a list of acceptable answers
        if isinstance(golds, str):
            golds = [golds]
        em = max(calculate_exact_match(pred, g) for g in golds)
        f1 = max(calculate_f1(pred, g)          for g in golds)
        em_scores.append(em)
        f1_scores.append(f1)
    return em_scores, f1_scores


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CHECKPOINT LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_checkpoint(model, ckpt_dir: str) -> None:
    """
    Load projector weights, mem_bias, and LoRA adapters from ckpt_dir.

    Expected directory layout:
        ckpt_dir/
        ├── clara_extra.pth   # projector state_dict + mem_bias
        └── lora/             # HuggingFace PEFT LoRA adapter
    """
    extra_path = os.path.join(ckpt_dir, 'clara_extra.pth')
    lora_dir   = os.path.join(ckpt_dir, 'lora')

    if not os.path.exists(extra_path) and not os.path.exists(lora_dir):
        raise FileNotFoundError(
            f"No checkpoint found at '{ckpt_dir}'. "
            "Make sure you have trained the model or downloaded the pretrained weights."
        )

    if os.path.exists(extra_path):
        saved = torch.load(extra_path, map_location='cuda')
        model.proj.load_state_dict(saved['proj'])
        model.mem_bias.data = saved['mem_bias']
        epoch = saved.get('epoch', '?')
        val   = saved.get('val_loss', '?')
        print(f"  ✓ Projector & MemBias  ← {extra_path}  "
              f"(epoch={epoch}, val_loss={val})")
    else:
        print(f"  ⚠ clara_extra.pth not found — skipping projector weights.")

    if os.path.exists(lora_dir):
        lora_weights = load_peft_weights(lora_dir)
        set_peft_model_state_dict(model.backbone, lora_weights)
        print(f"  ✓ LoRA adapter         ← {lora_dir}")
    else:
        print(f"  ⚠ lora/ directory not found — using base LoRA weights.")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. EVALUATION LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate(model, val_loader, cfg, max_new_tokens: int = 32) -> dict:
    """
    Run inference over val_loader and return aggregate EM and F1.

    Args:
        model           : CLaRaModel (already loaded with checkpoint)
        val_loader      : DataLoader using CLaRaDataset.collate_eval()
        cfg             : CLaRaConfig
        max_new_tokens  : max tokens to generate per answer

    Returns:
        {'em': float, 'f1': float, 'n_samples': int,
         'predictions': list[str], 'ground_truths': list[str]}
    """
    model.eval()
    all_em, all_f1 = [], []
    all_preds, all_golds = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Evaluating', unit='batch'):
            # Pull ground truth strings out BEFORE moving to CUDA
            ground_truths: List[str] = batch.pop('answers')

            batch = {k: v.cuda() for k, v in batch.items()}

            predictions: List[str] = model.generate_answer(
                batch['doc_input_ids'],
                batch['doc_attention_mask'],
                batch['question_input_ids'],
                batch['question_attention_mask'],
                max_new_tokens=max_new_tokens,
            )

            em_scores, f1_scores = score_batch(predictions, ground_truths)
            all_em.extend(em_scores)
            all_f1.extend(f1_scores)
            all_preds.extend(predictions)
            all_golds.extend(ground_truths)

    n = len(all_em)
    return {
        'em':            sum(all_em) / n if n else 0.0,
        'f1':            sum(all_f1) / n if n else 0.0,
        'n_samples':     n,
        'predictions':   all_preds,
        'ground_truths': all_golds,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Config ────────────────────────────────────────────────────────────────
    cfg = CLaRaConfig(
        dataset_name=os.environ.get('CLARA_DATASET', 'triviaqa'),
        eval_mode=os.environ.get('CLARA_EVAL_MODE', 'oracle'),
        eval_batch_size=int(os.environ.get('CLARA_EVAL_BS', 4)),
        n_val=int(os.environ.get('CLARA_N_VAL', 500)),
    )

    print("=" * 60)
    print("CLaRa Evaluation Pipeline")
    print("=" * 60)
    print(f"  Dataset    : {cfg.dataset_name}")
    print(f"  Eval mode  : {cfg.eval_mode}")
    print(f"  Batch size : {cfg.eval_batch_size}")
    print(f"  Val samples: {cfg.n_val}")
    print(f"  Checkpoint : {cfg.pretrained_ckpt_dir}")
    print("=" * 60)

    # ── Build model & tokenizer ────────────────────────────────────────────────
    print("\n[1/3] Building model...")
    model, tokenizer = build_clara_model(cfg)
    print(print_vram_usage())

    # ── Load checkpoint ────────────────────────────────────────────────────────
    print(f"\n[2/3] Loading checkpoint from '{cfg.pretrained_ckpt_dir}'...")

    # Allow overriding with a specific best_epN directory
    ckpt_epoch = os.environ.get('CLARA_CKPT_EPOCH')
    if ckpt_epoch is not None:
        ckpt_dir = os.path.join(cfg.output_dir, f'best_ep{ckpt_epoch}')
    else:
        ckpt_dir = cfg.pretrained_ckpt_dir

    load_checkpoint(model, ckpt_dir)
    print(print_vram_usage())

    # ── Build eval DataLoader ──────────────────────────────────────────────────
    print(f"\n[3/3] Loading '{cfg.dataset_name}' validation set...")
    val_loader = get_eval_loader(tokenizer, cfg, split='validation')

    # ── Run evaluation ─────────────────────────────────────────────────────────
    print("\nRunning evaluation...")
    results = evaluate(model, val_loader, cfg)

    # ── Print results ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Dataset    : {cfg.dataset_name}  (eval_mode={cfg.eval_mode})")
    print(f"  Samples    : {results['n_samples']}")
    print(f"  Exact Match: {results['em'] * 100:.2f}%")
    print(f"  F1 Score   : {results['f1'] * 100:.2f}%")
    print("=" * 60)

    # ── Show a few prediction examples ────────────────────────────────────────
    print("\nSample predictions (first 5):")
    for pred, gold in zip(results['predictions'][:5], results['ground_truths'][:5]):
        em = calculate_exact_match(pred, gold)
        f1 = calculate_f1(pred, gold)
        print(f"  Gold : {gold}")
        print(f"  Pred : {pred}")
        print(f"  EM={em:.0f}  F1={f1:.2f}")
        print("  " + "-" * 50)

    # ── Show a few prediction examples ────────────────────────────────────────
    print("\nSample predictions (first 5):")
    for pred, gold in zip(results['predictions'][:5], results['ground_truths'][:5]):
        em = calculate_exact_match(pred, gold)
        f1 = calculate_f1(pred, gold)
        print(f"  Gold : {gold}")
        print(f"  Pred : {pred}")
        print(f"  EM={em:.0f}  F1={f1:.2f}")
        print("  " + "-" * 50)

    import csv
    from datetime import datetime
    
    # 1. Tính điểm % trung bình của toàn bộ tập test
    total_samples = len(results['predictions'])
    total_em = sum([calculate_exact_match(p, g) for p, g in zip(results['predictions'], results['ground_truths'])])
    total_f1 = sum([calculate_f1(p, g) for p, g in zip(results['predictions'], results['ground_truths'])])
    
    final_em = (total_em / total_samples) * 100 if total_samples > 0 else 0
    final_f1 = (total_f1 / total_samples) * 100 if total_samples > 0 else 0
    
    print("\n" + "="*40)
    print("🏆 OVERALL EVALUATION RESULTS 🏆")
    print("="*40)
    print(f"Total Samples Tested : {total_samples}")
    print(f"Overall Exact Match  : {final_em:.2f}%")
    print(f"Overall F1 Score     : {final_f1:.2f}%")
    print("="*40)

    # 2. Lưu vào file CSV (Bảng điểm)
    os.makedirs('results', exist_ok=True)
    csv_file = 'results/eval_scores.csv'
    file_exists = os.path.isfile(csv_file)
    
    # BẠN NHỚ SỬA TÊN NÀY TRƯỚC MỖI LẦN CHẠY TEST NHÉ!
    # Ví dụ: "Apple_Pretrained", "My_Model_1_Epoch_4bit", v.v.
    model_version = "Apple_Pretrained" 
    
    # Lấy thông tin từ config (Giả sử trên đầu hàm main Claude có khởi tạo `cfg = CLaRaConfig()`)
    # Nếu Claude không truyền `cfg` xuống đây, bạn có thể thay bằng chuỗi "2wikimultihop"
    dataset_name = cfg.dataset_name if 'cfg' in locals() else "unknown_dataset"
    eval_mode = cfg.eval_mode if 'cfg' in locals() else "oracle"

    with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            # Ghi header nếu file mới toanh
            writer.writerow(['Timestamp', 'Model_Version', 'Dataset', 'Eval_Mode', 'Exact_Match(%)', 'F1_Score(%)'])
        
        # Ghi data
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
            model_version, 
            dataset_name, 
            eval_mode, 
            f"{final_em:.2f}", 
            f"{final_f1:.2f}"
        ])
        
    print(f"✅ Báo cáo đã được lưu vào: {csv_file}\n")

    return results


if __name__ == '__main__':
    main()