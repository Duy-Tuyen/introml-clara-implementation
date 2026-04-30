"""
scripts/evaluate.py — Evaluation Pipeline for CLaRa.

Calculates Exact Match (EM) and F1 Score on any supported dataset
using the SQuAD-style normalisation pipeline, exactly as done in the
original CLaRa paper and standard QA benchmarks.

Usage:
    # From repo root:
    python -m scripts.evaluate

    # Override dataset or checkpoint via env vars:
    CLARA_DATASET=triviaqa python -m scripts.evaluate
    CLARA_DATASET=squad CLARA_CKPT_DIR=/kaggle/input/... python -m scripts.evaluate
"""

from __future__ import annotations

import os
import re
import string
import collections
from typing import List, Tuple

import torch
from peft import set_peft_model_state_dict
from tqdm import tqdm

from configs.config import CLaRaConfig
from models.clara_model import build_clara_model
from models.utils import load_peft_weights_local, print_vram_usage
from data.dataset import get_retrieval_eval_loader


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ANSWER NORMALISATION  (SQuAD-style, identical to the reference scorer)
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_answer(s: str) -> str:
    def _lower(text):          return text.lower()
    def _remove_punc(text):    return ''.join(ch for ch in text if ch not in set(string.punctuation))
    def _remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
    def _fix_whitespace(text): return ' '.join(text.split())
    return _fix_whitespace(_remove_articles(_remove_punc(_lower(s))))


# ═══════════════════════════════════════════════════════════════════════════════
# 2. METRIC FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def calculate_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = sum((collections.Counter(pred_tokens) & collections.Counter(gold_tokens)).values())
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall    = common / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def score_batch(
    predictions: List[str],
    ground_truths: List[str],
) -> Tuple[List[float], List[float]]:
    em_scores, f1_scores = [], []
    for pred, golds in zip(predictions, ground_truths):
        if isinstance(golds, str):
            golds = [golds]
        em_scores.append(max(calculate_exact_match(pred, g) for g in golds))
        f1_scores.append(max(calculate_f1(pred, g)          for g in golds))
    return em_scores, f1_scores


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CHECKPOINT LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_checkpoint(model, ckpt_dir: str) -> None:
    query_dir = os.path.join(ckpt_dir, 'adapters', 'query')
    gen_dir = os.path.join(ckpt_dir, 'adapters', 'generator')
    extra_path = os.path.join(ckpt_dir, 'clara_stage2_extra.pth')

    if not os.path.isdir(query_dir) and not os.path.isdir(gen_dir):
        raise FileNotFoundError(
            f"No Stage II checkpoint found at '{ckpt_dir}'.\n"
            f"Expected: {query_dir} and {gen_dir}"
        )

    if os.path.isdir(query_dir):
        q_weights = load_peft_weights_local(query_dir)
        set_peft_model_state_dict(model.backbone, q_weights, adapter_name='query')
        print(f"  ✓ Query adapter ← {query_dir}")

    if os.path.isdir(gen_dir):
        g_weights = load_peft_weights_local(gen_dir)
        set_peft_model_state_dict(model.backbone, g_weights, adapter_name='generator')
        print(f"  ✓ Generator adapter ← {gen_dir}")

    if os.path.exists(extra_path):
        saved = torch.load(extra_path, map_location='cuda')
        model.mem_token_embed.data = saved['mem_token_embed']
        print(f"  ✓ Memory tokens ← {extra_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. EVALUATION LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate(model, val_loader, cfg, max_new_tokens: int = 32) -> dict:
    model.eval()
    all_em, all_f1 = [], []
    all_preds, all_golds = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Evaluating', unit='batch'):
            # Pull ground truth strings trước khi filter tensor
            ground_truths: List[str] = batch.pop('answers')

            # FIX: filter tensor trước khi đưa lên GPU
            tensor_batch = {k: v for k, v in batch.items()
                            if isinstance(v, torch.Tensor)}
            tensor_batch = {k: v.cuda() for k, v in tensor_batch.items()}

            predictions: List[str] = model.generate_answer_e2e(
                tensor_batch['candidate_doc_input_ids'],
                tensor_batch['candidate_doc_attention_mask'],
                tensor_batch['candidate_mask'],
                tensor_batch['question_input_ids'],
                tensor_batch['question_attention_mask'],
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
    cfg = CLaRaConfig(
        dataset_name    = os.environ.get('CLARA_DATASET',  'triviaqa'),
        eval_mode       = os.environ.get('CLARA_EVAL_MODE','oracle'),
        eval_batch_size = int(os.environ.get('CLARA_EVAL_BS', 4)),
        n_val           = int(os.environ.get('CLARA_N_VAL',  500)),
    )

    print("=" * 60)
    print("CLaRa Evaluation Pipeline")
    print("=" * 60)
    print(f"  Dataset    : {cfg.dataset_name}")
    print(f"  Eval mode  : {cfg.eval_mode}")
    print(f"  Batch size : {cfg.eval_batch_size}")
    print(f"  Val samples: {cfg.n_val}")

    # FIX: ưu tiên CLARA_CKPT_DIR (pretrained Apple weights) trước
    # sau đó mới fallback về CLARA_CKPT_EPOCH (checkpoint tự train)
    ckpt_dir = (
        os.environ.get('CLARA_STAGE2_DIR')
        or os.environ.get('CLARA_CKPT_DIR')
        or cfg.pretrained_ckpt_dir
    )
    print(f"  Checkpoint : {ckpt_dir}")
    print("=" * 60)

    # Build model
    print("\n[1/3] Building model...")
    model, tokenizer = build_clara_model(cfg)
    print(print_vram_usage())

    # Load checkpoint
    print(f"\n[2/3] Loading checkpoint from '{ckpt_dir}'...")
    load_checkpoint(model, ckpt_dir)
    print(print_vram_usage())

    # Build eval DataLoader
    print(f"\n[3/3] Loading '{cfg.dataset_name}' validation set...")
    val_loader = get_retrieval_eval_loader(tokenizer, cfg, split='validation')

    # Run evaluation
    print("\nRunning evaluation...")
    results = evaluate(model, val_loader, cfg)

    # In kết quả
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Dataset    : {cfg.dataset_name}  (eval_mode={cfg.eval_mode})")
    print(f"  Samples    : {results['n_samples']}")
    print(f"  Exact Match: {results['em'] * 100:.2f}%")
    print(f"  F1 Score   : {results['f1'] * 100:.2f}%")
    print("=" * 60)

    # In 5 ví dụ đầu
    print("\nSample predictions (first 5):")
    for pred, gold in zip(results['predictions'][:5], results['ground_truths'][:5]):
        print(f"  Gold : {gold}")
        print(f"  Pred : {pred}")
        print(f"  EM={calculate_exact_match(pred, gold):.0f}  F1={calculate_f1(pred, gold):.2f}")
        print("  " + "-" * 50)

    # Lưu CSV
    import csv
    from datetime import datetime

    os.makedirs('results', exist_ok=True)
    csv_file = 'results/eval_scores.csv'
    file_exists = os.path.isfile(csv_file)

    # Đổi model_version tùy theo lần chạy
    # Ví dụ: "Apple_Pretrained", "Finetuned_HotpotQA_ep1", v.v.
    model_version = os.environ.get('CLARA_MODEL_VERSION', 'Apple_Pretrained')

    with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Timestamp', 'Model_Version', 'Dataset',
                             'Eval_Mode', 'Exact_Match(%)', 'F1_Score(%)'])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            model_version,
            cfg.dataset_name,
            cfg.eval_mode,
            f"{results['em'] * 100:.2f}",
            f"{results['f1'] * 100:.2f}",
        ])

    print(f"\n✅ Kết quả đã lưu vào: {csv_file}")
    return results


if __name__ == '__main__':
    main()