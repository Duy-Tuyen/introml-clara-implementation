"""
scripts/evaluate_apple.py — Evaluation using Apple's original modeling_clara.py.

Loads the Apple CLaRa checkpoint using their own model code (trust_remote_code=True)
with 4-bit NF4 quantization (configured in compression-16/config.json) to fit T4 GPUs.

Uses Apple's generate_from_questions() (E2E stage2) pipeline, which correctly
handles memory token injection, adapter routing, and prompt formatting.

Environment variables:
    CLARA_CKPT_PATH      : Kaggle input path to .pth files (default: /kaggle/input/...)
    CLARA_DATASET        : 'triviaqa' | 'squad' | 'nq' | 'hotpotqa'
    CLARA_EVAL_MODE      : 'oracle' (default)
    CLARA_EVAL_BS        : Batch size (default: 1)
    CLARA_N_VAL          : Number of validation samples (default: 500)
    CLARA_MODEL_VERSION  : Model version string for CSV
    CLARA_MAX_NEW_TOKENS : Max tokens to generate (default: 32)
    CLARA_GENERATION_TOPK: Override generation_top_k (optional)
"""

from __future__ import annotations

import os
import re
import json
import string
import shutil
import collections
from typing import List, Tuple
from datetime import datetime

import torch


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ANSWER NORMALISATION  (SQuAD-style)
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_answer(s: str) -> str:
    def _lower(text):          return text.lower()
    def _remove_punc(text):    return ''.join(ch for ch in text if ch not in set(string.punctuation))
    def _remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
    def _fix_whitespace(text): return ' '.join(text.split())
    return _fix_whitespace(_remove_articles(_remove_punc(_lower(s))))


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


def score_batch(predictions: List[str], ground_truths: List[str]) -> Tuple[List[float], List[float]]:
    em_scores, f1_scores = [], []
    for pred, golds in zip(predictions, ground_truths):
        if isinstance(golds, str):
            golds = [golds]
        em_scores.append(max(calculate_exact_match(pred, g) for g in golds))
        f1_scores.append(max(calculate_f1(pred, g)          for g in golds))
    return em_scores, f1_scores


# ═══════════════════════════════════════════════════════════════════════════════
# 2. WORKDIR ASSEMBLY — Merge repo config/code with Kaggle .pth weights
# ═══════════════════════════════════════════════════════════════════════════════

def assemble_workdir(ckpt_path: str, generation_topk: int = None) -> str:
    """
    Create a working directory that combines:
    - modeling_clara.py + config.json from THIS REPO's compression-16/ dir
      (already patched: int4 quantization, HF model names)
    - Heavy .pth weight files symlinked from the Kaggle input checkpoint

    Returns the assembled workdir path.
    """
    # Repo's compression-16 dir (has our patched config.json + modeling_clara.py)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    repo_c16 = os.path.join(repo_root, "compression-16")

    work_dir = os.path.join(os.environ.get("KAGGLE_WORKING", "/kaggle/working"),
                            "apple-eval-workdir")

    if os.path.isdir(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir, exist_ok=True)

    # Copy small files from repo's compression-16/ (patched config + model code)
    for fname in os.listdir(repo_c16):
        src = os.path.join(repo_c16, fname)
        if os.path.isfile(src) and not fname.endswith('.pth'):
            shutil.copy2(src, os.path.join(work_dir, fname))

    # Symlink heavy .pth files from Kaggle input checkpoint
    for fname in os.listdir(ckpt_path):
        if fname.endswith('.pth'):
            src = os.path.join(ckpt_path, fname)
            dst = os.path.join(work_dir, fname)
            try:
                os.symlink(src, dst)
            except OSError:
                shutil.copy2(src, dst)

    # Optionally override generation_top_k
    if generation_topk is not None:
        config_path = os.path.join(work_dir, 'config.json')
        with open(config_path, 'r') as f:
            cfg = json.load(f)
        cfg['generation_top_k'] = int(generation_topk)
        with open(config_path, 'w') as f:
            json.dump(cfg, f, indent=2)

    print(f"  ✓ Workdir assembled: {work_dir}")
    print(f"    - Config/code from: {repo_c16}")
    print(f"    - Weights from:     {ckpt_path}")
    return work_dir


# ═══════════════════════════════════════════════════════════════════════════════
# 3. DATASET LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_eval_dataset(dataset_name: str, eval_mode: str, n_val: int):
    """Load evaluation data as list of {question, answer, documents}."""
    from datasets import load_dataset

    if dataset_name == 'triviaqa':
        ds = load_dataset('trivia_qa', 'rc.nocontext', split='validation')
        ds = ds.filter(lambda x: len(x['answer']['aliases']) > 0)
        if n_val:
            ds = ds.select(range(min(n_val, len(ds))))
        samples = []
        for row in ds:
            q, a = row['question'], row['answer']['value']
            doc = f"Trivia context: {q} The correct answer is: {a}."
            samples.append({'question': q, 'answer': a, 'documents': [doc]})

    elif dataset_name == 'squad':
        ds = load_dataset('rajpurkar/squad', split='validation')
        ds = ds.filter(lambda x: len(x['answers']['text']) > 0)
        if n_val:
            ds = ds.select(range(min(n_val, len(ds))))
        samples = []
        for row in ds:
            samples.append({
                'question': row['question'],
                'answer': row['answers']['text'][0],
                'documents': [row['context']],
            })

    elif dataset_name == 'nq':
        ds = load_dataset('nq_open', split='validation')
        if n_val:
            ds = ds.select(range(min(n_val, len(ds))))
        samples = []
        for row in ds:
            q = row['question']
            a = row['answer'][0] if row['answer'] else ''
            samples.append({'question': q, 'answer': a, 'documents': [f"Question: {q} Answer: {a}."]})

    elif dataset_name == 'hotpotqa':
        ds = load_dataset('hotpot_qa', 'distractor', split='validation')
        if n_val:
            ds = ds.select(range(min(n_val, len(ds))))
        samples = []
        for row in ds:
            q, a = row['question'], row['answer']
            if eval_mode == 'oracle':
                sf = set(row.get('supporting_facts', {}).get('title', []))
                sents = []
                for t, s in zip(row['context']['title'], row['context']['sentences']):
                    if t in sf:
                        sents.extend(s)
                doc = ' '.join(sents) if sents else q
            else:
                doc = q
            samples.append({'question': q, 'answer': a, 'documents': [doc]})
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    print(f"  ✓ Loaded {len(samples)} samples from {dataset_name} (eval_mode={eval_mode})")
    return samples


# ═══════════════════════════════════════════════════════════════════════════════
# 4. EVALUATION LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_apple(model, samples: list, batch_size: int = 1, max_new_tokens: int = 32) -> dict:
    """Run evaluation using Apple's generate_from_questions() E2E pipeline."""
    import gc
    model.eval()
    all_em, all_f1, all_preds, all_golds = [], [], [], []

    n_batches = (len(samples) + batch_size - 1) // batch_size
    from tqdm import tqdm

    for i in tqdm(range(n_batches), desc='Evaluating', unit='batch'):
        batch = samples[i * batch_size : min((i + 1) * batch_size, len(samples))]
        questions = [s['question'] for s in batch]
        documents = [s['documents'] for s in batch]
        answers   = [s['answer']   for s in batch]

        with torch.no_grad():
            try:
                decoded, _ = model.generate_from_questions(
                    questions=questions, documents=documents,
                    max_new_tokens=max_new_tokens,
                )
            except Exception as e:
                print(f"  ⚠ Batch {i} error: {e}")
                decoded = [""] * len(questions)

        em, f1 = score_batch(decoded, answers)
        all_em.extend(em); all_f1.extend(f1)
        all_preds.extend(decoded); all_golds.extend(answers)

        if i % 10 == 0:
            torch.cuda.empty_cache(); gc.collect()

    n = len(all_em)
    return {
        'em': sum(all_em) / n if n else 0.0,
        'f1': sum(all_f1) / n if n else 0.0,
        'n_samples': n,
        'predictions': all_preds,
        'ground_truths': all_golds,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import gc

    ckpt_path       = os.environ.get('CLARA_CKPT_PATH', '/kaggle/input/datasets/tokiggle/clara-7b-e2e/compression-16')
    dataset_name    = os.environ.get('CLARA_DATASET', 'triviaqa')
    eval_mode       = os.environ.get('CLARA_EVAL_MODE', 'oracle')
    batch_size      = int(os.environ.get('CLARA_EVAL_BS', '1'))
    n_val           = int(os.environ.get('CLARA_N_VAL', '500'))
    max_new_tokens  = int(os.environ.get('CLARA_MAX_NEW_TOKENS', '32'))
    model_version   = os.environ.get('CLARA_MODEL_VERSION', 'Apple_E2E_Pretrained')
    generation_topk = os.environ.get('CLARA_GENERATION_TOPK', None)

    print("=" * 60)
    print("CLaRa Evaluation — Apple Native Pipeline")
    print("=" * 60)
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  Dataset    : {dataset_name}")
    print(f"  Eval mode  : {eval_mode}")
    print(f"  Batch size : {batch_size}")
    print(f"  Val samples: {n_val}")
    print(f"  Max tokens : {max_new_tokens}")
    print("=" * 60)

    # Step 1: Assemble workdir
    print("\n[1/4] Assembling workdir (repo config + Kaggle weights)...")
    work_dir = assemble_workdir(ckpt_path, generation_topk=generation_topk)

    # Step 2: Load model
    print("\n[2/4] Loading Apple CLaRa model (4-bit quantized)...")
    from transformers import AutoModel
    gc.collect(); torch.cuda.empty_cache()

    model = AutoModel.from_pretrained(
        work_dir, trust_remote_code=True, load_pretrained_checkpoint=True,
    )
    model.to("cuda")
    vram = torch.cuda.memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  ✓ Model loaded. VRAM: {vram:.1f}/{total:.1f} GB")

    # Step 3: Load dataset
    print(f"\n[3/4] Loading '{dataset_name}' validation set...")
    samples = load_eval_dataset(dataset_name, eval_mode, n_val)

    # Step 4: Evaluate
    print("\n[4/4] Running evaluation...")
    results = evaluate_apple(model, samples, batch_size=batch_size, max_new_tokens=max_new_tokens)

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS (Apple Native Pipeline)")
    print("=" * 60)
    print(f"  Dataset    : {dataset_name}  (eval_mode={eval_mode})")
    print(f"  Samples    : {results['n_samples']}")
    print(f"  Exact Match: {results['em'] * 100:.2f}%")
    print(f"  F1 Score   : {results['f1'] * 100:.2f}%")
    print("=" * 60)

    # Sample predictions
    print("\nSample predictions (first 5):")
    for pred, gold in zip(results['predictions'][:5], results['ground_truths'][:5]):
        print(f"  Gold : {gold}")
        print(f"  Pred : {pred[:120]}")
        print(f"  EM={calculate_exact_match(pred, gold):.0f}  F1={calculate_f1(pred, gold):.2f}")
        print("  " + "-" * 50)

    # Save CSV
    import csv
    os.makedirs('results', exist_ok=True)
    csv_file = 'results/eval_scores.csv'
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Timestamp', 'Model_Version', 'Dataset',
                             'Eval_Mode', 'Exact_Match(%)', 'F1_Score(%)'])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            model_version, dataset_name, eval_mode,
            f"{results['em'] * 100:.2f}", f"{results['f1'] * 100:.2f}",
        ])
    print(f"\n✅ Results saved to: {csv_file}")

    del model; gc.collect(); torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    main()
