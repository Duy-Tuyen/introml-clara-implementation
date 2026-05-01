"""
scripts/evaluate_apple.py — Evaluation using Apple's original modeling_clara.py.

Loads the Apple CLaRa checkpoint using their own model code (trust_remote_code=True)
with 4-bit NF4 quantization to fit T4 GPUs.

Uses Apple's generate_from_questions() (E2E stage2) pipeline, which correctly
handles memory token injection, adapter routing, and prompt formatting.

Environment variables (set by the notebook, same pattern as scripts/evaluate.py):
    CLARA_CKPT_PATH      : Path to Apple compression-16 dir on Kaggle
                           (default: /kaggle/input/datasets/tokiggle/clara-7b-e2e/compression-16)
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
# 2. WORKDIR ASSEMBLY
#    The Apple checkpoint on Kaggle (CLARA_CKPT_PATH) contains everything:
#      modeling_clara.py, config.json, adapters.pth, decoder_first_last_layers.pth, tokenizer files
#    We copy it to a writable dir so we can patch config.json for int4 quantization
#    and fix the hardcoded Apple-internal model paths.
# ═══════════════════════════════════════════════════════════════════════════════

def assemble_workdir(ckpt_path: str, generation_topk: int = None) -> str:
    """
    Create a writable working copy of the Apple checkpoint directory with:
      - config.json patched: quantization=int4, HF model names
      - All other files symlinked (or copied on Windows) from ckpt_path

    Args:
        ckpt_path: Path to the read-only Kaggle input checkpoint dir
        generation_topk: Optional override for generation_top_k

    Returns:
        Path to the writable workdir
    """
    work_dir = "/kaggle/working/apple-eval-workdir"

    if os.path.isdir(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir, exist_ok=True)

    # Link or copy all files from the checkpoint
    for fname in os.listdir(ckpt_path):
        src = os.path.join(ckpt_path, fname)
        dst = os.path.join(work_dir, fname)
        if not os.path.isfile(src):
            continue
        if fname.endswith('.pth'):
            # Symlink heavy files to save space/time
            try:
                os.symlink(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        else:
            # Copy small files (we may patch config.json)
            shutil.copy2(src, dst)

    # Patch config.json: fix Apple-internal paths + enable int4 quantization
    config_path = os.path.join(work_dir, 'config.json')
    with open(config_path, 'r') as f:
        config_data = json.load(f)

    config_data['quantization'] = 'int4'
    config_data['decoder_model_name'] = 'mistralai/Mistral-7B-Instruct-v0.2'
    config_data['compr_base_model_name'] = 'mistralai/Mistral-7B-Instruct-v0.2'
    config_data['device_map'] = 'auto'  # Required for bitsandbytes 4-bit loading

    # generation_top_k must be <= number of documents per sample.
    # Our oracle eval provides 1 doc/sample, so default to 1.
    # Apple's differentiable_topk crashes if k > num_docs ("index k out of range").
    if generation_topk is not None:
        config_data['generation_top_k'] = int(generation_topk)
    else:
        config_data['generation_top_k'] = 1

    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)

    print(f"  ✓ Workdir: {work_dir}")
    print(f"    - Source: {ckpt_path}")
    print(f"    - Patched: quantization=int4, decoder=mistralai/Mistral-7B-Instruct-v0.2")
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
            samples.append({'question': q, 'answer': a,
                            'documents': [f"Question: {q} Answer: {a}."]})

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

def evaluate_apple(model, samples: list, batch_size: int = 1,
                   max_new_tokens: int = 32) -> dict:
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

    ckpt_path       = os.environ.get('CLARA_CKPT_PATH',
                                     '/kaggle/input/datasets/tokiggle/clara-7b-e2e/compression-16')
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

    # Step 1: Assemble workdir (copy from read-only Kaggle input, patch config)
    print("\n[1/4] Assembling workdir...")
    work_dir = assemble_workdir(ckpt_path, generation_topk=generation_topk)

    # Step 2: Load model via Apple's modeling_clara.py (trust_remote_code)
    print("\n[2/4] Loading Apple CLaRa model (4-bit NF4)...")
    from transformers import AutoModel
    gc.collect(); torch.cuda.empty_cache()

    model = AutoModel.from_pretrained(
        work_dir, trust_remote_code=True, load_pretrained_checkpoint=True,
    )
    # Note: do NOT call model.to("cuda") — 4-bit bitsandbytes models are
    # already on GPU after from_pretrained. Calling .to() raises ValueError.
    vram = torch.cuda.memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  ✓ Model loaded. VRAM: {vram:.1f}/{total:.1f} GB")

    # Step 3: Load dataset
    print(f"\n[3/4] Loading '{dataset_name}' validation set...")
    samples = load_eval_dataset(dataset_name, eval_mode, n_val)

    # Step 4: Evaluate
    print("\n[4/4] Running evaluation...")
    results = evaluate_apple(model, samples, batch_size=batch_size,
                             max_new_tokens=max_new_tokens)

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
