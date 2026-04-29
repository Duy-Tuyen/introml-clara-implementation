"""
data/dataset.py — Generic Dataset Loader for CLaRa.

Supports multiple QA datasets with a unified interface.
Each dataset is parsed into the same standardised schema:
    {"doc": str, "question": str, "answer": str}

To add a new dataset:
  1. Add a loader entry in _DATASET_REGISTRY below.
  2. Write a _parse_<name>() function that returns (doc, question, answer).
  3. Register the dataset_name in CLaRaConfig's __post_init__ assert.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple
from datasets import load_dataset, Dataset as HFDataset
from torch.utils.data import Dataset, DataLoader
import torch


# ── Dataset Registry ──────────────────────────────────────────────────────────
# Maps cfg.dataset_name → (hf_path, hf_config, split_map)
# split_map renames canonical splits so every dataset exposes 'train'/'validation'.

_DATASET_REGISTRY: Dict[str, Dict] = {
    '2wikimultihop': {
        'hf_path':   '2wikimultihop',
        'hf_config': None,
        # 2WikiMultiHopQA uses 'train' / 'dev'
        'split_map': {'train': 'train', 'validation': 'dev'},
        'filter':    None,
    },
    'hotpotqa': {
        'hf_path':   'hotpot_qa',
        'hf_config': 'distractor',        # or 'fullwiki'
        'split_map': {'train': 'train', 'validation': 'validation'},
        'filter':    None,
    },
    'nq': {
        'hf_path':   'nq_open',
        'hf_config': None,
        'split_map': {'train': 'train', 'validation': 'validation'},
        'filter':    None,
    },

    # ── Novel / custom datasets ────────────────────────────────────────────
    # These are placeholders — replace hf_path with your actual HF dataset id
    # or load from a local JSON/CSV file inside _load_raw() below.
    'squad': {
        'hf_path':   'rajpurkar/squad',
        'hf_config': None,
        'split_map': {'train': 'train', 'validation': 'validation'},
        'filter':    lambda x: len(x['answers']['text']) > 0,
    },
    'triviaqa': {
        'hf_path':   'trivia_qa',
        'hf_config': 'rc.nocontext',
        # TriviaQA already uses 'train' / 'validation'
        'split_map': {'train': 'train', 'validation': 'validation'},
        'filter':    lambda x: len(x['answer']['aliases']) > 0,
    },
}


# ── Per-dataset parsers ───────────────────────────────────────────────────────
# Each parser receives one raw HF row and returns (doc, question, answer).
# 'doc' is the gold passage in oracle mode; in normal mode a retriever supplies it.

def _parse_triviaqa(row: dict, eval_mode: str) -> Tuple[str, str, str]:
    question = row['question']
    answer   = row['answer']['value']
    if eval_mode == 'oracle':
        # rc.nocontext has no passage → synthesise an oracle doc from the answer
        doc = f"Trivia context: {question} The correct answer is: {answer}."
    else:
        # 'normal' mode: caller should supply a retrieved passage.
        # Fallback to the same synthetic doc until a retriever is wired in.
        doc = f"Trivia context: {question} The correct answer is: {answer}."
    return doc, question, answer


def _parse_squad(row: dict, eval_mode: str) -> Tuple[str, str, str]:
    question = row['question']
    answer   = row['answers']['text'][0] if row['answers']['text'] else ''
    if eval_mode == 'oracle':
        doc = row['context']   # SQuAD có sẵn passage rất chuẩn
    else:
        doc = row['context']   # dùng luôn vì SQuAD luôn có context
    return doc, question, answer


def _parse_2wikimultihop(row: dict, eval_mode: str) -> Tuple[str, str, str]:
    question = row['question']
    answer   = row['answer']
    if eval_mode == 'oracle':
        # Concatenate all supporting facts as the oracle document
        facts = row.get('supporting_facts', {})
        titles    = facts.get('title', [])
        sentences = facts.get('sent_id', [])
        # Build a readable passage from context sentences
        ctx_map: dict = {}
        for title, para in zip(row.get('context', {}).get('title', []),
                               row.get('context', {}).get('sentences', [])):
            ctx_map[title] = para
        oracle_sents = []
        for title, sent_id in zip(titles, sentences):
            try:
                oracle_sents.append(ctx_map[title][sent_id])
            except (KeyError, IndexError):
                pass
        doc = ' '.join(oracle_sents) if oracle_sents else question
    else:
        doc = question   # placeholder until retriever is wired in
    return doc, question, answer


def _parse_hotpotqa(row: dict, eval_mode: str) -> Tuple[str, str, str]:
    question = row['question']
    answer   = row['answer']
    if eval_mode == 'oracle':
        # HotpotQA provides supporting_facts with paragraph titles + sentence indices
        sf_titles = set(row.get('supporting_facts', {}).get('title', []))
        ctx_titles = row['context']['title']
        ctx_sents  = row['context']['sentences']
        oracle_sents = []
        for title, sents in zip(ctx_titles, ctx_sents):
            if title in sf_titles:
                oracle_sents.extend(sents)
        doc = ' '.join(oracle_sents) if oracle_sents else question
    else:
        doc = question
    return doc, question, answer


def _parse_nq(row: dict, eval_mode: str) -> Tuple[str, str, str]:
    question = row['question']
    # nq_open only has short answers (list) and no document passage
    answer   = row['answer'][0] if row['answer'] else ''
    doc      = f"Question: {question} Answer: {answer}."
    return doc, question, answer


def _parse_medical_novel(row: dict, eval_mode: str) -> Tuple[str, str, str]:
    """
    Placeholder parser for medical_novel dataset.
    Adapt field names to match your actual dataset schema.
    """
    question = row.get('question', '')
    answer   = row.get('answer', row.get('exp', ''))
    doc      = row.get('context', f"Medical context: {question}")
    return doc, question, answer


def _parse_legal_novel(row: dict, eval_mode: str) -> Tuple[str, str, str]:
    """
    Placeholder parser for legal_novel dataset.
    Adapt field names to match your actual dataset schema.
    """
    question = row.get('question', '')
    answer   = row.get('answer', '')
    doc      = row.get('context', f"Legal context: {question}")
    return doc, question, answer


# Registry linking dataset_name → parser function
_PARSERS: Dict[str, Callable] = {
    'triviaqa':       _parse_triviaqa,
    '2wikimultihop':  _parse_2wikimultihop,
    'hotpotqa':       _parse_hotpotqa,
    'nq':             _parse_nq,
    'squad':          _parse_squad,        # ← thêm
    'medical_novel':  _parse_medical_novel,
    'legal_novel':    _parse_legal_novel,
}


# ── Generic Dataset Class ─────────────────────────────────────────────────────

class CLaRaDataset(Dataset):
    """
    Unified dataset class for CLaRa evaluation and fine-tuning.

    Every item returned by __getitem__ follows this schema:
        {"doc": str, "question": str, "answer": str}

    The collate() method tokenizes the batch and produces model-ready tensors,
    identical to what CLaRaModel.forward() and generate_answer() expect.
    """

    def __init__(self, split: str, tok, cfg, n: Optional[int] = None):
        self.tok      = tok
        self.cfg      = cfg
        self.split    = split           # 'train' or 'validation'
        self.parser   = _PARSERS[cfg.dataset_name]
        self.eval_mode = cfg.eval_mode

        raw = self._load_raw(cfg.dataset_name, split)
        if n:
            raw = raw.select(range(min(n, len(raw))))
        self.data = raw
        print(f"[{cfg.dataset_name}|{split}] {len(self.data)} samples  "
              f"(eval_mode={cfg.eval_mode})")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load_raw(self, name: str, canonical_split: str) -> HFDataset:
        """Load the HuggingFace dataset and remap split names."""
        reg   = _DATASET_REGISTRY[name]
        split = reg['split_map'].get(canonical_split, canonical_split)

        ds = load_dataset(reg['hf_path'], reg['hf_config'], split=split)

        if reg['filter'] is not None:
            ds = ds.filter(reg['filter'])
        return ds

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int) -> dict:
        """Return a standardised dict for item i."""
        row = self.data[i]
        doc, question, answer = self.parser(row, self.eval_mode)
        return {"doc": doc, "question": question, "answer": answer}

    # ── Collate (tokenization) ────────────────────────────────────────────────

    def collate_train(self, batch: list) -> dict:
        """
        Collate for training.
        Produces: doc tensors + QA tensors + labels (answer tokens only).
        """
        docs = [b['doc']      for b in batch]
        qs   = [b['question'] for b in batch]
        ans  = [b['answer']   for b in batch]

        doc_enc = self.tok(
            docs, max_length=self.cfg.doc_max_length,
            padding='max_length', truncation=True, return_tensors='pt')

        qa_enc = self.tok(
            [f"[INST] {q} [/INST] {a}" for q, a in zip(qs, ans)],
            max_length=self.cfg.max_qa_len, padding='max_length',
            truncation=True, return_tensors='pt')

        labels = qa_enc['input_ids'].clone()
        labels[labels == self.tok.pad_token_id] = -100

        return dict(
            doc_input_ids=doc_enc['input_ids'],
            doc_attention_mask=doc_enc['attention_mask'],
            question_input_ids=qa_enc['input_ids'],
            question_attention_mask=qa_enc['attention_mask'],
            labels=labels,
        )

    def collate_eval(self, batch: list) -> dict:
        """
        Collate for evaluation.
        Produces: doc tensors + question-only tensors + raw answer strings.
        (Labels are NOT included — ground truth answers returned separately.)
        """
        docs = [b['doc']      for b in batch]
        qs   = [b['question'] for b in batch]
        ans  = [b['answer']   for b in batch]   # kept as strings for EM/F1

        doc_enc = self.tok(
            docs, max_length=self.cfg.doc_max_length,
            padding='max_length', truncation=True, return_tensors='pt')

        # Prompt-only (no answer appended) so model has to generate the answer
        q_enc = self.tok(
            [f"[INST] {q} [/INST]" for q in qs],
            max_length=self.cfg.max_qa_len, padding='max_length',
            truncation=True, return_tensors='pt')

        return dict(
            doc_input_ids=doc_enc['input_ids'],
            doc_attention_mask=doc_enc['attention_mask'],
            question_input_ids=q_enc['input_ids'],
            question_attention_mask=q_enc['attention_mask'],
            answers=ans,            # list[str] — passed through for scoring
        )


# ── DataLoader factory ────────────────────────────────────────────────────────

def get_dataloaders(tokenizer, cfg) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders using CLaRaDataset.
    Uses cfg.dataset_name to select the dataset automatically.
    """
    train_ds = CLaRaDataset('train',      tokenizer, cfg, cfg.n_train)
    val_ds   = CLaRaDataset('validation', tokenizer, cfg, cfg.n_val)

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=train_ds.collate_train,
        num_workers=2,
        pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        collate_fn=val_ds.collate_eval,
        num_workers=2,
        pin_memory=True,
    )
    return train_dl, val_dl


def get_eval_loader(tokenizer, cfg, split: str = 'validation') -> DataLoader:
    """
    Convenience function: returns a single eval DataLoader for any split.
    Useful in scripts/evaluate.py without needing to build the train loader.
    """
    ds = CLaRaDataset(split, tokenizer, cfg, cfg.n_val)
    return DataLoader(
        ds,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        collate_fn=ds.collate_eval,
        num_workers=2,
        pin_memory=True,
    )