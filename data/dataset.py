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

_DATASET_REGISTRY: Dict[str, Dict] = {
    'triviaqa': {
        'hf_path':   'trivia_qa',
        'hf_config': 'rc.nocontext',
        'split_map': {'train': 'train', 'validation': 'validation'},
        'filter':    lambda x: len(x['answer']['aliases']) > 0,
    },
    '2wikimultihop': {
        'hf_path':   '2wikimultihop',
        'hf_config': None,
        'split_map': {'train': 'train', 'validation': 'dev'},
        'filter':    None,
    },
    'hotpotqa': {
        'hf_path':   'hotpot_qa',
        'hf_config': 'distractor',
        'split_map': {'train': 'train', 'validation': 'validation'},
        'filter':    None,
    },
    'nq': {
        'hf_path':   'nq_open',
        'hf_config': None,
        'split_map': {'train': 'train', 'validation': 'validation'},
        'filter':    None,
    },
    # FIX: thêm SQuAD — dataset mới để eval
    'squad': {
        'hf_path':   'rajpurkar/squad',
        'hf_config': None,
        'split_map': {'train': 'train', 'validation': 'validation'},
        'filter':    lambda x: len(x['answers']['text']) > 0,
    },
    # ── Novel / custom datasets ────────────────────────────────────────────
    'medical_novel': {
        'hf_path':   'medmcqa',
        'hf_config': None,
        'split_map': {'train': 'train', 'validation': 'validation'},
        'filter':    None,
    },
    'legal_novel': {
        'hf_path':   'nguyen-brat/legal_qa',
        'hf_config': None,
        'split_map': {'train': 'train', 'validation': 'validation'},
        'filter':    None,
    },
}


# ── Per-dataset parsers ───────────────────────────────────────────────────────

def _parse_triviaqa(row: dict, eval_mode: str) -> Tuple[str, str, str]:
    question = row['question']
    answer   = row['answer']['value']
    doc      = f"Trivia context: {question} The correct answer is: {answer}."
    return doc, question, answer


def _parse_2wikimultihop(row: dict, eval_mode: str) -> Tuple[str, str, str]:
    question = row['question']
    answer   = row['answer']
    if eval_mode == 'oracle':
        facts  = row.get('supporting_facts', {})
        titles = facts.get('title', [])
        sents  = facts.get('sent_id', [])
        ctx_map: dict = {}
        for title, para in zip(row.get('context', {}).get('title', []),
                               row.get('context', {}).get('sentences', [])):
            ctx_map[title] = para
        oracle_sents = []
        for title, sent_id in zip(titles, sents):
            try:
                oracle_sents.append(ctx_map[title][sent_id])
            except (KeyError, IndexError):
                pass
        doc = ' '.join(oracle_sents) if oracle_sents else question
    else:
        doc = question
    return doc, question, answer


def _parse_hotpotqa(row: dict, eval_mode: str) -> Tuple[str, str, str]:
    question = row['question']
    answer   = row['answer']
    if eval_mode == 'oracle':
        sf_titles  = set(row.get('supporting_facts', {}).get('title', []))
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
    answer   = row['answer'][0] if row['answer'] else ''
    doc      = f"Question: {question} Answer: {answer}."
    return doc, question, answer


def _parse_squad(row: dict, eval_mode: str) -> Tuple[str, str, str]:
    """
    SQuAD luôn có passage context sẵn → oracle mode tự nhiên, không cần giả lập.
    """
    question = row['question']
    answer   = row['answers']['text'][0] if row['answers']['text'] else ''
    doc      = row['context']   # passage thật, không phải synthetic
    return doc, question, answer


def _parse_medical_novel(row: dict, eval_mode: str) -> Tuple[str, str, str]:
    question = row.get('question', '')
    answer   = row.get('answer', row.get('exp', ''))
    doc      = row.get('context', f"Medical context: {question}")
    return doc, question, answer


def _parse_legal_novel(row: dict, eval_mode: str) -> Tuple[str, str, str]:
    question = row.get('question', '')
    answer   = row.get('answer', '')
    doc      = row.get('context', f"Legal context: {question}")
    return doc, question, answer


_PARSERS: Dict[str, Callable] = {
    'triviaqa':       _parse_triviaqa,
    '2wikimultihop':  _parse_2wikimultihop,
    'hotpotqa':       _parse_hotpotqa,
    'nq':             _parse_nq,
    'squad':          _parse_squad,        # FIX: thêm squad
    'medical_novel':  _parse_medical_novel,
    'legal_novel':    _parse_legal_novel,
}


# ── Generic Dataset Class ─────────────────────────────────────────────────────

class CLaRaDataset(Dataset):
    """
    Unified dataset class for CLaRa evaluation and fine-tuning.

    collate_train : dùng cho train loop và _validate (có labels tensor)
    collate_eval  : dùng cho scripts/evaluate.py (có answers string, không có labels)
    """

    def __init__(self, split: str, tok, cfg, n: Optional[int] = None):
        self.tok       = tok
        self.cfg       = cfg
        self.split     = split
        self.parser    = _PARSERS[cfg.dataset_name]
        self.eval_mode = cfg.eval_mode

        raw = self._load_raw(cfg.dataset_name, split)
        if n:
            raw = raw.select(range(min(n, len(raw))))
        self.data = raw
        print(f"[{cfg.dataset_name}|{split}] {len(self.data)} samples  "
              f"(eval_mode={cfg.eval_mode})")

    def _load_raw(self, name: str, canonical_split: str) -> HFDataset:
        reg   = _DATASET_REGISTRY[name]
        split = reg['split_map'].get(canonical_split, canonical_split)
        ds    = load_dataset(reg['hf_path'], reg['hf_config'], split=split)
        if reg['filter'] is not None:
            ds = ds.filter(reg['filter'])
        return ds

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int) -> dict:
        row = self.data[i]
        doc, question, answer = self.parser(row, self.eval_mode)
        return {"doc": doc, "question": question, "answer": answer}

    def collate_train(self, batch: list) -> dict:
        """
        Collate cho train loop và _validate.
        Tất cả values đều là Tensor — không có list string.
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
        Collate cho scripts/evaluate.py — generate answer + tính EM/F1.
        answers là list[str] (KHÔNG phải tensor) — chỉ dùng cho evaluate, không cho _validate.
        """
        docs = [b['doc']      for b in batch]
        qs   = [b['question'] for b in batch]
        ans  = [b['answer']   for b in batch]

        doc_enc = self.tok(
            docs, max_length=self.cfg.doc_max_length,
            padding='max_length', truncation=True, return_tensors='pt')

        q_enc = self.tok(
            [f"[INST] {q} [/INST]" for q in qs],
            max_length=self.cfg.max_qa_len, padding='max_length',
            truncation=True, return_tensors='pt')

        return dict(
            doc_input_ids=doc_enc['input_ids'],
            doc_attention_mask=doc_enc['attention_mask'],
            question_input_ids=q_enc['input_ids'],
            question_attention_mask=q_enc['attention_mask'],
            answers=ans,    # list[str] — chỉ dùng trong evaluate.py
        )


# ── DataLoader factory ────────────────────────────────────────────────────────

def get_dataloaders(tokenizer, cfg) -> Tuple[DataLoader, DataLoader]:
    """
    Build train và validation DataLoaders.

    Cả train_dl và val_dl đều dùng collate_train vì _validate cần labels tensor.
    collate_eval chỉ dùng trong get_eval_loader (cho scripts/evaluate.py).
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
    # FIX: đổi từ collate_eval → collate_train
    # _validate cần labels tensor để tính loss
    # collate_eval trả về answers là list[str] → không thể .cuda() → crash
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        collate_fn=val_ds.collate_train,   # ← FIX
        num_workers=2,
        pin_memory=True,
    )
    return train_dl, val_dl


def get_eval_loader(tokenizer, cfg, split: str = 'validation') -> DataLoader:
    """
    Dùng trong scripts/evaluate.py — generate answer và tính EM/F1.
    Dùng collate_eval vì evaluate cần answers string, không cần loss.
    """
    ds = CLaRaDataset(split, tokenizer, cfg, cfg.n_val)
    return DataLoader(
        ds,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        collate_fn=ds.collate_eval,   # ← đúng, chỉ dùng ở đây
        num_workers=2,
        pin_memory=True,
    )