from dataclasses import dataclass, field
from typing import List, Literal


@dataclass
class CLaRaConfig:
    # ── Base model ────────────────────────────────────────────────────────────
    base_model: str         = 'mistralai/Mistral-7B-Instruct-v0.2'

    # ── Architecture ──────────────────────────────────────────────────────────
    compr_rate: int         = 16
    doc_max_length: int     = 256   # Reduce to 128 if OOM
    n_memory_tokens: int    = 16    # Must equal doc_max_length // compr_rate

    # ── LoRA ──────────────────────────────────────────────────────────────────
    lora_r: int             = 16
    lora_alpha: int         = 32
    lora_dropout: float     = 0.05
    lora_targets: List[str] = field(
        default_factory=lambda: ['q_proj', 'v_proj', 'k_proj', 'o_proj'])

    # ── Training ──────────────────────────────────────────────────────────────
    batch_size: int         = 1
    grad_accum: int         = 8
    lr: float               = 2e-4
    num_epochs: int         = 1
    max_qa_len: int         = 96
    warmup_ratio: float     = 0.03
    max_grad_norm: float    = 1.0
    grad_ckpt: bool         = True
    n_train: int            = 8000
    n_val: int              = 500
    output_dir: str         = './clara-ckpts'

    # ── Dataset selection ─────────────────────────────────────────────────────
    # Supported values: 'triviaqa' | '2wikimultihop' | 'hotpotqa' | 'nq'
    # Novel/custom datasets: 'medical_novel' | 'legal_novel'
    dataset_name: str       = 'triviaqa'

    # ── Evaluation ────────────────────────────────────────────────────────────
    # 'oracle' : document is the gold context (upper-bound setting from paper)
    # 'normal' : document is retrieved by a retriever (realistic setting)
    eval_mode: Literal['oracle', 'normal'] = 'oracle'

    # Separate batch size for eval — can be larger than train since no gradients
    eval_batch_size: int    = 4

    # Path to pretrained CLaRa checkpoint (projector + mem_bias + LoRA)
    pretrained_ckpt_dir: str = './clara-ckpts/pretrained'

    def __post_init__(self):
        assert self.doc_max_length // self.compr_rate == self.n_memory_tokens, (
            f"doc_max_length ({self.doc_max_length}) // compr_rate ({self.compr_rate}) "
            f"must equal n_memory_tokens ({self.n_memory_tokens}). "
            f"Got {self.doc_max_length // self.compr_rate} != {self.n_memory_tokens}."
        )
        supported = {'triviaqa', '2wikimultihop', 'hotpotqa', 'nq',
                     'medical_novel', 'legal_novel'}
        assert self.dataset_name in supported, (
            f"dataset_name '{self.dataset_name}' not recognised. "
            f"Supported: {supported}"
        )
        assert self.eval_mode in ('oracle', 'normal'), (
            f"eval_mode must be 'oracle' or 'normal', got '{self.eval_mode}'"
        )