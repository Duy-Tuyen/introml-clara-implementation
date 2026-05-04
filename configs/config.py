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
    use_memory_tokens: bool = True  # Paper: append memory tokens to inputs

    # ── LoRA ──────────────────────────────────────────────────────────────────
    lora_r: int             = 16
    lora_alpha: int         = 32
    lora_dropout: float     = 0.1    # Paper Table 10: LoRA Dropout = 0.1
    lora_targets: List[str] = field(
        default_factory=lambda: ['q_proj', 'v_proj', 'k_proj', 'o_proj'])

    # ── Training (Stage I: SCP) ───────────────────────────────────────────────
    # Paper LR = 1e-4, nhưng giảm xuống 2e-5 để tránh loss diverge trên T4
    stage1_lr: float        = 2e-5
    stage1_epochs: int      = 3
    stage1_mse_lambda: float = 0.1

    # ── Training (Stage II: End-to-End) ───────────────────────────────────────
    batch_size: int         = 1
    # grad_accum=16 để gradient ổn định hơn với lr nhỏ
    grad_accum: int         = 16
    # Paper B.4: end-to-end learning rate = 5e-6
    lr: float               = 5e-6
    num_epochs: int         = 3
    max_qa_len: int         = 96
    warmup_ratio: float     = 0.03
    max_grad_norm: float    = 1.0
    grad_ckpt: bool         = True
    # Giảm từ 8000 → 500 để kịp Kaggle T4 quota (42000s)
    # 500 steps × ~68s/step = ~9500s/stage → tổng ~20000s (an toàn)
    n_train: int            = 500
    n_val: int              = 100
    output_dir: str         = './clara-ckpts'

    # ── Retrieval (Stage II) ─────────────────────────────────────────────────-
    num_candidates: int     = 8     # candidates per question (paper uses larger)
    top_k: int              = 2     # selected documents (paper uses 5)
    st_tau: float           = 0.7   # temperature for ST estimator

    # Stage I checkpoint (used to init Stage II)
    stage1_ckpt_dir: str    = './clara-ckpts/stage1_ep1'

    # ── Dataset selection ─────────────────────────────────────────────────────
    # Supported: 'triviaqa' | '2wikimultihop' | 'hotpotqa' | 'nq' | 'squad'
    # Novel/custom: 'medical_novel' | 'legal_novel'
    dataset_name: str       = 'hotpotqa'

    # ── Evaluation ────────────────────────────────────────────────────────────
    # 'oracle' : document là gold context (upper-bound, theo paper)
    # 'normal' : document được retrieved bởi retriever (realistic)
    eval_mode: Literal['oracle', 'normal'] = 'oracle'

    # Batch size riêng cho eval — có thể lớn hơn train vì không cần gradient
    eval_batch_size: int    = 4

    # Path đến pretrained CLaRa checkpoint (Apple E2E weights)
    pretrained_ckpt_dir: str = './clara-ckpts/pretrained-e2e'

    def __post_init__(self):
        assert self.doc_max_length // self.compr_rate == self.n_memory_tokens, (
            f"doc_max_length ({self.doc_max_length}) // compr_rate ({self.compr_rate}) "
            f"must equal n_memory_tokens ({self.n_memory_tokens}). "
            f"Got {self.doc_max_length // self.compr_rate} != {self.n_memory_tokens}."
        )
        # FIX: thêm 'squad' vào supported datasets
        supported = {'triviaqa', '2wikimultihop', 'hotpotqa', 'nq',
                     'squad',
                     'medical_novel', 'legal_novel'}
        assert self.dataset_name in supported, (
            f"dataset_name '{self.dataset_name}' not recognised. "
            f"Supported: {supported}"
        )
        assert self.eval_mode in ('oracle', 'normal'), (
            f"eval_mode must be 'oracle' or 'normal', got '{self.eval_mode}'"
        )
        assert 1 <= self.top_k <= self.num_candidates, (
            f"top_k ({self.top_k}) must be between 1 and num_candidates ({self.num_candidates})"
        )