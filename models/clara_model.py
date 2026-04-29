import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training


class CLaRaModel(nn.Module):
    def __init__(self, backbone, tok, cfg):
        super().__init__()
        self.cfg, self.tok = cfg, tok
        H = backbone.config.hidden_size

        backbone = prepare_model_for_kbit_training(
            backbone, use_gradient_checkpointing=cfg.grad_ckpt)

        self.backbone = get_peft_model(backbone, LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_targets,
            bias='none', task_type=TaskType.CAUSAL_LM))

        self.proj = nn.Sequential(
            nn.Linear(H, H, bias=False),
            nn.GELU(),
            nn.Linear(H, H * cfg.n_memory_tokens, bias=False),
            nn.LayerNorm(H * cfg.n_memory_tokens),
        ).to(dtype=torch.bfloat16, device='cuda')

        self.mem_bias = nn.Parameter(
            torch.zeros(cfg.n_memory_tokens, H, dtype=torch.bfloat16, device='cuda'))

    @property
    def _embed(self):
        return self.backbone.base_model.model.model.embed_tokens

    def _compress(self, doc_ids, doc_mask):
        with torch.no_grad():
            hs = self.backbone.base_model(
                input_ids=doc_ids, attention_mask=doc_mask,
                output_hidden_states=True, return_dict=True,
            ).hidden_states[-1]

        mask = doc_mask.unsqueeze(-1).to(hs.dtype)
        pooled = (hs * mask).sum(1) / mask.sum(1).clamp(1)
        B, H = pooled.shape

        target_device = next(self.proj.parameters()).device
        pooled = pooled.to(target_device)
        mem = self.proj(pooled.to(torch.bfloat16)).view(B, self.cfg.n_memory_tokens, H)
        return mem + self.mem_bias.to(target_device).unsqueeze(0)

    def forward(self, doc_input_ids, doc_attention_mask,
                question_input_ids, question_attention_mask, labels=None):
        B, K = doc_input_ids.size(0), self.cfg.n_memory_tokens
        dev = question_attention_mask.device
        mem = self._compress(doc_input_ids, doc_attention_mask).to(dev)
        q_e = self._embed(question_input_ids).to(dev)
        emb = torch.cat([mem, q_e], dim=1)
        mm = torch.ones(B, K, device=dev, dtype=question_attention_mask.dtype)
        mask = torch.cat([mm, question_attention_mask], dim=1)

        if labels is not None:
            pl = torch.full((B, K), -100, device=dev, dtype=labels.dtype)
            labels = torch.cat([pl, labels], dim=1)

        return self.backbone(inputs_embeds=emb, attention_mask=mask,
                             labels=labels, return_dict=True)

    @torch.no_grad()
    def generate_answer(self, doc_input_ids, doc_attention_mask,
                        question_input_ids, question_attention_mask, max_new_tokens=64):
        self.eval()
        B, K = doc_input_ids.size(0), self.cfg.n_memory_tokens
        dev = question_attention_mask.device
        mem = self._compress(doc_input_ids, doc_attention_mask).to(dev)
        q_e = self._embed(question_input_ids).to(dev)
        emb = torch.cat([mem, q_e], dim=1)
        mm = torch.ones(B, K, device=dev, dtype=question_attention_mask.dtype)
        mask = torch.cat([mm, question_attention_mask], dim=1)

        ids = self.backbone.generate(
            inputs_embeds=emb, attention_mask=mask,
            max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=self.tok.eos_token_id)

        return self.tok.batch_decode(ids, skip_special_tokens=True)


def build_clara_model(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    base = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, quantization_config=bnb, device_map={'': 0},
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    base.config.use_cache = False
    if cfg.grad_ckpt:
        base.gradient_checkpointing_enable()

    model = CLaRaModel(base, tokenizer, cfg)

    for name, p in model.named_parameters():
        if p.is_floating_point():
            p.requires_grad_(any(k in name for k in ('lora_', 'proj.', 'mem_bias')))

    return model, tokenizer