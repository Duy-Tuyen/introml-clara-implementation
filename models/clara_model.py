import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training


class CLaRaModel(nn.Module):
    def __init__(self, backbone, tok, cfg):
        super().__init__()
        self.cfg, self.tok = cfg, tok
        self.H = backbone.config.hidden_size

        backbone = prepare_model_for_kbit_training(
            backbone, use_gradient_checkpointing=cfg.grad_ckpt)

        lora_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_targets,
            bias='none',
            task_type=TaskType.CAUSAL_LM,
        )

        self.backbone = get_peft_model(backbone, lora_cfg, adapter_name='compressor')
        self.backbone.add_adapter('query', lora_cfg)
        self.backbone.add_adapter('generator', lora_cfg)
        self.backbone.set_adapter('compressor')

        self.mem_token_embed = nn.Parameter(
            torch.zeros(cfg.n_memory_tokens, self.H, dtype=torch.bfloat16))

    @property
    def device(self):
        """Get the device of the backbone model."""
        return next(self.backbone.parameters()).device

    @property
    def _embed(self):
        return self.backbone.base_model.model.model.embed_tokens

    def _run_with_memory_tokens(self, adapter: str, input_ids, attention_mask):
        self.backbone.set_adapter(adapter)
        tok_emb = self._embed(input_ids)
        B = tok_emb.size(0)
        mem = self.mem_token_embed.unsqueeze(0).expand(B, -1, -1).to(self.device)
        emb = torch.cat([tok_emb, mem], dim=1)
        attention_mask = attention_mask.to(self.device)
        mm = torch.ones(B, mem.size(1), device=self.device, dtype=attention_mask.dtype)
        mask = torch.cat([attention_mask, mm], dim=1)
        out = self.backbone(
            inputs_embeds=emb,
            attention_mask=mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hs = out.hidden_states[-1]
        doc_h = hs[:, :tok_emb.size(1)]
        mem_h = hs[:, tok_emb.size(1):]
        return doc_h, mem_h, mask

    def _compress_docs(self, doc_ids, doc_mask, track_grad: bool = False):
        if track_grad:
            doc_h, mem_h, _ = self._run_with_memory_tokens('compressor', doc_ids, doc_mask)
        else:
            with torch.no_grad():
                doc_h, mem_h, _ = self._run_with_memory_tokens('compressor', doc_ids, doc_mask)
        return doc_h, mem_h

    def _encode_query(self, question_ids, question_mask):
        _, mem_h, _ = self._run_with_memory_tokens('query', question_ids, question_mask)
        return mem_h

    def _mse_alignment(self, doc_h, mem_h, doc_mask):
        mask = doc_mask.unsqueeze(-1).to(doc_h.dtype)
        doc_avg = (doc_h * mask).sum(1) / mask.sum(1).clamp(1)
        mem_avg = mem_h.mean(1)
        return F.mse_loss(mem_avg, doc_avg)

    def forward_scp(self, doc_input_ids, doc_attention_mask,
                    question_input_ids, question_attention_mask, labels):
        doc_h, mem_h = self._compress_docs(doc_input_ids, doc_attention_mask, track_grad=True)
        mse_loss = self._mse_alignment(doc_h, mem_h, doc_attention_mask)

        self.backbone.set_adapter('generator')
        mem = mem_h.to(self.device)
        q_e = self._embed(question_input_ids).to(self.device)
        question_attention_mask = question_attention_mask.to(self.device)
        labels = labels.to(self.device)
        emb = torch.cat([mem, q_e], dim=1)
        mm = torch.ones(mem.size(0), mem.size(1), device=self.device,
                        dtype=question_attention_mask.dtype)
        mask = torch.cat([mm, question_attention_mask], dim=1)
        pl = torch.full((mem.size(0), mem.size(1)), -100,
                        device=self.device, dtype=labels.dtype)
        labels = torch.cat([pl, labels], dim=1)

        out = self.backbone(inputs_embeds=emb, attention_mask=mask,
                            labels=labels, return_dict=True)
        return out.loss, mse_loss

    def _st_topk(self, scores, k: int, tau: float):
        # Paper Algorithm 1: iterative top-k with mask update between ranks
        # Prevents selecting the same document twice across ranks j=1..k
        B, D = scores.shape
        eps = 1e-9

        z_hard = torch.zeros(B, k, D, device=scores.device, dtype=scores.dtype)
        z_soft = torch.zeros(B, k, D, device=scores.device, dtype=scores.dtype)

        # taken tracks which docs have been selected (hard), used to mask next rank
        # Algorithm 1 step 12: taken ← min(taken + Z_hard[:,j,:], 1)
        taken = torch.zeros(B, D, device=scores.device, dtype=scores.dtype)

        s_scaled = scores / max(tau, 1e-6)  # Algorithm 1 step 3

        for j in range(k):
            # Step 8: mask ← 1 - SG(taken)  — block already-selected docs
            mask = 1.0 - taken.detach()

            # Step 9: logits_j ← s_scaled + log(mask + ε)
            logits_j = s_scaled + torch.log(mask + eps)

            # Step 10: p_j ← softmax(logits_j)
            p_j = F.softmax(logits_j, dim=-1)
            z_soft[:, j, :] = p_j

            # Step 6-7: hard selection = argmax of soft (on unmasked candidates)
            r_j = p_j.argmax(dim=-1)  # (B,)
            z_hard[:, j, :].scatter_(1, r_j.unsqueeze(-1), 1.0)

            # Step 12: update taken with the hard selection
            taken = torch.clamp(taken + z_hard[:, j, :].detach(), max=1.0)

        # Step 14: Z = Z_hard + (Z_soft - SG(Z_soft))
        z = z_hard + (z_soft - z_soft.detach())
        return z

    def forward_e2e(self, candidate_doc_input_ids, candidate_doc_attention_mask, candidate_mask,
                    question_input_ids, question_attention_mask, labels,
                    # aliases kept for back-compat
                    candidate_doc_ids=None, candidate_doc_mask=None):
        # support both naming conventions
        if candidate_doc_ids is not None:
            candidate_doc_input_ids = candidate_doc_ids
        if candidate_doc_mask is not None:
            candidate_doc_attention_mask = candidate_doc_mask
        candidate_doc_ids = candidate_doc_input_ids
        candidate_doc_mask = candidate_doc_attention_mask
        B, C, L = candidate_doc_ids.shape
        flat_ids = candidate_doc_ids.view(B * C, L)
        flat_mask = candidate_doc_mask.view(B * C, L)

        doc_h, mem_h = self._compress_docs(flat_ids, flat_mask, track_grad=False)
        mem_h = mem_h.view(B, C, self.cfg.n_memory_tokens, self.H)

        q_mem = self._encode_query(question_input_ids, question_attention_mask)
        q_vec = q_mem.mean(1)
        d_vec = mem_h.mean(2)
        scores = F.cosine_similarity(q_vec.unsqueeze(1), d_vec, dim=-1)
        if candidate_mask is not None:
            scores = scores.masked_fill(candidate_mask == 0, -1e9)

        z = self._st_topk(scores, self.cfg.top_k, self.cfg.st_tau)
        selected = torch.einsum('bkc,bcmd->bkmd', z, mem_h)
        selected = selected.reshape(B, self.cfg.top_k * self.cfg.n_memory_tokens, self.H)

        self.backbone.set_adapter('generator')
        q_e = self._embed(question_input_ids).to(self.device)
        selected = selected.to(self.device)
        question_attention_mask = question_attention_mask.to(self.device)
        labels = labels.to(self.device)
        emb = torch.cat([selected, q_e], dim=1)
        mm = torch.ones(B, selected.size(1), device=self.device,
                        dtype=question_attention_mask.dtype)
        mask = torch.cat([mm, question_attention_mask], dim=1)
        pl = torch.full((B, selected.size(1)), -100,
                        device=self.device, dtype=labels.dtype)
        labels = torch.cat([pl, labels], dim=1)

        return self.backbone(inputs_embeds=emb, attention_mask=mask,
                             labels=labels, return_dict=True)

    @torch.no_grad()
    def generate_answer_e2e(self, candidate_doc_input_ids=None, candidate_doc_attention_mask=None,
                            candidate_mask=None, question_input_ids=None, question_attention_mask=None,
                            max_new_tokens=64,
                            candidate_doc_ids=None, candidate_doc_mask=None):
        # support both naming conventions
        if candidate_doc_ids is not None:
            candidate_doc_input_ids = candidate_doc_ids
        if candidate_doc_mask is not None:
            candidate_doc_attention_mask = candidate_doc_mask
        self.eval()
        B, C, L = candidate_doc_input_ids.shape
        flat_ids = candidate_doc_input_ids.view(B * C, L)
        flat_mask = candidate_doc_attention_mask.view(B * C, L)

        _, mem_h = self._compress_docs(flat_ids, flat_mask, track_grad=False)
        mem_h = mem_h.view(B, C, self.cfg.n_memory_tokens, self.H)

        q_mem = self._encode_query(question_input_ids, question_attention_mask)
        q_vec = q_mem.mean(1)
        d_vec = mem_h.mean(2)
        scores = F.cosine_similarity(q_vec.unsqueeze(1), d_vec, dim=-1)
        if candidate_mask is not None:
            scores = scores.masked_fill(candidate_mask == 0, -1e9)

        topk_idx = torch.topk(scores, k=self.cfg.top_k, dim=-1).indices
        selected = mem_h.gather(
            1,
            topk_idx.unsqueeze(-1).unsqueeze(-1).expand(
                B, self.cfg.top_k, self.cfg.n_memory_tokens, self.H
            ),
        )
        selected = selected.reshape(B, self.cfg.top_k * self.cfg.n_memory_tokens, self.H)

        self.backbone.set_adapter('generator')
        # Ensure all inputs are on backbone device before generation
        q_e = self._embed(question_input_ids).to(self.device)
        selected = selected.to(self.device)
        question_input_ids = question_input_ids.to(self.device)
        question_attention_mask = question_attention_mask.to(self.device)
        emb = torch.cat([selected, q_e], dim=1)
        mm = torch.ones(B, selected.size(1), device=self.device,
                        dtype=question_attention_mask.dtype)
        mask = torch.cat([mm, question_attention_mask], dim=1)

        ids = self.backbone.generate(
            inputs_embeds=emb,
            attention_mask=mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tok.eos_token_id,
        )

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
        cfg.base_model, quantization_config=bnb, device_map="cuda",
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    base.config.use_cache = False
    if cfg.grad_ckpt:
        base.gradient_checkpointing_enable()

    model = CLaRaModel(base, tokenizer, cfg)
    return model, tokenizer
        if p.is_floating_point():
            p.requires_grad_(False)

    return model, tokenizer