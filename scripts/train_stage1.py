"""
Stage I (SCP) training: compressor + generator with L_CE + lambda * L_MSE.
"""

from __future__ import annotations

import gc
import math
import os
import torch
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

from configs.config import CLaRaConfig
from models.clara_model import build_clara_model
from data.dataset import get_dataloaders


def _vram() -> str:
    u = torch.cuda.memory_allocated() / 1e9
    t = torch.cuda.get_device_properties(0).total_memory / 1e9
    return f"{u:.1f}/{t:.1f}GB"


def _set_trainable(model, adapters: list, train_mem_tokens: bool = True) -> None:
    for name, p in model.named_parameters():
        if p.is_floating_point():
            p.requires_grad_(False)
            if "lora_" in name and any(a in name for a in adapters):
                p.requires_grad_(True)
    if train_mem_tokens:
        model.mem_token_embed.requires_grad_(True)


def _validate(model, dl, cfg, max_b: int = 40) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for i, batch in enumerate(dl):
            if i >= max_b:
                break
            batch = {k: v.cuda() for k, v in batch.items() if isinstance(v, torch.Tensor)}
            ce_loss, mse_loss = model.forward_scp(**batch)
            loss = ce_loss + cfg.stage1_mse_lambda * mse_loss
            total += loss.item()
            n += 1
    model.train()
    return total / max(n, 1)


def train_stage1(model, train_dl, val_dl, cfg) -> None:
    os.makedirs(cfg.output_dir, exist_ok=True)

    _set_trainable(model, adapters=['compressor', 'generator'], train_mem_tokens=True)

    opt = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.stage1_lr,
        weight_decay=0.01,
    )

    total_steps = math.ceil(len(train_dl) / cfg.grad_accum) * cfg.stage1_epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    sched = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps)

    best_val, gstep = float('inf'), 0
    model.train()

    for epoch in range(cfg.stage1_epochs):
        ep_loss = 0.0
        opt.zero_grad()

        for step, batch in enumerate(train_dl):
            batch = {k: v.cuda() for k, v in batch.items() if isinstance(v, torch.Tensor)}
            ce_loss, mse_loss = model.forward_scp(**batch)
            loss = ce_loss + cfg.stage1_mse_lambda * mse_loss
            loss = loss / cfg.grad_accum
            loss.backward()
            ep_loss += loss.item() * cfg.grad_accum

            if (step + 1) % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    cfg.max_grad_norm,
                )
                opt.step()
                sched.step()
                opt.zero_grad()
                gstep += 1

                if gstep % 100 == 0:
                    print(f"  Ep{epoch+1} step{gstep:4d} | "
                          f"loss {ep_loss/(step+1):.4f} | "
                          f"lr {sched.get_last_lr()[0]:.1e} | {_vram()}")

            if step % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()

        val_loss = _validate(model, val_dl, cfg)
        train_loss = ep_loss / len(train_dl)
        print(f"\nEpoch {epoch+1}/{cfg.stage1_epochs}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  {_vram()}")

        if val_loss < best_val:
            best_val = val_loss
            ckpt = os.path.join(cfg.output_dir, f"stage1_ep{epoch+1}")
            os.makedirs(ckpt, exist_ok=True)
            model.backbone.save_pretrained(os.path.join(ckpt, 'adapters'), adapter_name='compressor')
            model.backbone.save_pretrained(os.path.join(ckpt, 'adapters'), adapter_name='generator')
            torch.save(
                {'mem_token_embed': model.mem_token_embed.data,
                 'epoch': epoch + 1,
                 'val_loss': val_loss},
                os.path.join(ckpt, 'clara_stage1_extra.pth'),
            )
            print(f" Saved → {ckpt}  (val={val_loss:.4f})\n")


if __name__ == "__main__":
    cfg = CLaRaConfig(dataset_name=os.environ.get('CLARA_DATASET', 'hotpotqa'))
    if os.environ.get('CLARA_N_TRAIN'):
        cfg.n_train = int(os.environ['CLARA_N_TRAIN'])
    if os.environ.get('CLARA_N_VAL'):
        cfg.n_val = int(os.environ['CLARA_N_VAL'])
    if os.environ.get('CLARA_GRAD_ACC'):
        cfg.grad_accum = int(os.environ['CLARA_GRAD_ACC'])
    if os.environ.get('CLARA_OUTPUT_DIR'):
        cfg.output_dir = os.environ['CLARA_OUTPUT_DIR']

    model, tokenizer = build_clara_model(cfg)
    train_dl, val_dl = get_dataloaders(tokenizer, cfg)
    train_stage1(model, train_dl, val_dl, cfg)
