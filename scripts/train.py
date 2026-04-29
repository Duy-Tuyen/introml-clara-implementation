import os
import gc
import math
import torch
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup


def _vram() -> str:
    u = torch.cuda.memory_allocated() / 1e9
    t = torch.cuda.get_device_properties(0).total_memory / 1e9
    return f'{u:.1f}/{t:.1f}GB'


def _validate(model, dl, max_b: int = 40) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for i, batch in enumerate(dl):
            if i >= max_b:
                break
            
            # FIX 1: filter như train_pipeline
            tensor_batch = {k: v for k, v in batch.items() 
                           if isinstance(v, torch.Tensor)}
            
            # FIX 2: sanity check batch không rỗng
            if not tensor_batch:
                raise ValueError(
                    f"Batch {i} rỗng sau filter!\n"
                    f"Keys & types: { {k: type(v).__name__ for k, v in batch.items()} }"
                )
            
            # FIX 3: log batch đầu để verify
            if i == 0:
                print(f"[Validate] Keys kept: {list(tensor_batch.keys())}")
                print(f"[Validate] Keys dropped: "
                      f"{[k for k in batch if k not in tensor_batch]}")
            
            batch = {k: v.cuda() for k, v in tensor_batch.items()}
            total += model(**batch).loss.item()
            n += 1
    model.train()
    return total / max(n, 1)


def train_pipeline(model, train_dl, val_dl, cfg) -> None:
    """
    Vòng lặp huấn luyện chính của CLaRa.
    Được gọi từ main.py hoặc chạy độc lập qua __main__.
    Lưu checkpoint tốt nhất vào cfg.output_dir/best_ep{N}/.
    """
    os.makedirs(cfg.output_dir, exist_ok=True)

    opt = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr, weight_decay=0.01)

    total_steps = math.ceil(len(train_dl) / cfg.grad_accum) * cfg.num_epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    sched = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps)

    model.train()
    best_val, gstep = float('inf'), 0

    for epoch in range(cfg.num_epochs):
        ep_loss = 0.0
        opt.zero_grad()

        for step, batch in enumerate(train_dl):
            batch = {k: v.cuda() for k, v in batch.items()}
            loss = model(**batch).loss / cfg.grad_accum
            loss.backward()
            ep_loss += loss.item() * cfg.grad_accum

            if (step + 1) % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    cfg.max_grad_norm)
                opt.step()
                sched.step()
                opt.zero_grad()
                gstep += 1

                if gstep % 100 == 0:
                    print(f'  Ep{epoch+1} step{gstep:4d} | '
                          f'loss {ep_loss/(step+1):.4f} | '
                          f'lr {sched.get_last_lr()[0]:.1e} | {_vram()}')

            if step % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()

        val_loss = _validate(model, val_dl)
        train_loss = ep_loss / len(train_dl)
        print(f'\nEpoch {epoch+1}/{cfg.num_epochs}  '
              f'train={train_loss:.4f}  val={val_loss:.4f}  {_vram()}')

        if val_loss < best_val:
            best_val = val_loss
            ckpt = os.path.join(cfg.output_dir, f'best_ep{epoch+1}')
            os.makedirs(ckpt, exist_ok=True)
            model.backbone.save_pretrained(os.path.join(ckpt, 'lora'))
            torch.save(
                {'proj': model.proj.state_dict(),
                 'mem_bias': model.mem_bias.data,
                 'epoch': epoch + 1,
                 'val_loss': val_loss},
                os.path.join(ckpt, 'clara_extra.pth'))
            print(f' Saved → {ckpt}  (val={val_loss:.4f})\n')


if __name__ == "__main__":
    # Chạy độc lập: python -m scripts.train
    from configs.config import CLaRaConfig
    from models.clara_model import build_clara_model
    from data.dataset import get_dataloaders

    cfg = CLaRaConfig()
    model, tokenizer = build_clara_model(cfg)
    train_dl, val_dl = get_dataloaders(tokenizer, cfg)
    train_pipeline(model, train_dl, val_dl, cfg)