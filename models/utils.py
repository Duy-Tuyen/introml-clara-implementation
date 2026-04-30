from __future__ import annotations

import os
import torch
from peft import load_peft_weights

try:
    from safetensors.torch import load_file as _safe_load_file
except Exception:
    _safe_load_file = None


def print_vram_usage() -> str:
    """Kiểm tra lượng VRAM GPU đang sử dụng để tránh lỗi OOM."""
    if torch.cuda.is_available():
        u = torch.cuda.memory_allocated() / 1e9
        t = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f'VRAM: {u:.1f}/{t:.1f}GB'
    return 'VRAM: N/A (Không tìm thấy GPU)'


def load_peft_weights_local(adapter_dir: str) -> dict:
    """Load PEFT adapter weights from a local directory across PEFT versions."""
    if not os.path.isdir(adapter_dir):
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    last_exc = None
    for kwargs in ({"is_local": True}, {"local_files_only": True}, {}):
        try:
            return load_peft_weights(adapter_dir, **kwargs)
        except TypeError as exc:
            last_exc = exc
            continue
        except Exception as exc:
            last_exc = exc
            break

    for name in ("adapter_model.safetensors", "adapter_model.bin", "adapter_model.pt"):
        path = os.path.join(adapter_dir, name)
        if not os.path.exists(path):
            continue
        if path.endswith(".safetensors"):
            if _safe_load_file is None:
                raise RuntimeError("safetensors is required to load adapter_model.safetensors")
            return _safe_load_file(path)
        return torch.load(path, map_location="cpu")

    if last_exc is not None:
        raise last_exc
    raise FileNotFoundError(f"No adapter weights found in: {adapter_dir}")