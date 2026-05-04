from __future__ import annotations

import os
import torch

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
    """Load PEFT adapter weights from a local directory.

    Reads directly from disk — bypasses HuggingFace Hub validation entirely
    so local absolute paths (e.g. /kaggle/working/...) never trigger
    HFValidationError from load_peft_weights().
    
    Tries: adapter_model.safetensors → .bin → .pt → any .pt file → subdirectory.
    """
    if not os.path.isdir(adapter_dir):
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    # Priority: safetensors > bin > pt
    for name in ("adapter_model.safetensors", "adapter_model.bin", "adapter_model.pt"):
        path = os.path.join(adapter_dir, name)
        if not os.path.exists(path):
            continue
        if path.endswith(".safetensors"):
            if _safe_load_file is None:
                raise RuntimeError(
                    "safetensors package is required. "
                    "Install with: pip install safetensors"
                )
            return _safe_load_file(path)
        return torch.load(path, map_location="cpu", weights_only=True)

    # Fallback 1: look for any .pt file in directory
    for fname in os.listdir(adapter_dir):
        if fname.endswith(".pt") and fname != "adapter_model.pt":
            path = os.path.join(adapter_dir, fname)
            return torch.load(path, map_location="cpu", weights_only=True)

    # Fallback 2: check if this is a parent directory with subdirectories
    # (e.g. adapters/query/ containing query/adapter_model.safetensors from nested saves)
    contents = os.listdir(adapter_dir)
    for item in contents:
        subdir = os.path.join(adapter_dir, item)
        if os.path.isdir(subdir):
            # Try to load from subdirectory recursively
            try:
                return load_peft_weights_local(subdir)
            except FileNotFoundError:
                continue

    # Debug: show directory structure
    files = os.listdir(adapter_dir) if os.path.isdir(adapter_dir) else []
    raise FileNotFoundError(
        f"No adapter weights found in: {adapter_dir}\n"
        f"Expected one of: adapter_model.safetensors / .bin / .pt\n"
        f"Directory contents: {files}"
    )