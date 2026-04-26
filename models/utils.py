import torch


def print_vram_usage() -> str:
    """Kiểm tra lượng VRAM GPU đang sử dụng để tránh lỗi OOM."""
    if torch.cuda.is_available():
        u = torch.cuda.memory_allocated() / 1e9
        t = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f'VRAM: {u:.1f}/{t:.1f}GB'
    return 'VRAM: N/A (Không tìm thấy GPU)'