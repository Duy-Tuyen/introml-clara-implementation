"""
setup_env.py — Chạy cell này TRƯỚC KHI RESTART KERNEL trên Kaggle.
Sửa các lỗi CUDA/torchvision thường gặp trên môi trường Kaggle GPU.
"""
import subprocess
import os
import glob
import torch


def run(cmd: str) -> None:
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if r.stdout:
        print(r.stdout[-300:])
    if r.stderr and 'error' in r.stderr.lower():
        print(r.stderr[-200:])


def setup_kaggle_env() -> None:
    # 1. Xoá torchvision bị broken
    run('pip uninstall -y torchvision torchaudio 2>/dev/null || true')

    # 2. Cài dependencies
    run('pip install -qU transformers==4.43.3 peft==0.11.1 bitsandbytes==0.43.1 '
        'datasets accelerate triton==2.2.0')

    # 3. Patch bitsandbytes để nhận CUDA 12.8 của Kaggle
    bnb_dir = '/usr/local/lib/python3.12/dist-packages/bitsandbytes'
    so_128 = os.path.join(bnb_dir, 'libbitsandbytes_cuda128.so')
    if not os.path.exists(so_128):
        so_files = sorted(glob.glob(os.path.join(bnb_dir, 'libbitsandbytes_cuda12*.so')))
        if so_files:
            best_so = so_files[-1]
            os.symlink(best_so, so_128)
            print(f"Patched bitsandbytes CUDA 12.8 → {os.path.basename(best_so)}")

    # 4. Tạo stub torchvision để transformers không crash khi import
    tv_dir = '/usr/local/lib/python3.12/dist-packages/torchvision'
    os.makedirs(tv_dir, exist_ok=True)
    with open(os.path.join(tv_dir, '__init__.py'), 'w') as f:
        f.write('# stub\n__version__ = "0.0.0"\n')

    print(f'Torch: {torch.__version__} | CUDA: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print('✅ Môi trường Kaggle đã sẵn sàng. Hãy RESTART KERNEL rồi chạy tiếp.')


if __name__ == "__main__":
    setup_kaggle_env()