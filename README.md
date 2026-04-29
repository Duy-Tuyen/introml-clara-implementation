# CLaRa — Compressive Language-Document Representation

**Architecture:** Compressor (Mistral-7B frozen) + Memory Tokens + Decoder (Mistral-7B + LoRA)  
**Dataset:** TriviaQA `rc.nocontext`  
**VRAM:** ~13–14 GB với 4-bit NF4 + gradient checkpointing + batch_size=1

---

## Cấu trúc repo

```
clara-repo/
├── configs/
│   └── config.py          # Trạm điều khiển trung tâm: Chỉnh model, dataset, hyperparameters
├── models/
│   ├── clara_model.py     # Khởi tạo kiến trúc CLaRa (Cốt lõi không thay đổi)
│   └── utils.py           # Tiện ích theo dõi phần cứng (VRAM)
├── data/
│   └── dataset.py         # Generic Loader: Tự động parse data tùy theo config.py
├── scripts/
│   ├── train.py           # Pipeline huấn luyện (Train from scratch / Fine-tune)
│   ├── evaluate.py        # Pipeline chấm điểm (Tính EM, F1) & xuất báo cáo CSV
│   └── inference.py       # Test nhanh QA (Predict)
├── notebooks/
│   └── analysis.ipynb     # Phân tích thực nghiệm: Đọc CSV và vẽ đồ thị 
├── results/
│   └── eval_scores.csv    # File lịch sử điểm số tự động sinh ra sau khi Evaluate
├── setup_env.py           # Script vá lỗi môi trường (bitsandbytes/torchvision) trên Kaggle
└── requirements.txt       # Danh sách thư viện chuẩn
```

---

## Chạy trên Kaggle
Notebook sẵn để chạy full pipeline theo paper: [notebook/clara_fullpaper_kaggle.ipynb](notebook/clara_fullpaper_kaggle.ipynb)

### Bước 1 — Pull repo từ GitHub
```python
# Trong Kaggle Notebook (cell đầu tiên)
!git clone https://github.com/<your-username>/clara-repo.git
%cd clara-repo
```

### Bước 2 — Setup môi trường (chạy 1 lần, sau đó restart kernel)
```python
%run setup_env.py
```

### Bước 3 — Chạy toàn bộ pipeline
```python
%run main.py
```

### Tải pretrained E2E (khuyến nghị)
```python
!python -m scripts.download_pretrained --repo apple/CLaRa-7B-E2E --out ./clara-ckpts/pretrained-e2e
```
Sau đó chạy evaluate hoặc fine-tune, config mặc định sẽ dùng `./clara-ckpts/pretrained-e2e`.

### Chạy theo paper (Stage I + Stage II)
```python
# Stage I (SCP): HotpotQA, L_CE + lambda * L_MSE
!python -m scripts.train_stage1

# Stage II (E2E): differentiable retrieval + ST estimator
!python -m scripts.train_stage2

# Evaluate Stage II (retrieval-based)
!python -m scripts.evaluate
```

### Hoặc chạy từng bước riêng lẻ (legacy/simplified)
Open config/config.py để chỉnh dataset_name và eval_mode, sau đó chạy:
```python
# Chỉ train
!python -m scripts.train

# Chỉ evaluate 
!python -m scripts.evaluate

# Chỉ inference (cần có checkpoint)
!python -m scripts.inference
```

---

## Chạy local

```bash
pip install -r requirements.txt
python main.py
```

---

## Lưu checkpoint

Checkpoint tốt nhất được lưu tự động vào `./clara-ckpts/best_ep{N}/`:
```
clara-ckpts/
└── best_ep1/
    ├── lora/              # LoRA adapter (HuggingFace PEFT format)
    └── clara_extra.pth    # Projector weights + mem_bias
```

---

## Ghi chú về mức độ khớp paper
- Đã bổ sung Stage I (SCP) + Stage II (E2E) theo paper, gồm:
    - Memory tokens appended vào input.
    - Loss L_MSE alignment ở Stage I.
    - ST estimator top-k ở Stage II.
- Điều chỉnh thực tế để chạy trên Kaggle T4:
    - Dùng HotpotQA làm tested dataset thay cho data synthesis từ Qwen.
    - Giảm num_candidates và top_k so với paper (có thể tăng nếu đủ VRAM).
