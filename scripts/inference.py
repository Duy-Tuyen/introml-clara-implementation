import os
import torch
from peft import load_peft_weights, set_peft_model_state_dict

from configs.config import CLaRaConfig
from models.clara_model import build_clara_model
from models.utils import print_vram_usage


def load_checkpoint(model, cfg, epoch: int = 1) -> None:
    """Load projector, mem_bias và LoRA weights từ checkpoint đã lưu."""
    ckpt_dir = os.path.join(cfg.output_dir, f'best_ep{epoch}')
    ckpt_path = os.path.join(ckpt_dir, 'clara_extra.pth')
    lora_dir = os.path.join(ckpt_dir, 'lora')

    if os.path.exists(ckpt_path):
        saved = torch.load(ckpt_path, map_location='cuda')
        model.proj.load_state_dict(saved['proj'])
        model.mem_bias.data = saved['mem_bias']
        print(f'Loaded Projector & MemBias from: {ckpt_path}')

    if os.path.exists(lora_dir):
        lora_weights = load_peft_weights(lora_dir)
        set_peft_model_state_dict(model.backbone, lora_weights)
        print(f'Loaded LoRA adapter from: {lora_dir}')


def run_inference(model, tokenizer, cfg, tests: list) -> None:
    """Chạy inference trên danh sách test cases."""
    model.eval()
    with torch.no_grad():
        for t in tests:
            de = tokenizer(
                t['doc'], max_length=cfg.doc_max_length,
                padding='max_length', truncation=True, return_tensors='pt')
            qe = tokenizer(
                f"[INST] {t['q']} [/INST]", max_length=cfg.max_qa_len,
                padding='max_length', truncation=True, return_tensors='pt')

            ans = model.generate_answer(
                de['input_ids'].to('cuda:0'), de['attention_mask'].to('cuda:0'),
                qe['input_ids'].to('cuda:0'), qe['attention_mask'].to('cuda:0'),
                max_new_tokens=32)

            print(f"Q: {t['q']}")
            print(f"Expected: {t.get('expected', '?')}  |  Model: {ans[0]}")
            print('-' * 50)


if __name__ == "__main__":
    # Chạy độc lập: python -m scripts.inference
    cfg = CLaRaConfig()
    model, tokenizer = build_clara_model(cfg)
    load_checkpoint(model, cfg, epoch=1)

    print(print_vram_usage())

    tests = [
        {'doc': 'Weldenia is a monotypic genus native to Mexico and Guatemala.',
         'q': 'Which genus grows originally in Mexico and Guatemala, Phylica or Weldenia?',
         'expected': 'Weldenia'},
        {'doc': 'The Battle of Hastings was fought on 14 October 1066.',
         'q': 'In what year was the Battle of Hastings fought?',
         'expected': '1066'},
    ]
    run_inference(model, tokenizer, cfg, tests)