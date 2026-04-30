import os
import torch
from peft import set_peft_model_state_dict

from configs.config import CLaRaConfig
from models.clara_model import build_clara_model
from models.utils import load_peft_weights_local, print_vram_usage


def load_checkpoint(model, cfg, epoch: int = 1) -> None:
    """Load Stage II adapters + memory tokens from checkpoint."""
    ckpt_dir = os.path.join(cfg.output_dir, f'stage2_ep{epoch}')
    query_dir = os.path.join(ckpt_dir, 'adapters', 'query')
    gen_dir = os.path.join(ckpt_dir, 'adapters', 'generator')
    extra_path = os.path.join(ckpt_dir, 'clara_stage2_extra.pth')

    if os.path.isdir(query_dir):
        q_weights = load_peft_weights_local(query_dir)
        set_peft_model_state_dict(model.backbone, q_weights, adapter_name='query')
        print(f'Loaded query adapter from: {query_dir}')

    if os.path.isdir(gen_dir):
        g_weights = load_peft_weights_local(gen_dir)
        set_peft_model_state_dict(model.backbone, g_weights, adapter_name='generator')
        print(f'Loaded generator adapter from: {gen_dir}')

    if os.path.exists(extra_path):
        saved = torch.load(extra_path, map_location='cuda')
        model.mem_token_embed.data = saved['mem_token_embed']
        print(f'Loaded memory tokens from: {extra_path}')


def run_inference(model, tokenizer, cfg, tests: list) -> None:
    """Chạy inference trên danh sách test cases."""
    model.eval()
    with torch.no_grad():
        for t in tests:
            cand = t['doc']
            de = tokenizer(
                [cand], max_length=cfg.doc_max_length,
                padding='max_length', truncation=True, return_tensors='pt')
            qe = tokenizer(
                f"[INST] {t['q']} [/INST]", max_length=cfg.max_qa_len,
                padding='max_length', truncation=True, return_tensors='pt')

            doc_ids = de['input_ids'].unsqueeze(0).to('cuda:0')
            doc_mask = de['attention_mask'].unsqueeze(0).to('cuda:0')
            cand_mask = torch.tensor([[1]], dtype=torch.long, device='cuda:0')

            ans = model.generate_answer_e2e(
                doc_ids, doc_mask, cand_mask,
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