"""
Convert Apple E2E checkpoint to this repo's Stage II adapter format.

Expected inputs (either):
- adapters.pth (+ optional decoder_first_last_layers.pth)
- lora/ (PEFT format)

Outputs:
- <out>/adapters/query/
- <out>/adapters/generator/
- <out>/clara_stage2_extra.pth (mem_token_embed)
"""

from __future__ import annotations

import argparse
import os
import torch
from peft import get_peft_model_state_dict, load_peft_weights, set_peft_model_state_dict

from configs.config import CLaRaConfig
from models.clara_model import build_clara_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Apple E2E checkpoint")
    parser.add_argument("--input", required=True, help="Apple E2E checkpoint dir")
    parser.add_argument("--output", required=True, help="Output dir for Stage II adapters")
    return parser.parse_args()


def _load_mem_tokens(model, input_dir: str) -> None:
    decoder_path = os.path.join(input_dir, "decoder_first_last_layers.pth")
    if not os.path.exists(decoder_path):
        return

    dec = torch.load(decoder_path, map_location="cpu")
    if "mem_token_embed" in dec:
        model.mem_token_embed.data = dec["mem_token_embed"].to(model.mem_token_embed.device)
    elif "mem_bias" in dec and dec["mem_bias"].shape == model.mem_token_embed.shape:
        model.mem_token_embed.data = dec["mem_bias"].to(model.mem_token_embed.device)


def _get_lora_state(model, input_dir: str) -> dict:
    adapters_path = os.path.join(input_dir, "adapters.pth")
    lora_dir = os.path.join(input_dir, "lora")

    if os.path.exists(adapters_path):
        state = torch.load(adapters_path, map_location="cpu")
        model.backbone.set_adapter("query")
        model.backbone.load_state_dict(state, strict=False)
        return get_peft_model_state_dict(model.backbone, adapter_name="query")

    if os.path.isdir(lora_dir):
        return load_peft_weights(lora_dir)

    raise FileNotFoundError(
        "No adapters.pth or lora/ found in input checkpoint."
    )


def main() -> None:
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    cfg = CLaRaConfig()
    model, _ = build_clara_model(cfg)

    lora_state = _get_lora_state(model, args.input)
    set_peft_model_state_dict(model.backbone, lora_state, adapter_name="query")
    set_peft_model_state_dict(model.backbone, lora_state, adapter_name="generator")

    _load_mem_tokens(model, args.input)

    query_dir = os.path.join(args.output, "adapters", "query")
    gen_dir = os.path.join(args.output, "adapters", "generator")
    os.makedirs(query_dir, exist_ok=True)
    os.makedirs(gen_dir, exist_ok=True)

    model.backbone.save_pretrained(query_dir, adapter_name="query")
    model.backbone.save_pretrained(gen_dir, adapter_name="generator")

    torch.save(
        {"mem_token_embed": model.mem_token_embed.data},
        os.path.join(args.output, "clara_stage2_extra.pth"),
    )

    print(f"Converted checkpoint saved to: {args.output}")


if __name__ == "__main__":
    main()
