#!/usr/bin/env python3
import argparse
import os
from collections import OrderedDict

import torch
from safetensors.torch import save_file, load_file


def _map_key(key: str, role: str) -> str:
    if key.startswith("diffusion_model."):
        key = key[len("diffusion_model."):]
    if ".lora_down." in key:
        key = key.replace(".lora_down.", f".lora_A_{role}.")
    if ".lora_up." in key:
        key = key.replace(".lora_up.", f".lora_B_{role}.")
    return key


def convert_lora(in_path: str, out_path: str, role: str) -> None:
    if os.path.isdir(in_path):
        raise ValueError("Expected a .safetensors file, got a directory.")
    if not in_path.endswith(".safetensors"):
        raise ValueError("Input must be a .safetensors file.")

    state = load_file(in_path)
    mapped = OrderedDict()
    total = 0
    kept = 0
    for k, v in state.items():
        total += 1
        if ".lora_down." not in k and ".lora_up." not in k:
            continue
        mapped[_map_key(k, role)] = v
        kept += 1

    if kept == 0:
        raise RuntimeError("No LoRA keys found to convert.")

    save_file(mapped, out_path)
    print(f"Converted {kept}/{total} tensors -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert LoRA safetensors keys to Wan dual-role format.")
    parser.add_argument("--input", required=True, help="Input .safetensors LoRA file")
    parser.add_argument("--output", required=True, help="Output .safetensors file")
    parser.add_argument(
        "--role",
        choices=["generator", "critic"],
        default="generator",
        help="Which adapter role to target in the output keys",
    )
    args = parser.parse_args()
    convert_lora(args.input, args.output, args.role)


if __name__ == "__main__":
    main()
