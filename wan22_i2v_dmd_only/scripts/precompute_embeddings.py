#!/usr/bin/env python3
import argparse
import os

import lmdb
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm

from utils.lmdb import get_array_shape_from_lmdb, retrieve_row_from_lmdb
from utils.wan_wrapper import WanTextEncoder, WanVAEWrapper


def _open_envs(lmdb_path: str, map_size: int):
    data_mdb = os.path.join(lmdb_path, "data.mdb")
    if os.path.exists(data_mdb):
        env = lmdb.open(
            lmdb_path,
            map_size=map_size,
            subdir=True,
            readonly=False,
            metasync=True,
            sync=True,
            lock=True,
            readahead=False,
            meminit=False,
        )
        return [(lmdb_path, env)]

    envs = []
    for fname in sorted(os.listdir(lmdb_path)):
        shard_path = os.path.join(lmdb_path, fname)
        if not os.path.isdir(shard_path):
            continue
        env = lmdb.open(
            shard_path,
            map_size=map_size,
            subdir=True,
            readonly=False,
            metasync=True,
            sync=True,
            lock=True,
            readahead=False,
            meminit=False,
        )
        envs.append((shard_path, env))
    return envs


def _dtype_from_arg(name: str):
    name = name.lower()
    if name == "fp16":
        return torch.float16, np.float16
    if name == "fp32":
        return torch.float32, np.float32
    if name == "bf16":
        return torch.bfloat16, np.float16
    raise ValueError("compute_dtype must be one of: fp16, fp32, bf16")


def _ensure_dir(path: str, label: str, allow_skip: bool):
    if allow_skip:
        return
    if not path:
        raise ValueError(f"{label} must be set unless the corresponding --skip flag is used.")
    os.makedirs(path, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Precompute embeddings to disk from LMDB.")
    parser.add_argument("--lmdb_path", type=str, required=True, help="Path to LMDB (root or shard parent).")
    parser.add_argument("--model_name", type=str, required=True, help="Wan model name (for tokenizer/VAE).")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--map_size", type=int, default=int(1e12))
    parser.add_argument("--compute_dtype", type=str, default="bf16", choices=["fp16", "fp32", "bf16"])
    parser.add_argument("--skip_prompt_embeds", action="store_true")
    parser.add_argument("--skip_image_embeds", action="store_true")
    parser.add_argument("--prompt_embeds_dir", type=str, default="", help="Output folder for prompt embeddings.")
    parser.add_argument("--image_embeds_dir", type=str, default="", help="Output folder for image embeddings.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype, np_dtype = _dtype_from_arg(args.compute_dtype)
    if args.compute_dtype == "bf16":
        print("Warning: storing bf16 as float16 on disk (numpy has no bf16).")

    do_prompt = not args.skip_prompt_embeds
    do_image = not args.skip_image_embeds
    _ensure_dir(args.prompt_embeds_dir, "prompt_embeds_dir", not do_prompt)
    _ensure_dir(args.image_embeds_dir, "image_embeds_dir", not do_image)

    if not args.overwrite:
        if do_prompt and any(f.endswith(".pt") for f in os.listdir(args.prompt_embeds_dir)):
            print("prompt_embeds files already exist; skipping (use --overwrite to re-run).")
            do_prompt = False
        if do_image and any(f.endswith(".pt") for f in os.listdir(args.image_embeds_dir)):
            print("image_embeds files already exist; skipping (use --overwrite to re-run).")
            do_image = False

    if not do_prompt and not do_image:
        print("Nothing to compute. Exiting.")
        return

    text_encoder = WanTextEncoder(model_name=args.model_name)
    vae = WanVAEWrapper(model_name=args.model_name)
    vae.dtype = torch_dtype
    vae.model = vae.model.to(device=device, dtype=torch_dtype)

    envs = _open_envs(args.lmdb_path, args.map_size)
    if not envs:
        raise FileNotFoundError(f"No LMDB envs found under {args.lmdb_path}")

    global_offset = 0
    for env_path, env in envs:
        print(f"Processing LMDB: {env_path}")
        latents_shape = get_array_shape_from_lmdb(env, "latents")
        num_samples = latents_shape[0]
        img_shape = (832, 480, 3)

        for start in tqdm(range(0, num_samples, args.batch_size), desc="precompute", unit="batch"):
            end = min(start + args.batch_size, num_samples)
            prompts = []
            imgs = []
            global_indices = list(range(global_offset + start, global_offset + end))

            for idx in range(start, end):
                if do_prompt:
                    prompts.append(retrieve_row_from_lmdb(env, "prompts", str, idx))
                if do_image:
                    img_np = retrieve_row_from_lmdb(env, "img", np.uint8, idx, shape=img_shape)
                    img = Image.fromarray(img_np)
                    img = TF.to_tensor(img).sub_(0.5).div_(0.5)
                    imgs.append(img)

            if do_prompt:
                prompt_embeds = text_encoder(text_prompts=prompts)["prompt_embeds"]
                prompt_embeds = prompt_embeds.to(device="cpu", dtype=torch_dtype)
                for i, global_idx in enumerate(global_indices):
                    out_path = os.path.join(args.prompt_embeds_dir, f"{global_idx}.pt")
                    torch.save(prompt_embeds[i], out_path)

            if do_image:
                image_list = []
                for img in imgs:
                    y = vae.run_vae_encoder(img.to(device=device))[0]
                    y = y.to(device="cpu", dtype=torch_dtype)
                    image_list.append(y)
                image_embeds = torch.stack(image_list, dim=0)
                for i, global_idx in enumerate(global_indices):
                    out_path = os.path.join(args.image_embeds_dir, f"{global_idx}.pt")
                    torch.save(image_embeds[i], out_path)

        global_offset += num_samples

    print("Finished precomputing embeddings.")


if __name__ == "__main__":
    main()
