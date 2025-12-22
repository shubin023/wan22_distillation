import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Tuple

import torch
from safetensors.torch import load_file, save_file


WEIGHT_INDEX_NAME = "diffusion_pytorch_model.safetensors.index.json"
WEIGHT_SINGLE_NAME = "diffusion_pytorch_model.safetensors"


def _load_lora_state(path: Path) -> Dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        return load_file(str(path))
    state = torch.load(str(path), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        return state["state_dict"]
    if isinstance(state, dict):
        return state
    raise ValueError(f"Unsupported LoRA checkpoint format at {path}")


def _normalize_key(key: str) -> str:
    for prefix in ("diffusion_model.", "model."):
        if key.startswith(prefix):
            key = key[len(prefix):]
    return key


def _base_key_from_lora(key: str) -> Tuple[str, str]:
    """
    Returns (base_key, role) where role is "A", "B", or "alpha".
    If the key doesn't look like a LoRA tensor, returns ("", "").
    """
    key = _normalize_key(key)
    suffix_map = {
        ".lora_A.weight": ("A", ".weight"),
        ".lora_B.weight": ("B", ".weight"),
        ".lora_A_generator.weight": ("A", ".weight"),
        ".lora_B_generator.weight": ("B", ".weight"),
        ".lora_A_critic.weight": ("A", ".weight"),
        ".lora_B_critic.weight": ("B", ".weight"),
        ".lora_down.weight": ("A", ".weight"),
        ".lora_up.weight": ("B", ".weight"),
    }
    for suffix, (role, replacement) in suffix_map.items():
        if key.endswith(suffix):
            return key[: -len(suffix)] + replacement, role
    return "", ""


def _collect_lora_pairs(lora_state: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
    pairs: Dict[str, Dict[str, torch.Tensor]] = {}
    for key, tensor in lora_state.items():
        base_key, role = _base_key_from_lora(key)
        if not base_key:
            continue
        entry = pairs.setdefault(base_key, {})
        if role in ("A", "B"):
            entry[role] = tensor
    return pairs


def _apply_lora_to_weight(
    weight: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    alpha: float,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    if weight.ndim < 2:
        raise ValueError(f"Expected weight to have >=2 dims, got {weight.shape}")
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"Expected 2D LoRA matrices, got A{A.shape} and B{B.shape}")

    out_features = weight.shape[0]
    in_features = int(torch.prod(torch.tensor(weight.shape[1:])).item())

    if A.shape[1] == in_features:
        A_mat = A
    elif A.shape[0] == in_features:
        print(f"[LoRA] Transposing A from {A.shape} to ({A.shape[1]}, {A.shape[0]})")
        A_mat = A.t()
    else:
        raise ValueError(f"LoRA A shape mismatch: got {A.shape}, expected (*, {in_features})")
    rank = A_mat.shape[0]

    if B.shape == (out_features, rank):
        B_mat = B
    elif B.shape == (rank, out_features):
        print(f"[LoRA] Transposing B from {B.shape} to ({B.shape[1]}, {B.shape[0]})")
        B_mat = B.t()
    else:
        raise ValueError(f"LoRA B shape mismatch: got {B.shape}, expected ({out_features}, {rank})")

    scaling = torch.tensor(alpha / rank if rank > 0 else 1.0, dtype=compute_dtype)
    delta = torch.matmul(B_mat.to(compute_dtype), A_mat.to(compute_dtype)) * scaling
    if delta.shape != (out_features, in_features):
        raise ValueError(
            f"LoRA delta shape mismatch: got {delta.shape}, expected ({out_features}, {in_features})"
        )
    delta = delta.to(weight.dtype)
    if weight.ndim > 2:
        flat = weight.view(weight.shape[0], -1)
        flat.add_(delta.view(flat.shape))
        return flat.view_as(weight)
    return weight + delta


def _copy_non_weight_files(src_dir: Path, dst_dir: Path) -> None:
    for item in src_dir.iterdir():
        if item.name.startswith("diffusion_pytorch_model-") and item.suffix == ".safetensors":
            continue
        if item.name == WEIGHT_SINGLE_NAME:
            continue
        if item.name == WEIGHT_INDEX_NAME:
            continue
        if item.is_dir():
            shutil.copytree(item, dst_dir / item.name, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dst_dir / item.name)

def _summarize_missing_keys(missing_keys: set[str]) -> None:
    if not missing_keys:
        print("[LoRA] All LoRA targets matched base weights.")
        return
    prefixes: Dict[str, int] = {}
    for key in missing_keys:
        prefix = key.rsplit(".", 1)[0] if "." in key else key
        prefixes[prefix] = prefixes.get(prefix, 0) + 1
    top = sorted(prefixes.items(), key=lambda kv: kv[1], reverse=True)[:30]
    print(f"[LoRA] Missing targets: {len(missing_keys)}")
    print("[LoRA] Top missing prefixes:")
    for prefix, count in top:
        print(f"  {prefix}: {count}")


def _merge_sharded_weights(
    base_dir: Path,
    lora_pairs: Dict[str, Dict[str, torch.Tensor]],
    output_dir: Path,
    alpha_override: float | None,
    compute_dtype: torch.dtype,
) -> None:
    index_path = base_dir / WEIGHT_INDEX_NAME
    if not index_path.exists():
        raise FileNotFoundError(f"Missing index file: {index_path}")
    index = json.loads(index_path.read_text())
    weight_map = index["weight_map"]

    # Group keys by shard file
    shard_to_keys: Dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        shard_to_keys.setdefault(shard, []).append(key)

    output_dir.mkdir(parents=True, exist_ok=True)
    _copy_non_weight_files(base_dir, output_dir)

    total_merged = 0
    matched_keys: set[str] = set()
    missing_keys: set[str] = set(lora_pairs.keys())
    for shard_name, keys in shard_to_keys.items():
        shard_path = base_dir / shard_name
        shard_state = load_file(str(shard_path))
        updated = {}
        for key in keys:
            weight = shard_state[key]
            norm_key = _normalize_key(key)
            if norm_key in lora_pairs:
                pair = lora_pairs[norm_key]
                if "A" in pair and "B" in pair:
                    alpha = alpha_override if alpha_override is not None else float(pair["A"].shape[0])
                    weight = _apply_lora_to_weight(weight, pair["A"], pair["B"], alpha, compute_dtype)
                    total_merged += 1
                matched_keys.add(norm_key)
                missing_keys.discard(norm_key)
            updated[key] = weight
        save_file(updated, str(output_dir / shard_name))

    (output_dir / WEIGHT_INDEX_NAME).write_text(json.dumps(index, indent=2))
    print(f"Merged LoRA into {total_merged} tensors across {len(shard_to_keys)} shards.")
    print(f"[LoRA] Matched targets: {len(matched_keys)}")
    _summarize_missing_keys(missing_keys)


def _merge_single_weights(
    base_dir: Path,
    lora_pairs: Dict[str, Dict[str, torch.Tensor]],
    output_dir: Path,
    alpha_override: float | None,
    compute_dtype: torch.dtype,
) -> None:
    weight_path = base_dir / WEIGHT_SINGLE_NAME
    if not weight_path.exists():
        raise FileNotFoundError(f"Missing weights: {weight_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    _copy_non_weight_files(base_dir, output_dir)

    state = load_file(str(weight_path))
    updated = {}
    total_merged = 0
    matched_keys: set[str] = set()
    missing_keys: set[str] = set(lora_pairs.keys())
    for key, weight in state.items():
        norm_key = _normalize_key(key)
        if norm_key in lora_pairs:
            pair = lora_pairs[norm_key]
            if "A" in pair and "B" in pair:
                alpha = alpha_override if alpha_override is not None else float(pair["A"].shape[0])
                weight = _apply_lora_to_weight(weight, pair["A"], pair["B"], alpha, compute_dtype)
                total_merged += 1
            matched_keys.add(norm_key)
            missing_keys.discard(norm_key)
        updated[key] = weight
    save_file(updated, str(output_dir / WEIGHT_SINGLE_NAME))
    print(f"Merged LoRA into {total_merged} tensors (single-file weights).")
    print(f"[LoRA] Matched targets: {len(matched_keys)}")
    _summarize_missing_keys(missing_keys)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge a LoRA into Wan diffusion weights.")
    parser.add_argument("--base_model_dir", required=True, help="Path to the base model directory.")
    parser.add_argument("--lora_path", required=True, help="Path to the LoRA checkpoint (.safetensors or .pt).")
    parser.add_argument("--output_dir", required=True, help="Directory to write merged weights.")
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=None,
        help="Override LoRA alpha. If not set, uses rank as alpha.",
    )
    parser.add_argument(
        "--compute_dtype",
        choices=("bf16", "fp16", "fp32"),
        default="fp32",
        help="Dtype used when computing the LoRA delta.",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_model_dir)
    lora_path = Path(args.lora_path)
    output_dir = Path(args.output_dir)

    lora_state = _load_lora_state(lora_path)
    lora_pairs = _collect_lora_pairs(lora_state)
    if not lora_pairs:
        raise ValueError(f"No LoRA tensors found in {lora_path}")
    total_pairs = len(lora_pairs)
    complete_pairs = sum(1 for pair in lora_pairs.values() if "A" in pair and "B" in pair)
    print(f"Found {total_pairs} LoRA targets, {complete_pairs} with both A and B.")

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    compute_dtype = dtype_map[args.compute_dtype]

    if (base_dir / WEIGHT_INDEX_NAME).exists():
        _merge_sharded_weights(base_dir, lora_pairs, output_dir, args.lora_alpha, compute_dtype)
    else:
        _merge_single_weights(base_dir, lora_pairs, output_dir, args.lora_alpha, compute_dtype)


if __name__ == "__main__":
    main()
