# Wan2.2-I2V-A14B — DMD-Only Distillation

This is a trimmed copy of the Self-Forcing-Plus codebase focused on Distribution Matching Distillation (DMD) for Wan2.2-I2V-A14B. It keeps the components needed to train the 4-step I2V model (high-noise then low-noise) and run inference.

## Setup
```bash
cd wan22_i2v_dmd_only
conda create -n self_forcing python=3.10 -y
conda activate self_forcing
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
python setup.py develop
```

Download base weights:
```bash
huggingface-cli download Wan-AI/Wan2.2-I2V-A14B --local-dir wan_models/Wan2.2-I2V-A14B
```

## Data (I2V conditioning)
The training pipeline expects LMDB shards with:
- `latents`: VAE latents (shape 1×21×16×60×104); only the first-frame latent is used as conditioning.
- `img`: first-frame RGB (480×832×3 uint8).
- `prompts`: the prompt text.

You can build these from teacher-generated videos:
```bash
python scripts/compute_vae_latent.py --input_video_folder {videos} --output_latent_folder {latents} --model_name Wan2.1-T2V-14B --prompt_folder {prompts}
python scripts/create_lmdb_14b_shards.py --data_path {latents} --prompt_path {prompts} --video_path {videos} --lmdb_path {lmdb_out}
```
Point `data_path` in the configs to `{lmdb_out}`.

## Training (DMD only)
High-noise stage (uses configs/wan22_high_i2v.yaml):
```bash
torchrun --nnodes=8 --nproc_per_node=8 \
  --rdzv_id=5235 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  train.py \
  --config_path configs/wan22_high_i2v.yaml \
  --logdir logs/wan22_high_i2v \
  --no_visualize --disable-wandb
```

Convert the high-noise checkpoint for stage 2:
```bash
python convert_checkpoint.py \
  --input-checkpoint ./logs/wan22_high_i2v/checkpoint_model_001000/model.pt \
  --output-checkpoint Wan2.2-I2V-A14B/distill_models/high_noise_model/distill_model.safetensors \
  --to-bf16
```

Low-noise stage (uses configs/wan22_low_i2v.yaml, which references the safetensors above):
```bash
torchrun --nnodes=8 --nproc_per_node=8 \
  --rdzv_id=5235 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  train.py \
  --config_path configs/wan22_low_i2v.yaml \
  --logdir logs/wan22_low_i2v \
  --no_visualize --disable-wandb
```

Notes:
- These configs set `trainer: score_distillation` and `distribution_loss: dmd`. No GAN/SID paths are used.
- The reference setup uses 8×8 GPUs (64 total) with FSDP; single/dual GPU is not realistic for the 14B I2V model.

## Inference
Generate video from an input image:
```bash
python inference.py \
  --config_path configs/wan22_low_i2v.yaml \
  --checkpoint_path logs/wan22_low_i2v/checkpoint_model_XXXXXX/model.pt \
  --data_path {folder_with_images_and_prompts} \
  --output_folder outputs \
  --i2v \
  --num_output_frames 21 \
  --use_ema
```
You can change `--num_output_frames` (memory scales with length; training used 21).
