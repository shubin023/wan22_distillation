#!/usr/bin/env bash
set -euo pipefail

# Low-noise DMD distillation for Wan2.2-I2V-A14B. Requires high-noise safetensors in configs/wan22_low_i2v.yaml.

NNODES="${NNODES:-8}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
RDZV_ID="${RDZV_ID:-5236}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29501}"
LOGDIR="${LOGDIR:-logs/wan22_low_i2v}"

cd "$(dirname "$0")"

torchrun \
  --nnodes="${NNODES}" --nproc_per_node="${GPUS_PER_NODE}" \
  --rdzv_id="${RDZV_ID}" --rdzv_backend=c10d --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  train.py \
  --config_path configs/wan22_low_i2v.yaml \
  --logdir "${LOGDIR}" \
  --no_visualize \
  --disable-mlflow
