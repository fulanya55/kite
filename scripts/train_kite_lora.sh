#!/usr/bin/env bash
set -euo pipefail

# Full paper-aligned QLoRA run.  Set CUDA_VISIBLE_DEVICES to select one GPU.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  uv run python training/train_lora.py \
  --config training/kite_lora_full.yaml "$@"
