#!/usr/bin/env bash
set -euo pipefail

# Wait for a shared machine's GPUs to become idle, then run both complete
# RoboFAC evaluations. This intentionally never kills another user's process.
GPU_THRESHOLD_MIB="${GPU_THRESHOLD_MIB:-2000}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
LOG="${LOG:-outputs/full-reproduction-queue.log}"
mkdir -p "$(dirname "${LOG}")"
exec > >(tee -a "${LOG}") 2>&1

echo "[$(date -Is)] Waiting for GPUs ${GPUS} (threshold ${GPU_THRESHOLD_MIB} MiB each)"
IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
while true; do
  busy=0
  for gpu in "${GPU_ARR[@]}"; do
    used="$(nvidia-smi -i "${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -dc '0-9')"
    [[ -n "${used}" ]] || { busy=1; break; }
    if (( used > GPU_THRESHOLD_MIB )); then busy=1; break; fi
  done
  if (( busy == 0 )); then break; fi
  sleep 60
done

run_eval() {
  local label="$1"
  local model="$2"
  local adapter="$3"
  local out="$4"
  echo "[$(date -Is)] Starting ${label}"
  MODEL_NAME="${model}" LORA_ADAPTER="${adapter}" OUT_ROOT="${out}" \
    GPUS="${GPUS}" HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
    OVD_BACKEND="${OVD_BACKEND:-stub}" \
    uv run bash scripts/run_parallel_eval.sh
  local count
  count="$(find "${out}" -name stats_data.json -type f | wc -l)"
  [[ "${count}" -eq 47 ]] || { echo "${label}: expected 47 stats files, got ${count}" >&2; return 1; }
  uv run python tools/merge_split_stats.py --results-dir "${out}" \
    --output "${out}/results_merged.json"
  uv run python tools/consolidate_results.py --input "${out}/results_merged.json" \
    --output "${out}/consolidated.json"
}

run_eval baseline model/KITE-7B-Instruct "" outputs/kite-full-baseline
run_eval lora model/Qwen2.5-VL-7B-Instruct model/kite-lora outputs/kite-full-lora
uv run python tools/compare_results.py \
  --base outputs/kite-full-baseline/results_merged.json \
  --lora outputs/kite-full-lora/results_merged.json \
  --output outputs/kite-full-comparison.json
echo "[$(date -Is)] Complete full reproduction"
