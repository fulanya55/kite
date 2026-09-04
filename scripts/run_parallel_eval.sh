#!/usr/bin/env bash
set -euo pipefail

# Parallel full-dataset evaluator. One Transformers server and one evaluator
# worker are assigned to each listed GPU; every RoboFAC split is processed.
# Defaults match the deterministic VLM-only reproduction configuration.
DATA_ROOT="${DATA_ROOT:-data/robofac}"
MODEL_NAME="${MODEL_NAME:-model/KITE-7B-Instruct}"
LORA_ADAPTER="${LORA_ADAPTER:-}"
OUT_ROOT="${OUT_ROOT:-outputs/kite-full}"
GPUS_CSV="${GPUS:-0,1,2,3,4,5,6,7}"
PORT_BASE="${PORT_BASE:-8100}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
OVD_BACKEND="${OVD_BACKEND:-stub}"
ROBOT_PROFILE="${ROBOT_PROFILE:-examples/robot_profiles/sim_single_arm.json}"
HF_ENDPOINT_VALUE="${HF_ENDPOINT:-https://hf-mirror.com}"

IFS=',' read -r -a GPUS_ARR <<< "${GPUS_CSV}"
NUM_WORKERS="${#GPUS_ARR[@]}"
if (( NUM_WORKERS == 0 )); then
  echo "GPUS must contain at least one GPU id" >&2
  exit 2
fi

mapfile -t TEST_FILES < <(
  find "${DATA_ROOT}/test_qa_sim" "${DATA_ROOT}/test_qa_realworld" \
    -maxdepth 1 -type f -name '*.json' | sort -V
)
if (( ${#TEST_FILES[@]} == 0 )); then
  echo "No RoboFAC test split JSON files found under ${DATA_ROOT}" >&2
  exit 2
fi

mkdir -p "${OUT_ROOT}/logs"
declare -a SERVER_PIDS=()
cleanup() {
  for pid in "${SERVER_PIDS[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Starting ${NUM_WORKERS} VLM servers for ${#TEST_FILES[@]} splits"
for wi in "${!GPUS_ARR[@]}"; do
  gpu="${GPUS_ARR[$wi]}"
  port=$((PORT_BASE + wi))
  server_log="${OUT_ROOT}/logs/server-gpu${gpu}.log"
  extra=()
  if [[ -n "${LORA_ADAPTER}" ]]; then
    extra+=(--lora-adapter "${LORA_ADAPTER}")
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" uv run python scripts/transformers_server.py \
    --model "${MODEL_NAME}" --port "${port}" --max-new-tokens "${MAX_NEW_TOKENS}" \
    "${extra[@]}" >"${server_log}" 2>&1 &
  SERVER_PIDS+=("$!")
done

for wi in "${!GPUS_ARR[@]}"; do
  port=$((PORT_BASE + wi))
  for attempt in {1..120}; do
    if curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      break
    fi
    if (( attempt == 120 )); then
      echo "VLM server on port ${port} did not become ready" >&2
      exit 1
    fi
    sleep 1
  done
done

run_worker() {
  local wi="$1"
  local gpu="${GPUS_ARR[$wi]}"
  local port=$((PORT_BASE + wi))
  local worker_log="${OUT_ROOT}/logs/worker-gpu${gpu}.log"
  : >"${worker_log}"
  for ((j=wi; j<${#TEST_FILES[@]}; j+=NUM_WORKERS)); do
    local test_file="${TEST_FILES[$j]}"
    local split_dir
    split_dir="$(basename "$(dirname "${test_file}")")"
    local name
    name="$(basename "${test_file%.json}")"
    local out_dir="${OUT_ROOT}/${split_dir}/${name}"
    echo "[gpu=${gpu}] ${test_file} -> ${out_dir}" | tee -a "${worker_log}"
    CUDA_VISIBLE_DEVICES="${gpu}" HF_ENDPOINT="${HF_ENDPOINT_VALUE}" KITE_KEYFRAME_STRATEGY="${KITE_KEYFRAME_STRATEGY:-uniform}" \
      uv run python -m kite.cli \
      --dataset_folder "${DATA_ROOT}" --test_file "${test_file}" \
      --model_name "${MODEL_NAME}" --model_url "http://127.0.0.1:${port}/v1" \
      --ovd_backend "${OVD_BACKEND}" --robot_profile "${ROBOT_PROFILE}" \
      --disable_tatc --out_dir "${out_dir}" >>"${worker_log}" 2>&1
  done
}

declare -a WORKER_PIDS=()
for wi in "${!GPUS_ARR[@]}"; do
  run_worker "${wi}" &
  WORKER_PIDS+=("$!")
done
status=0
for pid in "${WORKER_PIDS[@]}"; do
  wait "${pid}" || status=1
done
if (( status != 0 )); then
  echo "At least one evaluator worker failed; inspect ${OUT_ROOT}/logs" >&2
  exit "${status}"
fi
echo "Completed all ${#TEST_FILES[@]} RoboFAC splits under ${OUT_ROOT}"
