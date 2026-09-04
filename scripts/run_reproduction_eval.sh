#!/usr/bin/env bash
set -euo pipefail

# Run every RoboFAC split with the same settings used for the reproduced
# baseline/LoRA measurements.  Start an OpenAI-compatible VLM server first
# (scripts/transformers_server.py or scripts/run_vllm.sh).
DATA_ROOT="${DATA_ROOT:-data/robofac}"
MODEL_NAME="${MODEL_NAME:-model/KITE-7B-Instruct}"
MODEL_URL="${MODEL_URL:-http://127.0.0.1:8000/v1}"
OUT_ROOT="${OUT_ROOT:-outputs/kite-full}"
ROBOT_PROFILE="${ROBOT_PROFILE:-examples/robot_profiles/sim_single_arm.json}"
OVD_BACKEND="${OVD_BACKEND:-stub}"
ENABLE_3D_GRAPH="${ENABLE_3D_GRAPH:-0}"

extra=()
if [[ "${ENABLE_3D_GRAPH}" == "1" ]]; then
  extra+=(--enable_3d_graph)
fi

run_split_dir() {
  local split_dir="$1"
  local out_dir="$2"
  mkdir -p "${out_dir}"
  while IFS= read -r test_file; do
    local name
    name="$(basename "${test_file%.json}")"
    echo "[KITE] ${split_dir}/${name}"
    KITE_KEYFRAME_STRATEGY="${KITE_KEYFRAME_STRATEGY:-uniform}" \
      uv run python -m kite.cli \
      --dataset_folder "${DATA_ROOT}" \
      --test_file "${test_file}" \
      --model_name "${MODEL_NAME}" \
      --model_url "${MODEL_URL}" \
      --ovd_backend "${OVD_BACKEND}" \
      --robot_profile "${ROBOT_PROFILE}" \
      --disable_tatc \
      "${extra[@]}" \
      --out_dir "${out_dir}/${name}"
  done < <(find "${DATA_ROOT}/${split_dir}" -maxdepth 1 -type f -name '*.json' | sort -V)
}

run_split_dir test_qa_sim "${OUT_ROOT}/sim"
run_split_dir test_qa_realworld "${OUT_ROOT}/realworld"

echo "Completed all RoboFAC splits. Consolidate with:"
echo "  uv run python tools/merge_split_stats.py --results-dir ${OUT_ROOT} --output ${OUT_ROOT}/results_merged.json"
echo "  uv run python tools/consolidate_results.py --input ${OUT_ROOT}/results_merged.json --output ${OUT_ROOT}/consolidated.json"
