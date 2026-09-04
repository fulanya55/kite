<p align="center">
  <h1 align="center">KITE [ICRA 2026]</h1>
  <h3 align="center">Keyframe-Indexed Tokenized Evidence for VLM-Based Robot Failure Analysis</h3>
 <p align="center">
    <a href="https://m80hz.github.io/" target="_blank">Mehdi Hosseinzadeh</a>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <a href="#">King Hang Wong</a>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <a href="https://ferasdayoub.com/" target="_blank">Feras Dayoub</a>
  </p>
 
 <h5 align="center">The Australian Institute for Machine Learning (AIML), Adelaide University, Australia</h5>

  <p align="center">
    <a href="https://m80hz.github.io/kite/" target="_blank">
      <img src="https://img.shields.io/badge/🌐_Project_Page-007ACC?style=for-the-badge" alt="Project Page" />
    </a>
    &nbsp;
    <a href="https://arxiv.org/abs/2604.07034" target="_blank">
      <img src="https://img.shields.io/badge/📄_Paper-B31B1B?style=for-the-badge" alt="arXiv" />
    </a>
    &nbsp;
    <a href="https://github.com/m80hz/kite" target="_blank">
      <img src="https://img.shields.io/badge/💻_Code-181717?style=for-the-badge&amp;logo=github" alt="GitHub" />
    </a>
    &nbsp;
    <a href="#demo">
      <img src="https://img.shields.io/badge/🎮_Demo-8A2BE2?style=for-the-badge" alt="Demo" />
    </a>
    &nbsp;
    <a href="https://huggingface.co/m80hz/KITE-7B-Instruct" target="_blank">
      <img src="https://img.shields.io/badge/🤗_Model-FFD21E?style=for-the-badge" alt="HuggingFace" />
    </a>
  </p>
</p>

<p align="center">
  <img src="docs/kite_teaser.png" alt="KITE Overview" width="100%">
</p>

---


> **🚧 ToDo**
>
> - [x] QLoRA fine-tuning scripts & training recipes
> - [x] [Pre-trained weights](https://huggingface.co/m80hz/KITE-7B-Instruct)
>

## Quick Start

1. Clone with submodules (GroundingDINO, Depth-Anything-V2):
```bash
git clone --recursive https://github.com/m80hz/kite.git
cd kite
```
If you already cloned without `--recursive`, run: `git submodule update --init`

2. Install dependencies with `uv` (the lock file records the CUDA 12.8 PyTorch
   build used for the reproduction):
```bash
uv venv --python 3.12
uv sync --extra download --extra serve
source .venv/bin/activate
```

3. Download the complete RoboFAC data and both model snapshots.  This places
   all data under `data/` and the un-tuned Qwen and released KITE checkpoint
   under `model/`:
```bash
uv run python scripts/download_assets.py
```
The archive contains 64,691 training QA records, 40 simulation test splits,
real-world test splits, and the complete simulation/real-world videos.  The
download is resumable; set `HF_ENDPOINT=https://huggingface.co` when the
official Hugging Face endpoint is reachable.

4. Start a VLM server. A convenience launcher for [vLLM](https://github.com/vllm-project/vllm) is included:
```bash
# Pass an HF repo id or a local model directory
bash scripts/run_vllm.sh Qwen/Qwen2.5-VL-7B-Instruct
# Or use our fine-tuned model:
bash scripts/run_vllm.sh m80hz/KITE-7B-Instruct
```
See `scripts/run_vllm.sh` for optional env vars (`TP`, `PORT`, `GPU_MEM_UTIL`, etc.).

5. Ensure you have the **dataset_folder** and **test_file** JSON (the download
   script creates these under `data/robofac`).

> **Note on RoboFAC annotations:** The original RoboFAC dataset has some mismatched / incorrect file paths in certain annotation files. See [MINT-SJTU/RoboFAC#2](https://github.com/MINT-SJTU/RoboFAC/issues/2) for details and how to fix them before running evaluation.

6. Run evaluation (full pipeline with keyframes, 2D/3D scene context, BEV, and narrative):
```bash
uv run python -m kite.cli \
  --dataset_folder data/robofac \
  --test_file data/robofac/test_qa_sim/annos_per_video_split0.json \
  --model_name model/KITE-7B-Instruct \
  --model_url http://127.0.0.1:8000/v1 \
  --ovd_backend stub \
  --robot_profile examples/robot_profiles/sim_single_arm.json \
  --disable_tatc \
  --out_dir outputs/kite_run
```

Outputs:
- Per-task JSONs under `<out_dir>/` and a consolidated `stats_data.json` with MCQ scores and descriptive metrics.
- Storyboard images: `*_storyboard_all_keyframes.jpg` and, when enabled, `*_storyboard_bev.jpg` for BEV alignment.
- Optional final narrative text per video: `*_final_narrative.txt`.

### LoRA fine-tuning and comparison

The training entry point consumes the official `training_qa.json` format and
uses QLoRA on the un-tuned Qwen checkpoint.  A one-step smoke test is useful
for checking the installation; omit the limits for a full run:

```bash
# smoke test (writes a real PEFT adapter checkpoint)
uv run python training/train_lora.py --base-model model/Qwen2.5-VL-7B-Instruct \
  --max-samples 1 --max-steps 1 --output-dir model/kite-lora-smoke

# full training recipe
uv run python training/train_lora.py --base-model model/Qwen2.5-VL-7B-Instruct \
  --output-dir model/kite-lora
```

For a non-interactive reproduction of every simulation and real-world split,
use `scripts/run_reproduction_eval.sh` after starting the VLM server.  It writes
one output directory per split and can merge them into weighted summaries:

```bash
MODEL_NAME=model/KITE-7B-Instruct MODEL_URL=http://127.0.0.1:8000/v1 \
  bash scripts/run_reproduction_eval.sh
uv run python tools/merge_split_stats.py --results-dir outputs/kite-full \
  --output outputs/kite-full/results_merged.json
uv run python tools/consolidate_results.py --input outputs/kite-full/results_merged.json \
  --output outputs/kite-full/consolidated.json
```

On multi-GPU machines, `scripts/run_parallel_eval.sh` starts one local server
per GPU and distributes all 47 splits automatically:

```bash
MODEL_NAME=model/KITE-7B-Instruct OUT_ROOT=outputs/kite-full-baseline-v2 \
  uv run bash scripts/run_parallel_eval.sh
```

The launcher defaults to `OVD_BACKEND=stub` and disables TATC so it can run
with only the downloaded VLM weights (the measured split-0 runs use these
settings).  For the full perception stack, provide the GroundingDINO weights
and set `OVD_BACKEND=groundingdino ENABLE_3D_GRAPH=1`.

When vLLM is unavailable, the repository includes a lightweight Transformers
server (single GPU) that supports the same API:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/transformers_server.py \
  --model model/KITE-7B-Instruct --port 8000
```

To serve a trained adapter, point the server at the base Qwen snapshot and
attach the PEFT directory:

```bash
CUDA_VISIBLE_DEVICES=1 uv run python scripts/transformers_server.py \
  --model model/Qwen2.5-VL-7B-Instruct --lora-adapter model/kite-lora \
  --port 8001
```

Evaluate the released and LoRA models with the same command and retain the
result directories for a direct comparison (for example, `outputs/kite` and
`outputs/kite-lora`).  `tools/merge_split_stats.py` first merges split-level
`stats_data.json` files into one `results_merged.json`; then
`tools/consolidate_results.py` computes weighted question-type metrics.

For a quick numeric comparison of two runs (including the per-split
`stats_data.json` format), use:

```bash
uv run python tools/compare_results.py \
  --base outputs/kite-baseline-sim-split0/stats_data.json \
  --lora outputs/kite-lora-sim-split0/stats_data.json \
  --output outputs/kite-comparison-sim-split0.json
```

## Demo

Launch the interactive Gradio app to explore the full pipeline on a single video:

```
python app.py
```

The demo lets you step through keyframes, view 2D detections, per-keyframe BEV maps, colored depth, an interactive Plotly 3D point-cloud viewer, and run QA queries against the VLM. A few example video sequences are included under `examples/` to get started quickly.


## Notes

- Training-free by default; depth/3D are optional but recommended for BEV and spatial grounding.
- Any model exposing an OpenAI-compatible **/chat/completions** endpoint works out of the box.

## Evaluation

We score MCQs with normalized string containment and compute non-LLM text similarity metrics for descriptive QAs. See `docs/Evaluation.md` for details.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{hosseinzadeh2025kite,
  title     = {KITE: Keyframe-Indexed Tokenized Evidence for VLM-Based Robot Failure Analysis},
  author    = {Hosseinzadeh, Mehdi and Wong, King Hang and Dayoub, Feras},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2026}
}
```
