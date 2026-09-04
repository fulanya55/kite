#!/usr/bin/env python3
"""Download the exact RoboFAC assets used by the KITE reproduction.

Examples::

    uv run python scripts/download_assets.py
    uv run python scripts/download_assets.py --skip-data --models-only

The official dataset is mirrored on ModelScope.  Hugging Face downloads use
``HF_ENDPOINT`` (defaulting to ``https://hf-mirror.com`` in this environment),
which can be overridden for an unrestricted network with the official endpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def download_dataset(root: Path) -> None:
    from modelscope.hub.snapshot_download import dataset_snapshot_download

    root.mkdir(parents=True, exist_ok=True)
    dataset_snapshot_download(
        "MinghaoYE/RoboFAC-dataset",
        revision="master",
        local_dir=str(root),
        max_workers=16,
    )
    archive = root / "simulation_data.zip"
    extracted = root / "simulation_data"
    if archive.exists() and not extracted.exists():
        print(f"Extracting {archive} -> {extracted}")
        shutil.unpack_archive(str(archive), str(root))
        # Some archive revisions contain a top-level simulation_data directory;
        # others contain task directories directly.  Normalize the latter.
        if not extracted.exists():
            extracted.mkdir()
            for child in list(root.iterdir()):
                if child.name in {"simulation_data", archive.name} or child.name.startswith("test_qa"):
                    continue
                if child.is_dir() and child.name not in {"realworld_data"}:
                    child.rename(extracted / child.name)
    files = [p for p in root.rglob("*") if p.is_file() and ".cache" not in p.parts]
    manifest = {
        "source": "modelscope://MinghaoYE/RoboFAC-dataset",
        "revision": "master",
        "file_count": len(files),
        "total_bytes": sum(p.stat().st_size for p in files),
        "files": [{"path": str(p.relative_to(root)), "bytes": p.stat().st_size} for p in sorted(files)],
    }
    (root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(f"Dataset ready: {root} ({manifest['file_count']} files, {manifest['total_bytes']:,} bytes)")


def download_model(repo_id: str, target: Path, endpoint: str) -> None:
    from huggingface_hub import snapshot_download

    target.mkdir(parents=True, exist_ok=True)
    # Qwen's snapshot uses chat_template.json while KITE uses the equivalent
    # chat_template.jinja; likewise tokenizer auxiliary files differ slightly.
    # Treat these as alternatives so a completed snapshot is not re-requested
    # on every invocation.
    required = [
        "config.json", "generation_config.json", "model.safetensors.index.json",
        "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json",
        "vocab.json", "merges.txt",
    ]
    alternatives = [("chat_template.jinja", "chat_template.json")]
    missing = [name for name in required if not (target / name).exists()]
    for group in alternatives:
        if not any((target / name).exists() for name in group):
            missing.extend(group)
    if missing:
        # Keep metadata requests serial: hf-mirror may return HTTP 429 when a
        # large multi-shard download opens many HEAD requests simultaneously.
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            revision="main",
            local_dir=str(target),
            endpoint=endpoint,
            allow_patterns=missing,
            max_workers=1,
        )
    print(f"Model ready: {repo_id} -> {target}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/robofac"))
    p.add_argument("--model-dir", type=Path, default=Path("model"))
    p.add_argument("--hf-endpoint", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    p.add_argument("--skip-data", action="store_true")
    p.add_argument("--skip-models", action="store_true")
    p.add_argument("--models-only", action="store_true")
    args = p.parse_args()
    if not args.skip_data and not args.models_only:
        download_dataset(args.data_dir)
    if not args.skip_models:
        download_model("Qwen/Qwen2.5-VL-7B-Instruct", args.model_dir / "Qwen2.5-VL-7B-Instruct", args.hf_endpoint)
        download_model("m80hz/KITE-7B-Instruct", args.model_dir / "KITE-7B-Instruct", args.hf_endpoint)


if __name__ == "__main__":
    main()
