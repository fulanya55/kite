#!/usr/bin/env python3
"""Merge split-level ``stats_data.json`` files into ``results_merged.json``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    merged: dict[str, Any] = {}
    files = sorted(args.results_dir.rglob("stats_data.json"))
    if not files:
        raise SystemExit(f"No stats_data.json files under {args.results_dir}")
    for path in files:
        loaded = json.loads(path.read_text())
        rows = loaded if isinstance(loaded, list) else [
            dict(v, **{"_key": k}) for k, v in loaded.items() if isinstance(v, dict)
        ]
        split = path.parent.name
        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            task = row.get("task", f"row_{idx}")
            qtype = row.get("question_type", "unknown")
            # Keep every split distinct while preserving the task/qtype shape
            # expected by tools/consolidate_results.py.
            merged[f"{task}__{split}/{qtype}"] = row
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, ensure_ascii=False, indent=2))
    print(f"Merged {len(files)} files ({len(merged)} rows) -> {args.output}")


if __name__ == "__main__":
    main()
