#!/usr/bin/env python3
"""Compare two KITE ``results_merged.json`` files and write metric deltas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _flatten(value: Any, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            out.update(_flatten(child, f"{prefix}.{key}" if prefix else key))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        out[prefix] = float(value)
    return out


def _as_mapping(value: Any) -> dict[str, Any]:
    """Accept both consolidated dictionaries and per-run stats_data lists."""
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        mapped: dict[str, Any] = {}
        for idx, row in enumerate(value):
            if not isinstance(row, dict):
                continue
            task = row.get("task") or row.get("question_type") or f"row_{idx}"
            question_type = row.get("question_type")
            key = f"{task}/{question_type}" if question_type and question_type != task else str(task)
            mapped[key] = row
        return mapped
    raise TypeError(f"Expected a JSON object or list, got {type(value).__name__}")


def compare(base: Any, tuned: Any) -> dict[str, Any]:
    base, tuned = _as_mapping(base), _as_mapping(tuned)
    rows: dict[str, Any] = {}
    for key in sorted(set(base) | set(tuned)):
        b, t = _flatten(base.get(key, {})), _flatten(tuned.get(key, {}))
        metrics: dict[str, Any] = {}
        for metric in sorted(set(b) | set(t)):
            bv, tv = b.get(metric), t.get(metric)
            metrics[metric] = {"base": bv, "lora": tv, "delta": None if bv is None or tv is None else tv - bv}
        rows[key] = metrics
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--lora", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("outputs/comparison.json"))
    args = p.parse_args()
    result = compare(json.loads(args.base.read_text()), json.loads(args.lora.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
