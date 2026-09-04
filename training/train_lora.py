"""QLoRA fine-tuning for the KITE/RoboFAC video-QA format.

The released ``training_qa.json`` contains records of the form::

    {"video": "Task/view/id.mp4", "conversations": [
        {"from": "human", "value": "<video>\\nQuestion"},
        {"from": "assistant", "value": "Answer"}]}

This script intentionally keeps the data path and training knobs explicit so a
run can be reproduced from a shell command.  It uses Transformers' native
Qwen2.5-VL video processor and PEFT LoRA; with ``--max-samples``/``--max-steps``
it is also suitable for a smoke test on one GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
    set_seed,
)


def _load_records(path: Path, data_root: Path, max_samples: int | None) -> List[Dict[str, Any]]:
    records = json.loads(path.read_text())
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON list in {path}")
    valid: List[Dict[str, Any]] = []
    basename_index: Dict[str, Path] = {}
    for video_file in data_root.rglob("*.mp4"):
        basename_index.setdefault(video_file.name, video_file)
    for record in records:
        rel = str(record.get("video", ""))
        # The archive is extracted as data_root/simulation_data/...; accepting
        # both layouts makes the command work before/after extraction.
        candidates = [data_root / rel, data_root / "simulation_data" / rel, data_root / "realworld_data" / rel]
        if rel.startswith("dataset_success_cleaned/"):
            candidates.append(data_root / "simulation_data" / "success_data" / rel.split("/", 1)[1])
        if rel.startswith("dataset_failure_cleaned/"):
            candidates.append(data_root / "simulation_data" / "failure_data" / rel.split("/", 1)[1])
        video = next((p for p in candidates if p.is_file()), None)
        if video is None:
            video = basename_index.get(Path(rel).name)
        if video is None:
            continue
        conversations = record.get("conversations") or []
        human = next((x.get("value", "") for x in conversations if x.get("from") in ("human", "user")), "")
        answer = next((x.get("value", "") for x in conversations if x.get("from") in ("assistant", "gpt")), "")
        if not human or not answer:
            continue
        valid.append({"id": record.get("id"), "video": str(video), "question": human, "answer": str(answer)})
        if max_samples and len(valid) >= max_samples:
            break
    if not valid:
        raise FileNotFoundError(
            f"No usable records found. Check --data-root={data_root} and extract simulation_data.zip."
        )
    return valid


class VideoQADataset(torch.utils.data.Dataset):
    def __init__(self, records: Sequence[Dict[str, Any]], processor: Any, fps: float, max_pixels: int, num_frames: int):
        self.records = list(records)
        self.processor = processor
        self.fps = fps
        self.max_pixels = max_pixels
        self.num_frames = num_frames

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.records[index]
        question = row["question"].replace("<video>", "").strip()
        messages = [
            {
                "role": "user",
                "content": [{"type": "video", "video": row["video"]}, {"type": "text", "text": question}],
            },
            {"role": "assistant", "content": [{"type": "text", "text": row["answer"]}]},
        ]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        # A single-item batch avoids padding video tensors across samples.  The
        # collator below is correspondingly batch-size one by default.
        encoded = self.processor(
            text=[prompt],
            videos=[row["video"]],
            return_tensors="pt",
            padding=True,
            videos_kwargs={
                "do_sample_frames": True,
                **({"num_frames": self.num_frames} if self.num_frames > 0 else {"fps": self.fps}),
                "size": {"shortest_edge": int(self.max_pixels**0.5), "longest_edge": int(self.max_pixels**0.5)},
            },
        )
        return {
            k: v.squeeze(0)
            if torch.is_tensor(v) and v.ndim > 0 and k in {"input_ids", "attention_mask"}
            else v
            for k, v in encoded.items()
        }


@dataclass
class SingleVideoCollator:
    pad_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(features) != 1:
            raise ValueError("VideoQADataset uses batch_size=1 because video grids have variable lengths")
        item = dict(features[0])
        ids = item["input_ids"]
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        item["input_ids"] = ids
        if "attention_mask" in item:
            mask = item["attention_mask"]
            item["attention_mask"] = mask.unsqueeze(0) if mask.ndim == 1 else mask
        labels = ids.clone()
        labels[labels == self.pad_token_id] = -100
        item["labels"] = labels
        # Processor returns grid metadata as [num_videos, 3].  Keep that shape;
        # only add a batch dimension for token tensors.
        return item


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QLoRA fine-tune Qwen2.5-VL/KITE on RoboFAC training_qa.json")
    p.add_argument("--base-model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--train-file", type=Path, default=Path("data/robofac/training_qa.json"))
    p.add_argument("--data-root", type=Path, default=Path("data/robofac"))
    p.add_argument("--output-dir", type=Path, default=Path("model/kite-lora"))
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--num-train-epochs", type=float, default=1.0)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--gradient-accumulation-steps", type=int, default=16)
    p.add_argument("--save-steps", type=int, default=250)
    p.add_argument("--logging-steps", type=int, default=1)
    p.add_argument("--fps", type=float, default=1.0)
    p.add_argument("--num-frames", type=int, default=8)
    p.add_argument("--max-pixels", type=int, default=3136, help="Video processor spatial pixel budget (56x56 by default)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-4bit", action="store_true", help="Use bf16 LoRA instead of 4-bit QLoRA")
    p.add_argument("--trust-remote-code", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    set_seed(args.seed)
    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = _load_records(args.train_file, args.data_root, args.max_samples)

    quantization_config = None
    model_kwargs: Dict[str, Any] = {"trust_remote_code": args.trust_remote_code, "device_map": "auto"}
    if not args.no_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["torch_dtype"] = torch.bfloat16
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.base_model, **model_kwargs)
    if quantization_config is not None:
        model = prepare_model_for_kbit_training(model)
    model.enable_input_require_grads()
    model.config.use_cache = False
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    dataset = VideoQADataset(records, processor, fps=args.fps, max_pixels=args.max_pixels, num_frames=args.num_frames)
    train_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        save_total_limit=2,
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=0,
        seed=args.seed,
    )
    trainer = Trainer(model=model, args=train_args, train_dataset=dataset, data_collator=SingleVideoCollator(processor.tokenizer.pad_token_id))
    result = trainer.train(resume_from_checkpoint=None)
    trainer.save_model(str(args.output_dir))
    processor.save_pretrained(str(args.output_dir))
    summary = {"base_model": args.base_model, "records": len(records), "output_dir": str(args.output_dir), "train_metrics": result.metrics}
    (args.output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
