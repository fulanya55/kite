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
import math
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from threading import Lock
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import yaml
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


_FRAME_CACHE: "OrderedDict[tuple[str, int, int], Any]" = OrderedDict()
_FRAME_CACHE_LOCK = Lock()
# Keep enough entries for a complete micro-batch.  The grouped sampler emits
# split chunks of an oversized video consecutively, so this cache lets the
# second chunk reuse the decoded frames without reopening the MP4.
_FRAME_CACHE_LIMIT = 64


def _decode_sampled_frames(path: str, num_frames: int, image_size: int) -> Any:
    """Decode only the frames needed by one training example.

    Passing a filesystem path to Qwen's processor makes torchvision decode the
    video internally for every QA record.  We instead seek/sample frames once,
    resize them to the paper's 512px input, and pass an ndarray to the
    processor.  A small per-worker LRU cache also avoids decoding a video again
    when several QA questions reference the same episode.
    """
    import cv2
    import numpy as np
    cv2.setNumThreads(1)

    key = (os.path.abspath(path), int(num_frames), int(image_size))
    with _FRAME_CACHE_LOCK:
        cached = _FRAME_CACHE.get(key)
        if cached is not None:
            _FRAME_CACHE.move_to_end(key)
            return cached

    frames: List[Any] = []
    cap = cv2.VideoCapture(path)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if count > 0:
        indices = np.linspace(0, max(0, count - 1), max(1, int(num_frames))).round().astype(int)
        for frame_index in dict.fromkeys(indices.tolist()):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = cap.read()
            if ok and frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if image_size > 0 and (frame.shape[0] != image_size or frame.shape[1] != image_size):
                    frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
                frames.append(frame)
    cap.release()

    # OpenCV builds without AV1 support return no frames.  PyAV can decode
    # those files and is also restricted to the requested sample count.
    if not frames:
        try:
            import av

            container = av.open(path)
            stream = container.streams.video[0]
            wanted = set(np.linspace(0, max(0, int(stream.frames or 1) - 1), max(1, int(num_frames))).round().astype(int).tolist())
            for index, frame in enumerate(container.decode(stream)):
                if index in wanted:
                    image = frame.to_ndarray(format="rgb24")
                    if image_size > 0 and (image.shape[0] != image_size or image.shape[1] != image_size):
                        image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
                    frames.append(image)
                if len(frames) >= max(1, int(num_frames)):
                    break
            container.close()
        except Exception as exc:
            raise RuntimeError(f"Unable to decode training video {path}: {exc}") from exc
    if not frames:
        raise RuntimeError(f"Unable to decode training video {path}")

    # Some codecs cannot seek the final requested index exactly; pad by
    # repeating the last valid frame while preserving the requested frame count.
    while len(frames) < max(1, int(num_frames)):
        frames.append(frames[-1].copy())
    result = np.stack(frames[: max(1, int(num_frames))])
    with _FRAME_CACHE_LOCK:
        _FRAME_CACHE[key] = result
        _FRAME_CACHE.move_to_end(key)
        while len(_FRAME_CACHE) > _FRAME_CACHE_LIMIT:
            _FRAME_CACHE.popitem(last=False)
    return result


class VideoQADataset(torch.utils.data.Dataset):
    def __init__(self, records: Sequence[Dict[str, Any]], processor: Any, fps: float, max_pixels: int, num_frames: int):
        self.records = list(records)
        self.processor = processor
        self.fps = fps
        self.max_pixels = max_pixels
        self.num_frames = num_frames
        self.image_size = max(1, int(max_pixels**0.5))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        # Keep __getitem__ cheap.  The collator performs parallel video
        # decoding/encoding for the complete batch.
        return self.records[index]

    def encode_record(self, row: Dict[str, Any], sampled_frames: Any | None = None) -> Dict[str, torch.Tensor]:
        question = row["question"].replace("<video>", "").strip()
        if sampled_frames is None:
            sampled_frames = _decode_sampled_frames(row["video"], self.num_frames, self.image_size)
        messages = [
            {
                "role": "user",
                "content": [{"type": "video", "video": sampled_frames}, {"type": "text", "text": question}],
            },
            {"role": "assistant", "content": [{"type": "text", "text": row["answer"]}]},
        ]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        encoded = self.processor(
            text=[prompt],
            videos=[sampled_frames],
            return_tensors="pt",
            padding=True,
            videos_kwargs={
                "do_sample_frames": False,
                "size": {"shortest_edge": self.image_size, "longest_edge": self.image_size},
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
    dataset: VideoQADataset | None = None
    decode_workers: int = 16
    _executor: ThreadPoolExecutor | None = field(default=None, init=False, repr=False)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Dataset records are lightweight; encode the videos concurrently here
        # instead of serializing processor outputs through DataLoader workers.
        if features and "input_ids" not in features[0]:
            if self.dataset is None:
                raise ValueError("SingleVideoCollator needs a VideoQADataset for raw records")
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=max(1, int(self.decode_workers)))
            # Decode each unique video at most once per batch.  The old
            # implementation submitted every QA row independently, so rows
            # sharing an MP4 could race on the cache and open the file several
            # times.  Keeping the decoded frame array local to this batch also
            # makes the reuse explicit and independent of the process LRU.
            unique_rows: Dict[str, Dict[str, Any]] = {}
            for row in features:
                unique_rows.setdefault(row["video"], row)
            futures = {
                path: self._executor.submit(
                    _decode_sampled_frames,
                    row["video"],
                    self.dataset.num_frames,
                    self.dataset.image_size,
                )
                for path, row in unique_rows.items()
            }
            frame_map = {path: future.result() for path, future in futures.items()}
            encoded_futures = [
                self._executor.submit(self.dataset.encode_record, row, frame_map[row["video"]])
                for row in features
            ]
            features = [future.result() for future in encoded_futures]
        # Qwen's video patch tensors have variable first dimensions.  Token
        # tensors are padded, while visual patch tensors and their grid
        # metadata are concatenated across examples.
        max_len = max(int(f["input_ids"].numel()) for f in features)
        ids = torch.full((len(features), max_len), self.pad_token_id, dtype=torch.long)
        masks = torch.zeros((len(features), max_len), dtype=torch.long)
        for row, feature in enumerate(features):
            cur_ids = feature["input_ids"].reshape(-1)
            cur_mask = feature.get("attention_mask", torch.ones_like(cur_ids)).reshape(-1)
            ids[row, : cur_ids.numel()] = cur_ids
            masks[row, : cur_mask.numel()] = cur_mask
        batch: Dict[str, Any] = {"input_ids": ids, "attention_mask": masks}
        labels = ids.clone()
        labels[labels == self.pad_token_id] = -100
        batch["labels"] = labels

        concat_keys = {"pixel_values_videos", "video_grid_thw", "image_grid_thw", "second_per_grid_ts"}
        keys = set().union(*(f.keys() for f in features)) - {"input_ids", "attention_mask"}
        keys.discard("labels")
        for key in keys:
            values = [f[key] for f in features if key in f]
            if not values or not all(torch.is_tensor(v) for v in values):
                continue
            if key in concat_keys:
                batch[key] = torch.cat(values, dim=0)
            elif all(tuple(v.shape) == tuple(values[0].shape) for v in values):
                batch[key] = torch.stack(values, dim=0)
            else:
                # Preserve variable visual tensors rather than silently
                # padding them with a semantically incorrect shape.
                batch[key] = torch.cat(values, dim=0)
        return batch


def _load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text())
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"Training config must be a YAML mapping: {path}")
    return raw


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QLoRA fine-tune Qwen2.5-VL/KITE on RoboFAC training_qa.json")
    p.add_argument("--config", type=Path, default=Path("training/kite_lora_full.yaml"))
    # Defaults are None so a YAML value is used unless explicitly overridden.
    p.add_argument("--base-model")
    p.add_argument("--train-file", type=Path)
    p.add_argument("--data-root", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--max-samples", type=int)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--num-train-epochs", type=float)
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--warmup-ratio", type=float)
    p.add_argument("--optimizer")
    p.add_argument("--weight-decay", type=float)
    p.add_argument("--gradient-accumulation-steps", type=int)
    p.add_argument("--per-device-train-batch-size", type=str)
    p.add_argument("--target-vram-fraction", type=float)
    p.add_argument("--max-batch-size", type=int)
    p.add_argument("--lora-rank", type=int)
    p.add_argument("--lora-alpha", type=int)
    p.add_argument("--lora-dropout", type=float)
    p.add_argument("--save-steps", type=int)
    p.add_argument("--logging-steps", type=int)
    p.add_argument("--save-total-limit", type=int)
    p.add_argument("--dataloader-num-workers", type=int)
    p.add_argument("--decode-workers", type=int)
    p.add_argument("--videos-per-window", type=int)
    p.add_argument("--max-samples-per-video-per-batch", type=int)
    p.add_argument("--fps", type=float)
    p.add_argument("--num-frames", type=int)
    p.add_argument("--max-pixels", type=int, help="Video processor spatial pixel budget (512x512 is 262144)")
    p.add_argument("--seed", type=int)
    p.add_argument("--no-4bit", action="store_true", default=None, help="Use bf16 LoRA instead of 4-bit QLoRA")
    p.add_argument("--unfreeze-merger", action="store_true", default=None)
    p.add_argument("--trust-remote-code", action="store_true", default=None)
    return p.parse_args()


def _merge_config(args: argparse.Namespace) -> argparse.Namespace:
    config = _load_config(args.config)
    defaults: Dict[str, Any] = {
        "base_model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "train_file": Path("data/robofac/training_qa.json"),
        "data_root": Path("data/robofac"),
        "output_dir": Path("model/kite-lora"),
        "max_samples": None,
        "max_steps": -1,
        "num_train_epochs": 1.0,
        "learning_rate": 1e-5,
        "warmup_ratio": 0.03,
        "optimizer": "adamw_torch",
        "weight_decay": 0.01,
        "gradient_accumulation_steps": 16,
        "per_device_train_batch_size": "1",
        "target_vram_fraction": 0.80,
        "max_batch_size": 8,
        "lora_rank": 8,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "save_steps": 250,
        "logging_steps": 10,
        "save_total_limit": 2,
        "dataloader_num_workers": 0,
        "decode_workers": 16,
        "videos_per_window": 8,
        "max_samples_per_video_per_batch": 4,
        "fps": 1.0,
        "num_frames": 8,
        "max_pixels": 262144,
        "seed": 42,
        "use_4bit": True,
        "unfreeze_merger": False,
        "trust_remote_code": False,
    }
    defaults.update(config)
    for name, default in defaults.items():
        if not hasattr(args, name) or getattr(args, name) is None:
            setattr(args, name, default)
    args.train_file = Path(args.train_file)
    args.data_root = Path(args.data_root)
    args.output_dir = Path(args.output_dir)
    if args.no_4bit is None:
        args.no_4bit = not bool(args.use_4bit)
    if args.unfreeze_merger is None:
        args.unfreeze_merger = bool(defaults.get("unfreeze_merger", False))
    if args.trust_remote_code is None:
        args.trust_remote_code = bool(defaults.get("trust_remote_code", False))
    return args


def _replace_quantized_merger_linears(model: torch.nn.Module) -> int:
    """Restore visual.merger Linear4bit layers to bf16 for paper-style training.

    QLoRA keeps the language model quantized, but the KITE paper explicitly
    leaves the visual merger trainable.  This converts only that small module
    back to bf16 before PEFT freezes the quantized backbone.
    """
    if not hasattr(model, "visual") or not hasattr(model.visual, "merger"):
        return 0
    try:
        from bitsandbytes.functional import dequantize_4bit
    except Exception:
        dequantize_4bit = None
    replaced = 0

    def visit(parent: torch.nn.Module) -> None:
        nonlocal replaced
        for name, child in list(parent.named_children()):
            if child.__class__.__name__ == "Linear4bit":
                if dequantize_4bit is None or getattr(child.weight, "quant_state", None) is None:
                    raise RuntimeError("Cannot restore visual merger: bitsandbytes dequantizer unavailable")
                weight = dequantize_4bit(child.weight.data, child.weight.quant_state).to(torch.bfloat16)
                bias = child.bias.detach().to(torch.bfloat16) if child.bias is not None else None
                new = torch.nn.Linear(child.in_features, child.out_features, bias=bias is not None, device=weight.device, dtype=torch.bfloat16)
                new.weight.data.copy_(weight)
                if bias is not None:
                    new.bias.data.copy_(bias)
                setattr(parent, name, new)
                replaced += 1
            else:
                visit(child)

    visit(model.visual.merger)
    return replaced


def _unfreeze_merger(model: torch.nn.Module) -> int:
    count = 0
    for name, parameter in model.named_parameters():
        # PEFT prefixes wrapped parameters with ``base_model.model`` (and the
        # exact prefix differs between Transformers versions), so match the
        # stable module path rather than relying on one wrapper spelling.
        if "visual.merger." in name:
            parameter.requires_grad = True
            count += parameter.numel()
    return count


def _model_device(model: torch.nn.Module) -> Optional[torch.device]:
    for parameter in model.parameters():
        if parameter.device.type == "cuda":
            return parameter.device
    return None


def _auto_batch_size(model: torch.nn.Module, dataset: VideoQADataset, collator: SingleVideoCollator, max_batch: int, fraction: float) -> tuple[int, Optional[float], Optional[float]]:
    """Probe real video batches and select the largest batch under a VRAM target."""
    device = _model_device(model)
    if device is None or not torch.cuda.is_available():
        print("[VRAM] CUDA unavailable; using per_device_train_batch_size=1")
        return 1, None, None
    total = float(torch.cuda.get_device_properties(device).total_memory)
    target = total * float(fraction)
    selected = 1
    selected_peak = None
    model.train()
    limit = max(1, int(max_batch))
    # Probe a real, conservative batch.  Text lengths vary considerably in
    # RoboFAC; repeating a short row (the previous implementation) understated
    # the logits/loss allocation and selected a batch that later OOMed.  Pick
    # the longest records by cheap character count, encode each once, then
    # reuse those tensors for the memory probes.
    probe_rows = sorted(
        dataset.records,
        key=lambda row: len(str(row.get("question", ""))) + len(str(row.get("answer", ""))),
        reverse=True,
    )[:limit]
    probe_features = [dataset.encode_record(row) for row in probe_rows]
    candidates = []
    probe_size = 1
    while probe_size < limit:
        candidates.append(probe_size)
        probe_size *= 2
    if not candidates or candidates[-1] != limit:
        candidates.append(limit)
    for batch_size in candidates:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            batch = collator(probe_features[:batch_size])
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            model.zero_grad(set_to_none=True)
            outputs = model(**batch)
            outputs.loss.backward()
            peak = float(torch.cuda.max_memory_allocated(device))
            print(f"[VRAM] probe batch={batch_size} peak={peak / 2**30:.2f} GiB target={target / 2**30:.2f} GiB")
            if peak <= target:
                selected, selected_peak = batch_size, peak
            else:
                break
        except torch.cuda.OutOfMemoryError:
            print(f"[VRAM] probe batch={batch_size} OOM; keeping batch={selected}")
            break
        finally:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
    return selected, selected_peak, target


class VideoGroupedBatchSampler(torch.utils.data.Sampler[List[int]]):
    """Use coarse video windows without collapsing a video's QA distribution.

    A window contains a small shuffled set of videos (eight by default).  Rows
    are interleaved round-robin and capped per video in each batch, so a video
    cannot monopolize a batch.  The window keeps only a few videos hot in the
    decoder cache, while the window and row order remain randomized.
    """

    def __init__(
        self,
        dataset: VideoQADataset,
        batch_size: int,
        seed: int = 42,
        videos_per_window: int = 8,
        max_samples_per_video_per_batch: int = 4,
    ):
        self.batch_size = max(1, int(batch_size))
        self.seed = int(seed)
        self.videos_per_window = max(1, int(videos_per_window))
        self.max_samples_per_video_per_batch = max(1, int(max_samples_per_video_per_batch))
        self.epoch = 0
        groups: Dict[str, List[int]] = {}
        for index, row in enumerate(dataset.records):
            groups.setdefault(row["video"], []).append(index)
        self.groups = list(groups.values())

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        groups = [list(indices) for indices in self.groups]
        rng.shuffle(groups)
        batches: List[List[int]] = []
        for start in range(0, len(groups), self.videos_per_window):
            window = [list(indices) for indices in groups[start : start + self.videos_per_window]]
            for indices in window:
                rng.shuffle(indices)
            # Keep drawing from several active videos.  A video contributes at
            # most max_samples_per_video_per_batch rows before the next one is
            # selected, preserving QA diversity inside each batch.
            while any(window):
                active = [indices for indices in window if indices]
                rng.shuffle(active)
                batch: List[int] = []
                for indices in active:
                    if len(batch) >= self.batch_size:
                        break
                    take = min(
                        self.max_samples_per_video_per_batch,
                        self.batch_size - len(batch),
                        len(indices),
                    )
                    batch.extend(indices[:take])
                    del indices[:take]
                if batch:
                    rng.shuffle(batch)
                    batches.append(batch)
        self.epoch += 1
        yield from batches

    def __len__(self) -> int:
        return math.ceil(sum(len(group) for group in self.groups) / self.batch_size)


class VideoGroupedTrainer(Trainer):
    """Trainer using video-grouped batches and the collator's decode pool."""

    def __init__(self, *args: Any, video_batch_sampler: VideoGroupedBatchSampler, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.video_batch_sampler = video_batch_sampler

    def get_train_dataloader(self) -> torch.utils.data.DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer requires a train_dataset")
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_sampler=self.video_batch_sampler,
            collate_fn=self.data_collator,
            num_workers=0,
            pin_memory=True,
        )


def main() -> None:
    args = _merge_config(_parse_args())
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

    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.base_model, **model_kwargs)
    merger_layers = _replace_quantized_merger_linears(model) if args.unfreeze_merger and quantization_config is not None else 0
    if quantization_config is not None:
        model = prepare_model_for_kbit_training(model)
    model.enable_input_require_grads()
    model.config.use_cache = False
    lora = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    )
    model = get_peft_model(model, lora)
    merger_params = _unfreeze_merger(model) if args.unfreeze_merger else 0
    model.print_trainable_parameters()

    dataset = VideoQADataset(records, processor, fps=args.fps, max_pixels=args.max_pixels, num_frames=args.num_frames)
    collator = SingleVideoCollator(
        processor.tokenizer.pad_token_id,
        dataset=dataset,
        decode_workers=args.decode_workers,
    )
    batch_setting = str(args.per_device_train_batch_size).lower()
    if batch_setting == "auto":
        batch_size, probe_peak, target_peak = _auto_batch_size(model, dataset, collator, args.max_batch_size, args.target_vram_fraction)
    else:
        batch_size, probe_peak, target_peak = int(batch_setting), None, None
    train_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        optim=args.optimizer,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        save_total_limit=args.save_total_limit,
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=args.dataloader_num_workers > 0,
        seed=args.seed,
    )
    batch_sampler = VideoGroupedBatchSampler(
        dataset,
        batch_size,
        seed=args.seed,
        videos_per_window=args.videos_per_window,
        max_samples_per_video_per_batch=args.max_samples_per_video_per_batch,
    )
    trainer = VideoGroupedTrainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        data_collator=collator,
        video_batch_sampler=batch_sampler,
    )
    result = trainer.train(resume_from_checkpoint=None)
    trainer.save_model(str(args.output_dir))
    processor.save_pretrained(str(args.output_dir))
    summary = {
        "base_model": args.base_model,
        "records": len(records),
        "output_dir": str(args.output_dir),
        "config_file": str(args.config),
        "resolved_config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "selected_batch_size": batch_size,
        "probe_peak_memory_gib": None if probe_peak is None else probe_peak / 2**30,
        "target_memory_gib": None if target_peak is None else target_peak / 2**30,
        "unfrozen_merger_parameters": merger_params,
        "restored_merger_linear_layers": merger_layers,
        "train_metrics": result.metrics,
    }
    (args.output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
