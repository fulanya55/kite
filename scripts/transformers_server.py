#!/usr/bin/env python3
"""Small OpenAI-compatible server for reproducible local evaluation.

This is a dependency-light alternative to vLLM for a single GPU.  It accepts
the image messages emitted by :class:`kite.qa.adapter.RoboFACAdapter` and can
optionally attach a PEFT LoRA adapter to the base Qwen checkpoint.
"""

from __future__ import annotations

import argparse
import base64
import io
import threading
import time
from typing import Any

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import uvicorn


class CompletionRequest(BaseModel):
    model: str
    messages: list[dict[str, Any]]
    max_tokens: int = 256
    temperature: float = 0.0


def _image_from_url(url: str) -> Image.Image:
    if url.startswith("data:"):
        payload = url.split(",", 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")
    return Image.open(url).convert("RGB")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Local Qwen/KITE model directory")
    p.add_argument("--lora-adapter", default=None)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--max-new-tokens", type=int, default=256)
    args = p.parse_args()
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=dtype, device_map="auto"
    )
    if args.lora_adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.lora_adapter)
    model.eval()
    # Qwen checkpoints sometimes store temperature=0 in generation_config;
    # Transformers warns about that value even for greedy decoding.
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.temperature = None
    lock = threading.Lock()
    app = FastAPI(title="KITE Transformers server")

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": args.model, "object": "model", "owned_by": "local"}]}

    @app.post("/v1/chat/completions")
    def completions(req: CompletionRequest) -> dict[str, Any]:
        content = req.messages[-1].get("content", [])
        images: list[Image.Image] = []
        text_parts: list[str] = []
        for part in content if isinstance(content, list) else [{"type": "text", "text": str(content)}]:
            if part.get("type") == "image_url":
                images.append(_image_from_url(part["image_url"]["url"]))
            elif part.get("type") == "text":
                text_parts.append(part.get("text", ""))
        user_content: list[dict[str, Any]] = [{"type": "image", "image": im} for im in images]
        user_content.append({"type": "text", "text": "\n".join(text_parts)})
        messages = [{"role": "user", "content": user_content}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=images or None, return_tensors="pt", padding=True)
        device = next(model.parameters()).device
        inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
        generation_kwargs = {
            "max_new_tokens": min(req.max_tokens, args.max_new_tokens),
            "do_sample": req.temperature > 0,
        }
        if req.temperature > 0:
            generation_kwargs["temperature"] = req.temperature
        with lock, torch.inference_mode():
            output = model.generate(**inputs, **generation_kwargs)
        generated = output[:, inputs["input_ids"].shape[1] :]
        answer = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        now = int(time.time())
        return {
            "id": f"kite-{now}", "object": "chat.completion", "created": now, "model": req.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
        }

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
