"""Compare base Qwen3.5-18B-REAP-A3B-Coding against a saved LoRA run.

Usage:
    modal run modal_compare_qwen35_18b_base_vs_lora.py
    modal run modal_compare_qwen35_18b_base_vs_lora.py --run-name qwen35-18b-reap-a3b-coding-opus-msl8192-e2_0-lr2e4
    modal run modal_compare_qwen35_18b_base_vs_lora.py --prompt "Write a flappybird game using web technology in a single html file"
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import modal


APP_NAME = "qwen35-18b-base-vs-lora-compare"
BASE_MODEL = "Flagstone8878/Qwen3.5-18B-REAP-A3B-Coding"
HF_CACHE_DIR = "/root/.cache/huggingface"
OUTPUT_ROOT = "/outputs"
MODEL_SNAPSHOT_DIR = f"{HF_CACHE_DIR}/preloaded/flagstone_qwen35_18b_reap_a3b_coding"
DEFAULT_RUN_NAME = "qwen35-18b-reap-a3b-coding-opus-msl8192-e2_0-lr2e4"
DEFAULT_MAX_SEQ_LENGTH = 8192
DEFAULT_MAX_NEW_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_SEED = 3407
DEFAULT_PROMPT = "write a flappybird game using web technology in a single html file"

app = modal.App(APP_NAME)

hf_cache_volume = modal.Volume.from_name("hf-cache-qwen35-18b-reap-a3b", create_if_missing=True)
outputs_volume = modal.Volume.from_name("qwen35-18b-reap-a3b-opus-outputs", create_if_missing=True)

huggingface_secret = modal.Secret.from_name(
    "huggingface-secret", required_keys=["HF_TOKEN"]
)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .apt_install("git")
    .run_commands("pip install --upgrade -qqq uv")
    .run_commands(
        "uv pip install -qqq --system "
        "\"torch==2.8.0\" "
        "\"triton>=3.3.0\" "
        "numpy pillow torchvision bitsandbytes xformers==0.0.32.post2 "
        "\"unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo\" "
        "\"unsloth[base] @ git+https://github.com/unslothai/unsloth\""
    )
    .run_commands(
        "uv pip install --system --upgrade --no-deps "
        "tokenizers trl==0.22.2 unsloth unsloth_zoo"
    )
    .run_commands("uv pip install --system transformers==5.2.0")
    .run_commands(
        "uv pip install --system --no-build-isolation "
        "flash-linear-attention causal_conv1d==1.6.0"
    )
    .run_commands("pip uninstall unsloth unsloth_zoo -y")
    .run_commands("pip install git+https://github.com/unslothai/unsloth-zoo.git --no-deps")
    .run_commands("pip install git+https://github.com/unslothai/unsloth.git --no-deps")
    .pip_install(
        "huggingface-hub>=0.34.0",
        "sentencepiece",
        "protobuf",
    )
    .env({
        "HF_XET_HIGH_PERFORMANCE": "1",
        "UNSLOTH_MOE_DISABLE_AUTOTUNE": "1",
    })
)


def extract_response_text(decoded_text: str) -> str:
    assistant_marker = "<|im_start|>assistant\n"
    start = decoded_text.rfind(assistant_marker)
    if start != -1:
        decoded_text = decoded_text[start + len(assistant_marker):]
    return decoded_text.replace("<|im_end|>", "").strip()


def split_thinking(response_text: str) -> tuple[str, str]:
    match = re.search(r"<think>\s*(.*?)\s*</think>\s*(.*)", response_text, flags=re.DOTALL)
    if not match:
        return "", response_text.strip()
    return match.group(1).strip(), match.group(2).strip()


def build_result(
    *,
    label: str,
    model_path: str,
    prompt: str,
    tokenizer,
    response_text: str,
    prompt_token_count: int,
    generated_token_count: int,
    elapsed_seconds: float,
) -> dict:
    thinking_text, answer_text = split_thinking(response_text)
    thinking_token_count = len(tokenizer.encode(thinking_text, add_special_tokens=False)) if thinking_text else 0
    answer_token_count = len(tokenizer.encode(answer_text, add_special_tokens=False)) if answer_text else 0

    return {
        "label": label,
        "model_path": model_path,
        "prompt": prompt,
        "response_text": response_text,
        "thinking_text": thinking_text,
        "answer_text": answer_text,
        "has_thinking_block": bool(thinking_text),
        "prompt_token_count": prompt_token_count,
        "generated_token_count": generated_token_count,
        "thinking_token_count": thinking_token_count,
        "answer_token_count": answer_token_count,
        "response_char_count": len(response_text),
        "elapsed_seconds": elapsed_seconds,
        "tokens_per_second": (
            generated_token_count / elapsed_seconds
            if elapsed_seconds > 0
            else None
        ),
    }


@app.function(
    image=image,
    gpu=["A100-80GB"],
    volumes={
        HF_CACHE_DIR: hf_cache_volume,
        OUTPUT_ROOT: outputs_volume,
    },
    secrets=[huggingface_secret],
    timeout=60 * 60,
)
def compare_models(
    run_name: str = DEFAULT_RUN_NAME,
    prompt: str = DEFAULT_PROMPT,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    seed: int = DEFAULT_SEED,
):
    import torch
    from transformers import AutoTokenizer, set_seed
    from unsloth import FastLanguageModel

    run_output_dir = Path(OUTPUT_ROOT) / run_name
    lora_dir = run_output_dir / "lora"
    if not lora_dir.exists():
        raise FileNotFoundError(f"LoRA adapter directory not found: {lora_dir}")

    model_specs = [
        ("base", MODEL_SNAPSHOT_DIR if Path(MODEL_SNAPSHOT_DIR).exists() else BASE_MODEL),
        ("lora", str(lora_dir)),
    ]

    results = []

    for label, model_path in model_specs:
        print(f"Loading {label} model from {model_path}")
        model, processor = FastLanguageModel.from_pretrained(
            model_path,
            max_seq_length=max_seq_length,
            load_in_4bit=False,
            fast_inference=False,
        )
        if hasattr(FastLanguageModel, "for_inference"):
            FastLanguageModel.for_inference(model)

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        processor_tokenizer = getattr(processor, "tokenizer", None)
        if processor_tokenizer is None:
            processor_tokenizer = getattr(processor, "_tokenizer", None)
        if processor_tokenizer is not None and hasattr(processor_tokenizer, "apply_chat_template"):
            tokenizer = processor_tokenizer

        messages = [{"role": "user", "content": prompt}]
        chat_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(chat_text, return_tensors="pt")
        prompt_token_count = int(inputs["input_ids"].shape[-1])
        inputs = {key: value.to(model.device) for key, value in inputs.items()}

        set_seed(seed)
        start_time = time.perf_counter()
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                use_cache=True,
            )
        elapsed_seconds = time.perf_counter() - start_time

        generated_ids = outputs[0, prompt_token_count:]
        generated_token_count = int(generated_ids.shape[-1])
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=False)
        response_text = extract_response_text(decoded)

        result = build_result(
            label=label,
            model_path=model_path,
            prompt=prompt,
            tokenizer=tokenizer,
            response_text=response_text,
            prompt_token_count=prompt_token_count,
            generated_token_count=generated_token_count,
            elapsed_seconds=elapsed_seconds,
        )
        print(json.dumps(result, indent=2))
        results.append(result)

        del model
        torch.cuda.empty_cache()

    comparison = {
        "run_name": run_name,
        "prompt": prompt,
        "max_seq_length": max_seq_length,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "results": results,
    }
    print(json.dumps(comparison, indent=2))
    return comparison


@app.local_entrypoint()
def main(
    run_name: str = DEFAULT_RUN_NAME,
    prompt: str = DEFAULT_PROMPT,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    seed: int = DEFAULT_SEED,
):
    compare_models.remote(
        run_name=run_name,
        prompt=prompt,
        max_seq_length=max_seq_length,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
    )
