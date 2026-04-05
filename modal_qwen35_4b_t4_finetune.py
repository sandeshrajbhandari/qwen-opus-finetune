"""Fine-tune Qwen3.5-4B on Modal T4 with normalized mixed datasets.

Modeled after modal_qwen35_18b_reap_a3b_coding_opus_train.py.

Usage:
  modal run modal_qwen35_4b_t4_finetune.py
  modal run modal_qwen35_4b_t4_finetune.py --max-steps 20 --max-train-samples 256
  modal run modal_qwen35_4b_t4_finetune.py --hf-repo-id your-name/repo
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
APP_NAME = "qwen35-4b-t4-mixed-finetune"
BASE_MODEL = "unsloth/Qwen3.5-4B"
OUTPUT_ROOT = "/outputs"
HF_CACHE_DIR = "/root/.cache/huggingface"

DEFAULT_DATASETS = [
    "nohurry/Opus-4.6-Reasoning-3000x-filtered",
    "TeichAI/claude-4.5-opus-high-reasoning-250x",
    "Jackrong/Qwen3.5-reasoning-700x",
]

DEFAULT_MAX_SEQ_LENGTH = 2048
DEFAULT_EPOCHS = 1.0
DEFAULT_MAX_STEPS = -1
DEFAULT_PER_DEVICE_BATCH_SIZE = 1
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 8
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_LORA_R = 32
DEFAULT_LORA_ALPHA = 64
DEFAULT_WARMUP_STEPS = 10
DEFAULT_WEIGHT_DECAY = 0.001
DEFAULT_MAX_TRAIN_SAMPLES = 0
DEFAULT_MAX_PER_DATASET = 0
DEFAULT_SEED = 3407


# ---------------------------------------------------------------------------
# Modal resources
# ---------------------------------------------------------------------------
app = modal.App(APP_NAME)

hf_cache_volume = modal.Volume.from_name("hf-cache-qwen35-4b-t4", create_if_missing=True)
outputs_volume = modal.Volume.from_name("qwen35-4b-t4-outputs", create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .apt_install("git")
    .run_commands("pip install --upgrade pip")
    .run_commands("pip install --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoo")
    .pip_install(
        "torch==2.8.0",
        "torchvision",
        "bitsandbytes",
        "xformers==0.0.32.post2",
        "trl==0.22.2",
        "datasets>=3.6.0,<4.0.0",
        "accelerate>=1.7.0",
        "huggingface-hub>=0.34.0",
        "sentencepiece",
        "protobuf",
        "wandb>=0.21.1",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

MINUTES = 60


@dataclass
class RunSummary:
    run_name: str
    base_model: str
    dataset_rows: int
    max_seq_length: int
    epochs: float
    max_steps: int
    save_steps: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    effective_batch_size: int
    learning_rate: float
    lora_r: int
    lora_alpha: int
    train_runtime_seconds: float | None
    train_loss: float | None
    peak_reserved_memory_gb: float
    output_dir: str
    hf_repo_id: str
    hf_path_in_repo: str


def make_run_name(max_seq_length: int, epochs: float, learning_rate: float) -> str:
    lr = str(learning_rate).replace(".", "p")
    return f"qwen35-4b-t4-msl{max_seq_length}-e{str(epochs).replace('.', '_')}-lr{lr}"


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if "text" in item and isinstance(item["text"], str):
                    parts.append(item["text"])
                elif "value" in item and isinstance(item["value"], str):
                    parts.append(item["value"])
        return "\n".join(p.strip() for p in parts if p and p.strip()).strip()
    if isinstance(value, dict):
        if "text" in value and isinstance(value["text"], str):
            return value["text"].strip()
        if "value" in value and isinstance(value["value"], str):
            return value["value"].strip()
    return str(value).strip()


def _first_non_empty(example: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        if key in example:
            text = _as_text(example[key])
            if text:
                return text
    return ""


def _extract_user_assistant_from_turns(turns: Any) -> tuple[str, str]:
    if not isinstance(turns, list):
        return "", ""
    user_text = ""
    assistant_text = ""
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        role = (
            str(
                turn.get("role")
                or turn.get("from")
                or turn.get("speaker")
                or turn.get("author")
                or ""
            )
            .strip()
            .lower()
        )
        content = _as_text(
            turn.get("content")
            or turn.get("value")
            or turn.get("text")
            or turn.get("message")
            or ""
        )
        if not content:
            continue
        if role in {"user", "human", "instruction", "prompt"} and not user_text:
            user_text = content
        elif role in {"assistant", "gpt", "model", "bot", "response"}:
            assistant_text = content
    return user_text, assistant_text


def _has_think_tags(text: str) -> bool:
    return "<think>" in text and "</think>" in text


def _compose_assistant(thinking: str, answer: str) -> str:
    thinking = thinking.strip()
    answer = answer.strip()
    if thinking:
        return f"<think>\n{thinking}\n</think>\n\n{answer}".strip()
    return answer


def normalize_example(example: dict[str, Any]) -> dict[str, str]:
    prompt = _first_non_empty(
        example,
        ["problem", "prompt", "question", "instruction", "input", "query", "task"],
    )
    thinking = _first_non_empty(
        example,
        ["thinking", "reasoning", "cot", "chain_of_thought", "chain_of_thoughts", "rationale"],
    )
    answer = _first_non_empty(
        example,
        ["solution", "answer", "output", "response", "assistant", "completion", "final_answer"],
    )

    conv_like = (
        example.get("conversation")
        or example.get("conversations")
        or example.get("messages")
        or example.get("chat")
    )
    conv_user, conv_assistant = _extract_user_assistant_from_turns(conv_like)
    if not prompt and conv_user:
        prompt = conv_user

    assistant = ""
    if conv_assistant:
        assistant = conv_assistant
    elif answer:
        assistant = _compose_assistant(thinking, answer)

    if assistant and _has_think_tags(assistant):
        return {"prompt": prompt.strip(), "assistant": assistant.strip()}
    if assistant and thinking:
        assistant = _compose_assistant(thinking, assistant)
    return {"prompt": prompt.strip(), "assistant": assistant.strip()}


def safe_model_subdir(model_id: str) -> str:
    cleaned = model_id.strip().replace("/", "__")
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", cleaned)
    cleaned = cleaned.strip("._-")
    return cleaned or "model"


def format_metric_token(value: float) -> str:
    s = f"{value:.10f}".rstrip("0").rstrip(".")
    if not s:
        s = "0"
    return s.replace(".", "p").replace("-", "m")


@app.function(
    image=image,
    gpu="T4",
    timeout=10 * MINUTES,
    volumes={
        HF_CACHE_DIR: hf_cache_volume,
        OUTPUT_ROOT: outputs_volume,
    },
)
def train(
    model_id: str = BASE_MODEL,
    datasets_json: str = "",
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    epochs: float = DEFAULT_EPOCHS,
    max_steps: int = DEFAULT_MAX_STEPS,
    per_device_train_batch_size: int = DEFAULT_PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps: int = DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    lora_r: int = DEFAULT_LORA_R,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
    warmup_steps: int = DEFAULT_WARMUP_STEPS,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    max_train_samples: int = DEFAULT_MAX_TRAIN_SAMPLES,
    max_per_dataset: int = DEFAULT_MAX_PER_DATASET,
    seed: int = DEFAULT_SEED,
    report_to: str = "none",
    hf_repo_id: str = "",
    hf_token: str = "",
):
    import torch
    import unsloth
    from datasets import concatenate_datasets, load_dataset
    from huggingface_hub import HfApi
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only
    from trl import SFTConfig, SFTTrainer

    random.seed(seed)
    job_start_time = time.perf_counter()

    dataset_names = json.loads(datasets_json) if datasets_json else list(DEFAULT_DATASETS)
    run_name = make_run_name(max_seq_length=max_seq_length, epochs=epochs, learning_rate=learning_rate)
    run_output_dir = Path(OUTPUT_ROOT) / run_name
    trainer_output_dir = run_output_dir / "trainer"
    lora_output_dir = run_output_dir / "lora_adapter"
    run_output_dir.mkdir(parents=True, exist_ok=True)

    # ----- Load model & tokenizer (same pattern as 18b script) -----
    print("Loading base model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=lora_alpha,
        use_gradient_checkpointing=True,
        random_state=seed,
        bias="none",
    )

    # ----- Load & normalize datasets -----
    normalized_parts = []
    for dataset_name in dataset_names:
        print(f"\n=== Loading dataset: {dataset_name} ===")
        ds_obj = load_dataset(dataset_name)
        ds = ds_obj["train"] if "train" in ds_obj else ds_obj[next(iter(ds_obj.keys()))]
        print(f"Rows before normalization: {len(ds)}")

        normalized = ds.map(
            normalize_example,
            remove_columns=ds.column_names,
            desc=f"Normalizing {dataset_name}",
        )
        normalized = normalized.filter(
            lambda x: bool(x["prompt"]) and bool(x["assistant"]),
            desc=f"Dropping empty pairs for {dataset_name}",
        )
        normalized = normalized.add_column("source_dataset", [dataset_name] * len(normalized))

        if max_per_dataset > 0 and len(normalized) > max_per_dataset:
            normalized = normalized.shuffle(seed=seed).select(range(max_per_dataset))
            print(f"Applied max-per-dataset cap: {max_per_dataset}")

        print(f"Rows after normalization: {len(normalized)}")
        normalized_parts.append(normalized)

    merged = concatenate_datasets(normalized_parts).shuffle(seed=seed)
    if max_train_samples > 0 and len(merged) > max_train_samples:
        merged = merged.select(range(max_train_samples))
        print(f"Applied global max-train-samples cap: {max_train_samples}")

    # ----- Format into chat text (same pattern as 18b script) -----
    def _fmt(example):
        assistant_text = example["assistant"]
        messages = [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": assistant_text},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    train_dataset = merged.map(
        _fmt,
        remove_columns=merged.column_names,
        desc="Applying chat template",
    )

    # ----- Training setup -----
    gpu_stats = torch.cuda.get_device_properties(0)
    supports_bf16 = torch.cuda.is_bf16_supported()
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"Dataset rows = {len(train_dataset)}")
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved before training.")
    print(f"bf16 supported = {supports_bf16}")

    effective_batch_size = per_device_train_batch_size * gradient_accumulation_steps
    steps_per_epoch = math.ceil(len(train_dataset) / effective_batch_size)
    save_steps = max(1, steps_per_epoch // 2)
    setup_seconds = time.perf_counter() - job_start_time

    print(f"Effective batch size = {effective_batch_size}")
    print(f"Steps per epoch = {steps_per_epoch}")

    # ----- SFTTrainer (direct, same as 18b script) -----
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=SFTConfig(
            dataset_text_field="text",
            max_length=max_seq_length,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            num_train_epochs=epochs,
            max_steps=max_steps,
            learning_rate=learning_rate,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=weight_decay,
            lr_scheduler_type="linear",
            fp16=not supports_bf16,
            bf16=supports_bf16,
            seed=seed,
            output_dir=str(trainer_output_dir),
            report_to=report_to,
            dataset_num_proc=1,
            save_strategy="steps",
            save_steps=save_steps,
            save_total_limit=2,
        ),
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n<think>",
    )

    print("Starting training...")
    stats = trainer.train()

    # ----- Save -----
    lora_output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(lora_output_dir))
    tokenizer.save_pretrained(str(lora_output_dir))
    print(f"Saved LoRA adapter to: {lora_output_dir}")

    hf_path_in_repo = ""
    if hf_repo_id:
        token = hf_token or os.environ.get("HF_TOKEN", "")
        if not token:
            raise RuntimeError("Provide hf_token argument or HF_TOKEN env var when hf_repo_id is set.")
        model_subdir = safe_model_subdir(model_id)
        run_stamp = datetime.now(timezone.utc).strftime("run-%Y%m%d-%H%M%S")
        run_folder = (
            f"{run_stamp}-e{format_metric_token(epochs)}-lr{format_metric_token(learning_rate)}"
        )
        hf_path_in_repo = f"{model_subdir}/{run_folder}/lora_adapter"
        api = HfApi(token=token)
        api.create_repo(repo_id=hf_repo_id, repo_type="model", exist_ok=True)
        api.upload_folder(
            repo_id=hf_repo_id,
            repo_type="model",
            folder_path=str(lora_output_dir),
            path_in_repo=hf_path_in_repo,
            commit_message=f"Upload LoRA adapter for {model_id} ({run_stamp})",
        )
        print(f"Pushed LoRA adapter to: https://huggingface.co/{hf_repo_id}/tree/main/{hf_path_in_repo}")

    # ----- Summary -----
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    summary = RunSummary(
        run_name=run_name,
        base_model=model_id,
        dataset_rows=len(train_dataset),
        max_seq_length=max_seq_length,
        epochs=epochs,
        max_steps=max_steps,
        save_steps=save_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        effective_batch_size=effective_batch_size,
        learning_rate=learning_rate,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        train_runtime_seconds=stats.metrics.get("train_runtime"),
        train_loss=stats.metrics.get("train_loss"),
        peak_reserved_memory_gb=used_memory,
        output_dir=str(run_output_dir),
        hf_repo_id=hf_repo_id,
        hf_path_in_repo=hf_path_in_repo,
    )

    summary_path = run_output_dir / "summary.json"
    summary_path.write_text(json.dumps(asdict(summary), indent=2))
    print(json.dumps(asdict(summary), indent=2))

    outputs_volume.commit()
    hf_cache_volume.commit()
    return asdict(summary)


@app.local_entrypoint()
def main(
    model_id: str = BASE_MODEL,
    datasets_json: str = "",
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    epochs: float = DEFAULT_EPOCHS,
    max_steps: int = DEFAULT_MAX_STEPS,
    per_device_train_batch_size: int = DEFAULT_PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps: int = DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    lora_r: int = DEFAULT_LORA_R,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
    warmup_steps: int = DEFAULT_WARMUP_STEPS,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    max_train_samples: int = DEFAULT_MAX_TRAIN_SAMPLES,
    max_per_dataset: int = DEFAULT_MAX_PER_DATASET,
    seed: int = DEFAULT_SEED,
    report_to: str = "none",
    hf_repo_id: str = "",
    hf_token: str = "",
):
    train.remote(
        model_id=model_id,
        datasets_json=datasets_json,
        max_seq_length=max_seq_length,
        epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        max_train_samples=max_train_samples,
        max_per_dataset=max_per_dataset,
        seed=seed,
        report_to=report_to,
        hf_repo_id=hf_repo_id,
        hf_token=hf_token,
    )
