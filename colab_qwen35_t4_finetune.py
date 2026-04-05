"""
Colab T4 script: single-run Qwen3.5 LoRA fine-tuning (no sweep).

This script auto-normalizes multiple reasoning datasets with mixed schemas
into a unified chat format and runs one fine-tuning job.

Fixed LoRA settings:
  - rank (r): 32
  - alpha: 64

Datasets used by default:
  1) nohurry/Opus-4.6-Reasoning-3000x-filtered
  2) TeichAI/claude-4.5-opus-high-reasoning-250x
  3) Jackrong/Qwen3.5-reasoning-700x

Recommended Colab setup (run once in a notebook cell before this script):
  pip install -q --upgrade pip
  pip install -q --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoo
  pip install -q torch torchvision bitsandbytes xformers trl datasets accelerate \
      huggingface-hub sentencepiece protobuf

Example:
  python colab_qwen35_t4_finetune.py \
    --model-id unsloth/Qwen3.5-4B \
    --output-dir /content/qwen35_single_run \
    --max-seq-length 2048 \
    --learning-rate 2e-4 \
    --num-train-epochs 1 \
    --per-device-train-batch-size 1 \
    --gradient-accumulation-steps 8 \
    --max-steps 120 \
    --max-train-samples 1500

If you hit OOM on T4, switch to:
  --model-id unsloth/Qwen3.5-2B
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import math
import os
import random
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from huggingface_hub import HfApi
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only


DEFAULT_DATASETS = [
    "nohurry/Opus-4.6-Reasoning-3000x-filtered",
    "TeichAI/claude-4.5-opus-high-reasoning-250x",
    "Jackrong/Qwen3.5-reasoning-700x",
]

FIXED_LORA_R = 32
FIXED_LORA_ALPHA = 64


@dataclass
class TrainSummary:
    model_id: str
    dataset_rows: int
    lora_r: int
    lora_alpha: int
    learning_rate: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    effective_batch_size: int
    num_train_epochs: float
    max_steps: int
    max_seq_length: int
    train_runtime_seconds: float | None
    train_loss: float | None
    peak_reserved_memory_gb: float
    output_dir: str
    save_steps: int
    hf_repo_id: str
    hf_path_in_repo: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-run Qwen3.5 LoRA finetune for Colab T4.")
    parser.add_argument("--model-id", type=str, default="unsloth/Qwen3.5-4B")
    parser.add_argument("--output-dir", type=str, default="./qwen35_t4_finetune_out")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1, help="-1 disables max_steps.")
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-per-dataset", type=int, default=0)
    parser.add_argument("--report-to", type=str, default="none")
    parser.add_argument("--save-merged-16bit", action="store_true")
    parser.add_argument(
        "--hf-repo-id",
        type=str,
        default="sandeshrajx/qwopus-lora-tests",
        help="Destination model repo for LoRA upload.",
    )
    parser.add_argument(
        "--disable-hf-upload",
        action="store_true",
        help="Disable automatic LoRA upload to Hugging Face Hub.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="HF datasets to load and normalize.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_split(dataset_obj: Dataset | DatasetDict, preferred_split: str = "train") -> Dataset:
    if isinstance(dataset_obj, Dataset):
        return dataset_obj
    if preferred_split in dataset_obj:
        return dataset_obj[preferred_split]
    if "train" in dataset_obj:
        return dataset_obj["train"]
    first_key = next(iter(dataset_obj.keys()))
    print(f"[warn] Preferred split not found. Falling back to split: {first_key}")
    return dataset_obj[first_key]


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


def load_and_normalize_dataset(dataset_name: str, max_rows: int, seed: int) -> Dataset:
    print(f"\n=== Loading dataset: {dataset_name} ===")
    ds_obj = load_dataset(dataset_name)
    ds = pick_split(ds_obj)
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

    if max_rows > 0 and len(normalized) > max_rows:
        normalized = normalized.shuffle(seed=seed).select(range(max_rows))
        print(f"Applied max-per-dataset cap: {max_rows}")

    print(f"Rows after normalization: {len(normalized)}")
    return normalized


def format_to_chat_text(dataset: Dataset, tokenizer) -> Dataset:
    def _fmt(example: dict[str, Any]) -> dict[str, str]:
        text = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": example["prompt"]},
                {"role": "assistant", "content": example["assistant"]},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text, "source_dataset": example["source_dataset"]}

    return dataset.map(
        _fmt,
        remove_columns=dataset.column_names,
        desc="Applying chat template",
    )


def safe_model_subdir(model_id: str) -> str:
    # Keep model identity readable while making it path-safe for Hub subfolders.
    cleaned = model_id.strip().replace("/", "__")
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", cleaned)
    cleaned = cleaned.strip("._-")
    return cleaned or "model"


def format_metric_token(value: float) -> str:
    # Keep floats readable and path-safe (e.g. 2e-4 -> 0p0002).
    s = f"{value:.10f}".rstrip("0").rstrip(".")
    if not s:
        s = "0"
    s = s.replace(".", "p")
    s = s.replace("-", "m")
    return s


def build_sft_config_compat(base_kwargs: dict[str, Any], max_seq_length: int) -> SFTConfig:
    """Build SFTConfig while handling TRL version differences.

    Some TRL versions use `max_seq_length`, others use `max_length`.
    This helper also drops unsupported optional kwargs gracefully.
    """
    sig = inspect.signature(SFTConfig.__init__)
    supported = set(sig.parameters.keys())
    kwargs = dict(base_kwargs)

    if "max_seq_length" in supported:
        kwargs["max_seq_length"] = max_seq_length
    elif "max_length" in supported:
        kwargs["max_length"] = max_seq_length
    else:
        print("[warn] Neither max_seq_length nor max_length found in SFTConfig; using defaults.")

    filtered = {k: v for k, v in kwargs.items() if k in supported}
    dropped = sorted(k for k in kwargs.keys() if k not in supported)
    if dropped:
        print(f"[warn] Dropping unsupported SFTConfig args for this TRL version: {dropped}")

    return SFTConfig(**filtered)


def build_sft_trainer_compat(
    *,
    model,
    tokenizer,
    train_dataset: Dataset,
    sft_cfg: SFTConfig,
) -> SFTTrainer:
    """Build SFTTrainer compatibly across TRL versions.

    TRL versions differ on whether they accept `tokenizer` or `processing_class`.
    """
    sig = inspect.signature(SFTTrainer.__init__)
    supported = set(sig.parameters.keys())
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "train_dataset": train_dataset,
        "args": sft_cfg,
    }

    if "tokenizer" in supported:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in supported:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        print(
            "[warn] SFTTrainer accepts neither tokenizer nor processing_class;"
            " relying on model defaults."
        )

    return SFTTrainer(**trainer_kwargs)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available. This script is intended for Colab T4.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = out_dir / "single_run"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using model: {args.model_id}")
    print(f"Output dir: {out_dir.resolve()}")
    print(f"Fixed LoRA: r={FIXED_LORA_R}, alpha={FIXED_LORA_ALPHA}")
    print(f"Datasets: {args.datasets}")

    _, tokenizer_for_format = FastLanguageModel.from_pretrained(
        model_name=args.model_id,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
    )
    del _
    gc.collect()
    torch.cuda.empty_cache()

    normalized_parts: list[Dataset] = []
    for ds_name in args.datasets:
        ds_norm = load_and_normalize_dataset(
            dataset_name=ds_name,
            max_rows=args.max_per_dataset,
            seed=args.seed,
        )
        normalized_parts.append(ds_norm)

    if not normalized_parts:
        raise RuntimeError("No datasets loaded.")

    merged = concatenate_datasets(normalized_parts).shuffle(seed=args.seed)
    if args.max_train_samples > 0 and len(merged) > args.max_train_samples:
        merged = merged.select(range(args.max_train_samples))
        print(f"Applied global max-train-samples cap: {args.max_train_samples}")

    print(f"\nMerged normalized rows: {len(merged)}")
    by_source: dict[str, int] = {}
    for src in merged["source_dataset"]:
        by_source[src] = by_source.get(src, 0) + 1
    print("Rows by source dataset:")
    for k, v in sorted(by_source.items()):
        print(f"  - {k}: {v}")

    train_dataset = format_to_chat_text(merged, tokenizer_for_format)
    del tokenizer_for_format
    gc.collect()
    torch.cuda.empty_cache()

    preview_path = out_dir / "normalized_preview.jsonl"
    with open(preview_path, "w", encoding="utf-8") as f:
        for i in range(min(5, len(train_dataset))):
            f.write(json.dumps(train_dataset[i], ensure_ascii=False) + "\n")
    print(f"Wrote preview examples to: {preview_path}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_id,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=FIXED_LORA_R,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        lora_alpha=FIXED_LORA_ALPHA,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=args.seed,
        max_seq_length=args.max_seq_length,
        use_rslora=False,
        loftq_config=None,
    )

    supports_bf16 = torch.cuda.is_bf16_supported()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    denom = args.per_device_train_batch_size * args.gradient_accumulation_steps * max(world_size, 1)
    steps_per_epoch = max(1, math.ceil(len(train_dataset) / max(denom, 1)))
    save_steps = max(1, steps_per_epoch // 2)
    print(f"Estimated steps/epoch: {steps_per_epoch} | save_steps (~half epoch): {save_steps}")

    sft_kwargs = dict(
        dataset_text_field="text",
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=args.weight_decay,
        lr_scheduler_type="linear",
        fp16=not supports_bf16,
        bf16=supports_bf16,
        seed=args.seed,
        output_dir=str(run_dir / "trainer"),
        report_to=args.report_to,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        dataset_num_proc=1,
        packing=False,
    )
    sft_cfg = build_sft_config_compat(
        base_kwargs=sft_kwargs,
        max_seq_length=args.max_seq_length,
    )

    trainer = build_sft_trainer_compat(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        sft_cfg=sft_cfg,
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n<think>",
    )

    print("Training...")
    train_output = trainer.train()

    adapter_dir = run_dir / "lora_adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"Saved LoRA adapter to: {adapter_dir}")

    if args.save_merged_16bit:
        merged_dir = run_dir / "merged_16bit"
        merged_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained_merged(
            str(merged_dir),
            tokenizer,
            save_method="merged_16bit",
        )
        print(f"Saved merged 16-bit model to: {merged_dir}")

    hf_path_in_repo = ""
    if args.hf_repo_id and not args.disable_hf_upload:
        if "HF_TOKEN" not in os.environ:
            raise RuntimeError("HF_TOKEN must be set for Hub upload.")
        token = os.environ["HF_TOKEN"]
        model_subdir = safe_model_subdir(args.model_id)
        run_stamp = datetime.now(timezone.utc).strftime("run-%Y%m%d-%H%M%S")
        epochs_token = format_metric_token(args.num_train_epochs)
        lr_token = format_metric_token(args.learning_rate)
        run_folder = f"{run_stamp}-e{epochs_token}-lr{lr_token}"
        hf_path_in_repo = f"{model_subdir}/{run_folder}/lora_adapter"

        api = HfApi(token=token)
        api.create_repo(repo_id=args.hf_repo_id, repo_type="model", exist_ok=True)
        api.upload_folder(
            repo_id=args.hf_repo_id,
            repo_type="model",
            folder_path=str(adapter_dir),
            path_in_repo=hf_path_in_repo,
            commit_message=f"Upload LoRA adapter for {args.model_id} ({run_stamp})",
        )
        print(f"Pushed LoRA adapter to: https://huggingface.co/{args.hf_repo_id}/tree/main/{hf_path_in_repo}")

    peak_memory_gb = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    summary = TrainSummary(
        model_id=args.model_id,
        dataset_rows=len(train_dataset),
        lora_r=FIXED_LORA_R,
        lora_alpha=FIXED_LORA_ALPHA,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        effective_batch_size=args.per_device_train_batch_size * args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        max_seq_length=args.max_seq_length,
        train_runtime_seconds=train_output.metrics.get("train_runtime"),
        train_loss=train_output.metrics.get("train_loss"),
        peak_reserved_memory_gb=peak_memory_gb,
        output_dir=str(run_dir),
        save_steps=save_steps,
        hf_repo_id=args.hf_repo_id if not args.disable_hf_upload else "",
        hf_path_in_repo=hf_path_in_repo,
    )
    summary_path = run_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2)
    print(f"Saved summary to: {summary_path}")
    print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    if "HF_TOKEN" in os.environ:
        print("HF_TOKEN detected in environment.")
    main()
