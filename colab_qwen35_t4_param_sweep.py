"""
Colab T4 script: parameter sweeps for Qwen3.5 LoRA fine-tuning.

This script is designed for quick hyperparameter testing on smaller models
before scaling to larger runs.

Datasets used (auto-normalized into a shared chat format):
  1) nohurry/Opus-4.6-Reasoning-3000x-filtered
  2) TeichAI/claude-4.5-opus-high-reasoning-250x
  3) Jackrong/Qwen3.5-reasoning-700x

Recommended Colab setup (run once in a notebook cell before this script):
  pip install -q --upgrade pip
  pip install -q --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoo
  pip install -q torch torchvision bitsandbytes xformers trl datasets accelerate \
      huggingface-hub sentencepiece protobuf

Example:
  python colab_qwen35_t4_param_sweep.py \
    --output-dir /content/qwen35_sweeps \
    --model-id unsloth/Qwen3.5-4B \
    --max-seq-length 2048 \
    --max-steps-per-run 120 \
    --max-train-samples 1500

If you hit OOM on T4, switch to:
  --model-id unsloth/Qwen3.5-2B
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only


DEFAULT_DATASETS = [
    "nohurry/Opus-4.6-Reasoning-3000x-filtered",
    "TeichAI/claude-4.5-opus-high-reasoning-250x",
    "Jackrong/Qwen3.5-reasoning-700x",
]

# Quick sweep presets for T4 experimentation.
DEFAULT_SWEEP = [
    {
        "name": "lr2e4_r16_bs1_ga8",
        "learning_rate": 2e-4,
        "lora_r": 16,
        "lora_alpha": 16,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "num_train_epochs": 1.0,
    },
    {
        "name": "lr1e4_r16_bs1_ga8",
        "learning_rate": 1e-4,
        "lora_r": 16,
        "lora_alpha": 16,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "num_train_epochs": 1.0,
    },
    {
        "name": "lr2e4_r32_bs1_ga8",
        "learning_rate": 2e-4,
        "lora_r": 32,
        "lora_alpha": 32,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "num_train_epochs": 1.0,
    },
]


@dataclass
class RunSummary:
    run_name: str
    model_id: str
    dataset_rows: int
    learning_rate: float
    lora_r: int
    lora_alpha: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    effective_batch_size: int
    num_train_epochs: float
    max_steps: int
    max_seq_length: int
    train_runtime_seconds: float | None
    train_loss: float | None
    peak_reserved_memory_gb: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3.5 LoRA parameter sweep for Colab T4.")
    parser.add_argument(
        "--model-id",
        type=str,
        default="unsloth/Qwen3.5-4B",
        help="Use unsloth/Qwen3.5-4B first; fallback to unsloth/Qwen3.5-2B if OOM.",
    )
    parser.add_argument("--output-dir", type=str, default="./qwen35_t4_sweeps")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--max-steps-per-run", type=int, default=120)
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=0,
        help="0 means use all rows after normalization. Set e.g. 1000-2000 for faster sweeps.",
    )
    parser.add_argument(
        "--max-per-dataset",
        type=int,
        default=0,
        help="0 means no per-dataset cap. Useful for balancing quick experiments.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="HF datasets to load and normalize.",
    )
    parser.add_argument(
        "--sweep-json",
        type=str,
        default="",
        help="Optional path to JSON file containing a list of run configs.",
    )
    parser.add_argument(
        "--report-to",
        type=str,
        default="none",
        help='Trainer reporting target, e.g. "none", "wandb".',
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
    # Common prompt keys.
    prompt = _first_non_empty(
        example,
        [
            "problem",
            "prompt",
            "question",
            "instruction",
            "input",
            "query",
            "task",
        ],
    )

    # Try direct reasoning + answer fields first.
    thinking = _first_non_empty(
        example,
        [
            "thinking",
            "reasoning",
            "cot",
            "chain_of_thought",
            "chain_of_thoughts",
            "rationale",
        ],
    )
    answer = _first_non_empty(
        example,
        [
            "solution",
            "answer",
            "output",
            "response",
            "assistant",
            "completion",
            "final_answer",
        ],
    )

    # Try conversation-like formats.
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

    # If assistant already includes think tags, keep it as-is.
    if assistant and _has_think_tags(assistant):
        return {"prompt": prompt.strip(), "assistant": assistant.strip()}

    # If no think tags but we have explicit thinking, attach it.
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


def load_sweep(path: str) -> list[dict[str, Any]]:
    if not path:
        return DEFAULT_SWEEP
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError("Sweep JSON must be a non-empty list of run config objects.")
    return data


def make_sft_config(max_seq_length: int, **kwargs: Any) -> SFTConfig:
    """
    Build SFTConfig compatibly across TRL versions:
    - newer variants may expect `max_seq_length`
    - others use `max_length`
    Also drops unsupported optional kwargs for older versions.
    """
    sig = inspect.signature(SFTConfig.__init__)
    params = sig.parameters

    cfg_kwargs = dict(kwargs)
    if "max_seq_length" in params:
        cfg_kwargs["max_seq_length"] = max_seq_length
    elif "max_length" in params:
        cfg_kwargs["max_length"] = max_seq_length
    else:
        print("[warn] Neither max_seq_length nor max_length is supported by this TRL SFTConfig.")

    filtered = {k: v for k, v in cfg_kwargs.items() if k in params}
    dropped = sorted(set(cfg_kwargs) - set(filtered))
    if dropped:
        print(f"[warn] Dropping unsupported SFTConfig args for this TRL version: {dropped}")
    return SFTConfig(**filtered)


def train_one_run(
    run_cfg: dict[str, Any],
    args: argparse.Namespace,
    train_dataset: Dataset,
    out_dir: Path,
) -> RunSummary:
    run_name = run_cfg["name"]
    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 80}\nStarting run: {run_name}\nConfig: {json.dumps(run_cfg, indent=2)}\n{'=' * 80}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_id,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        load_in_16bit=False,
        full_finetuning=False,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=int(run_cfg["lora_r"]),
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        lora_alpha=int(run_cfg["lora_alpha"]),
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        max_seq_length=args.max_seq_length,
        use_rslora=False,
        loftq_config=None,
    )

    supports_bf16 = torch.cuda.is_bf16_supported()
    sft_cfg = make_sft_config(
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
        per_device_train_batch_size=int(run_cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(run_cfg["gradient_accumulation_steps"]),
        warmup_steps=args.warmup_steps,
        num_train_epochs=float(run_cfg.get("num_train_epochs", 1.0)),
        max_steps=args.max_steps_per_run if args.max_steps_per_run > 0 else -1,
        learning_rate=float(run_cfg["learning_rate"]),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=args.weight_decay,
        lr_scheduler_type="linear",
        fp16=not supports_bf16,
        bf16=supports_bf16,
        seed=args.seed,
        output_dir=str(run_dir / "trainer"),
        report_to=args.report_to,
        save_strategy="no",
        dataset_num_proc=1,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=sft_cfg,
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    print("Training...")
    train_output = trainer.train()

    adapter_dir = run_dir / "lora_adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving LoRA adapter to {adapter_dir}")
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    peak_memory_gb = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    summary = RunSummary(
        run_name=run_name,
        model_id=args.model_id,
        dataset_rows=len(train_dataset),
        learning_rate=float(run_cfg["learning_rate"]),
        lora_r=int(run_cfg["lora_r"]),
        lora_alpha=int(run_cfg["lora_alpha"]),
        per_device_train_batch_size=int(run_cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(run_cfg["gradient_accumulation_steps"]),
        effective_batch_size=int(run_cfg["per_device_train_batch_size"])
        * int(run_cfg["gradient_accumulation_steps"]),
        num_train_epochs=float(run_cfg.get("num_train_epochs", 1.0)),
        max_steps=args.max_steps_per_run,
        max_seq_length=args.max_seq_length,
        train_runtime_seconds=train_output.metrics.get("train_runtime"),
        train_loss=train_output.metrics.get("train_loss"),
        peak_reserved_memory_gb=peak_memory_gb,
    )

    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2)

    # Clean up aggressively between sweep runs.
    del trainer
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return summary


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available. This script is intended for Colab T4.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using model: {args.model_id}")
    print(f"Output dir: {out_dir.resolve()}")
    print(f"Datasets: {args.datasets}")

    # Load tokenizer once for chat formatting. It must match training model family.
    _, tokenizer_for_format = FastLanguageModel.from_pretrained(
        model_name=args.model_id,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        load_in_16bit=False,
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

    # Save a preview for inspection.
    preview_path = out_dir / "normalized_preview.jsonl"
    with open(preview_path, "w", encoding="utf-8") as f:
        for i in range(min(5, len(train_dataset))):
            row = train_dataset[i]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote preview examples to: {preview_path}")

    sweep = load_sweep(args.sweep_json)
    print(f"\nSweep runs: {[cfg['name'] for cfg in sweep]}")

    summaries: list[RunSummary] = []
    for cfg in sweep:
        summary = train_one_run(
            run_cfg=cfg,
            args=args,
            train_dataset=train_dataset,
            out_dir=out_dir,
        )
        summaries.append(summary)

    # Persist aggregate leaderboard-style output.
    all_summaries = [asdict(s) for s in summaries]
    with open(out_dir / "all_runs_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)

    sorted_runs = sorted(
        all_summaries,
        key=lambda x: (x["train_loss"] if x["train_loss"] is not None else float("inf")),
    )
    with open(out_dir / "sorted_by_loss.json", "w", encoding="utf-8") as f:
        json.dump(sorted_runs, f, indent=2)

    print("\nAll runs completed.")
    print(f"Summary file: {out_dir / 'all_runs_summary.json'}")
    print(f"Sorted file : {out_dir / 'sorted_by_loss.json'}")
    print("\nTip: If OOM occurs, use --model-id unsloth/Qwen3.5-2B and/or lower max-seq-length.")


if __name__ == "__main__":
    # Optional: let users pass token via env in Colab for private assets.
    if "HF_TOKEN" in os.environ:
        print("HF_TOKEN detected in environment.")
    main()
