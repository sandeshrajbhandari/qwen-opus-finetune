"""Single-file Qwen3.5-4B LoRA fine-tuning script for 2x 16 GB GPUs.

This keeps the same dataset and chat formatting as the main training script,
but uses lighter defaults so it can fit on a pair of smaller local GPUs.

Example:
    torchrun --nproc_per_node=2 train_qwen35_4b_16gb.py
    torchrun --nproc_per_node=2 train_qwen35_4b_16gb.py --max-seq-length 1024 --grad-accum 2 --epochs 1
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import unsloth  # noqa: F401  # Must come before trl/transformers imports.
import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only


BASE_MODEL = "unsloth/Qwen3.5-4B"
DATASET_NAME = "nohurry/Opus-4.6-Reasoning-3000x-filtered"
DATASET_SPLIT = "train"
DATASET_REVISION = "main"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/qwen35_4b_2x16gb"))
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--save-merged-16bit", action="store_true")
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        default=False,
        help="Use 4-bit base model loading for tighter VRAM budgets.",
    )
    parser.add_argument(
        "--no-load-in-4bit",
        dest="load_in_4bit",
        action="store_false",
        help="Disable 4-bit loading and try 16-bit weights instead.",
    )
    parser.add_argument(
        "--limit-samples",
        type=int,
        default=None,
        help="Optional cap for quick test runs.",
    )
    return parser.parse_args()


def format_example(tokenizer, sample: dict) -> dict:
    assistant_text = (
        f"<think>\n{sample['thinking'].strip()}\n</think>\n\n{sample['solution'].strip()}"
    )
    messages = [
        {"role": "user", "content": sample["problem"].strip()},
        {"role": "assistant", "content": assistant_text},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lora_dir = args.output_dir / "lora"
    trainer_dir = args.output_dir / "trainer"
    world_size = int(torch.cuda.device_count()) if torch.cuda.is_available() else 1

    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        load_in_16bit=not args.load_in_4bit,
        full_finetuning=False,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        max_seq_length=args.max_seq_length,
        use_rslora=False,
        loftq_config=None,
    )

    print("Loading dataset...")
    dataset = load_dataset(
        DATASET_NAME,
        split=DATASET_SPLIT,
        revision=DATASET_REVISION,
    )
    if args.limit_samples is not None:
        dataset = dataset.select(range(min(args.limit_samples, len(dataset))))

    converted_dataset = dataset.map(
        lambda sample: format_example(tokenizer, sample),
        remove_columns=dataset.column_names,
        desc="Formatting dataset into Qwen chat examples",
    )

    supports_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        total_memory_gb = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 2)
        print(f"GPU: {gpu_stats.name} ({total_memory_gb} GB)")
        print(f"Visible GPUs: {world_size}")
    print(f"Dataset rows: {len(converted_dataset)}")
    print(
        f"Per-device batch size: {args.batch_size}, grad accum: {args.grad_accum}, "
        f"global effective batch size: {args.batch_size * args.grad_accum * world_size}"
    )
    print(f"4-bit loading: {args.load_in_4bit}")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=converted_dataset,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=args.max_seq_length,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            warmup_steps=args.warmup_steps,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            fp16=not supports_bf16,
            bf16=supports_bf16,
            seed=args.seed,
            output_dir=str(trainer_dir),
            report_to="none",
            dataset_num_proc=1,
            save_strategy="epoch",
            save_total_limit=2,
        ),
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    print("Training...")
    trainer_stats = trainer.train()

    print("Saving LoRA adapter...")
    model.save_pretrained(str(lora_dir))
    tokenizer.save_pretrained(str(lora_dir))

    if args.save_merged_16bit:
        merged_dir = args.output_dir / "merged_16bit"
        print("Saving merged 16-bit model...")
        model.save_pretrained_merged(
            str(merged_dir),
            tokenizer,
            save_method="merged_16bit",
        )

    summary = {
        "base_model": BASE_MODEL,
        "dataset_name": DATASET_NAME,
        "dataset_rows": len(converted_dataset),
        "max_seq_length": args.max_seq_length,
        "epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "world_size": world_size,
        "global_effective_batch_size": args.batch_size * args.grad_accum * world_size,
        "learning_rate": args.learning_rate,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "load_in_4bit": args.load_in_4bit,
        "train_runtime_seconds": trainer_stats.metrics.get("train_runtime"),
        "train_loss": trainer_stats.metrics.get("train_loss"),
        "output_dir": str(args.output_dir.resolve()),
    }
    summary_path = args.output_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
