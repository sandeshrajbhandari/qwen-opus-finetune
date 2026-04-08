"""Local LoRA finetune script for Qwen/Qwen3.5-0.8B.

This mirrors the same dataset and core training defaults used in
`modal_qwen35_18b_reap_a3b_coding_opus_train.py`, but runs locally.

Example:
    python 0.8b-finetune-tests/finetune_qwen35_08b_local.py
    python 0.8b-finetune-tests/finetune_qwen35_08b_local.py --max-steps 20 --limit-samples 256
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import unsloth  # noqa: F401  # Must be imported before trl/transformers.
import torch
from datasets import load_dataset
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only


BASE_MODEL = "Qwen/Qwen3.5-0.8B"
DATASET_NAME = "nohurry/Opus-4.6-Reasoning-3000x-filtered"
DATASET_SPLIT = "train"
DATASET_REVISION = "main"

# Same defaults as the 18B Modal script.
DEFAULT_MAX_SEQ_LENGTH = 8192
DEFAULT_EPOCHS = 2.0
DEFAULT_PER_DEVICE_BATCH_SIZE = 6
DEFAULT_GRADIENT_ACCUM_STEPS = 4
DEFAULT_LEARNING_RATE = 6e-5
DEFAULT_LORA_R = 32
DEFAULT_LORA_ALPHA = 64
DEFAULT_WARMUP_STEPS = 2
DEFAULT_SEED = 3407
DEFAULT_SAVE_STEPS = 20


class LossLoggerCallback(TrainerCallback):
    """Persist per-step losses and print 20-step summaries."""

    def __init__(self, log_path: Path, report_every: int = 20):
        self.log_path = log_path
        self.report_every = report_every
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text("", encoding="utf-8")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        if "loss" not in logs:
            return
        step = int(state.global_step)
        row = {
            "step": step,
            "epoch": float(logs.get("epoch", state.epoch or 0.0)),
            "loss": float(logs["loss"]),
            "learning_rate": float(logs.get("learning_rate", 0.0)),
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        if step > 0 and step % self.report_every == 0:
            print(
                f"[loss-report] step={row['step']} epoch={row['epoch']:.3f} "
                f"loss={row['loss']:.6f} lr={row['learning_rate']:.8f}"
            )


def resolve_checkpoint_path(base_dir: Path, resume_from_checkpoint: str) -> str | None:
    if not resume_from_checkpoint:
        return None
    if resume_from_checkpoint == "latest":
        checkpoints = sorted(
            base_dir.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
        )
        return str(checkpoints[-1]) if checkpoints else None
    path = Path(resume_from_checkpoint)
    return str(path if path.is_absolute() else (base_dir / path))


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=Path,
        default=script_dir / "Qwen3.5-0.8B",
        help="Local model path. Falls back to Hugging Face repo ID if missing.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "outputs" / "qwen35_08b_opus_local",
    )
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument("--epochs", type=float, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_PER_DEVICE_BATCH_SIZE)
    parser.add_argument("--grad-accum", type=int, default=DEFAULT_GRADIENT_ACCUM_STEPS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument("--lora-r", type=int, default=DEFAULT_LORA_R)
    parser.add_argument("--lora-alpha", type=int, default=DEFAULT_LORA_ALPHA)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--save-steps", type=int, default=DEFAULT_SAVE_STEPS)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default="",
        help='Checkpoint path, relative checkpoint dir name, or "latest".',
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        default=False,
        help="Use 4-bit loading for lower VRAM usage.",
    )
    parser.add_argument(
        "--no-load-in-4bit",
        dest="load_in_4bit",
        action="store_false",
    )
    parser.add_argument("--save-merged-16bit", action="store_true")
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

    model_load_path = str(args.model_path) if args.model_path.exists() else BASE_MODEL
    if args.model_path.exists():
        print(f"Loading local model from: {args.model_path}")
    else:
        print(f"Local model path not found. Falling back to: {BASE_MODEL}")

    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_load_path,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        load_in_16bit=not args.load_in_4bit,
        full_finetuning=False,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=True,
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
            max_steps=args.max_steps,
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
            save_strategy="steps",
            save_steps=args.save_steps,
            save_total_limit=3,
        ),
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n<think>",
    )
    loss_log_path = args.output_dir / "loss_log.jsonl"
    trainer.add_callback(LossLoggerCallback(loss_log_path, report_every=20))
    resolved_resume_checkpoint = resolve_checkpoint_path(
        trainer_dir,
        args.resume_from_checkpoint,
    )
    if args.resume_from_checkpoint and not resolved_resume_checkpoint:
        raise FileNotFoundError(
            f"Could not resolve checkpoint from '{args.resume_from_checkpoint}' under {trainer_dir}"
        )
    if resolved_resume_checkpoint:
        print(f"Resuming from checkpoint: {resolved_resume_checkpoint}")

    print("Training...")
    trainer_stats = trainer.train(resume_from_checkpoint=resolved_resume_checkpoint)

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
        "model_load_path": model_load_path,
        "dataset_name": DATASET_NAME,
        "dataset_split": DATASET_SPLIT,
        "dataset_revision": DATASET_REVISION,
        "dataset_rows": len(converted_dataset),
        "max_seq_length": args.max_seq_length,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "resume_from_checkpoint": resolved_resume_checkpoint,
        "save_steps": args.save_steps,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "world_size": world_size,
        "global_effective_batch_size": args.batch_size * args.grad_accum * world_size,
        "learning_rate": args.learning_rate,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "warmup_steps": args.warmup_steps,
        "load_in_4bit": args.load_in_4bit,
        "load_in_16bit": not args.load_in_4bit,
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
