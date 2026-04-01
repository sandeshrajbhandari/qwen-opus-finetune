"""Fine-tune Qwen3.5-4B on the full Opus distilled dataset with Modal + Unsloth.

Usage:
    modal run modal_qwen35_4b_opus_train.py
    modal run modal_qwen35_4b_opus_train.py --epochs 2 --max-seq-length 4096
    modal run modal_qwen35_4b_opus_train.py --hf-repo-id your-name/qwen35-4b-opus-3000x
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
APP_NAME = "qwen35-4b-opus-sft"
BASE_MODEL = "unsloth/Qwen3.5-4B"
DATASET_NAME = "nohurry/Opus-4.6-Reasoning-3000x-filtered"
DATASET_SPLIT = "train"
DATASET_REVISION = "main"
OUTPUT_ROOT = "/outputs"
HF_CACHE_DIR = "/root/.cache/huggingface"

DEFAULT_MAX_SEQ_LENGTH = 2048
DEFAULT_EPOCHS = 1
DEFAULT_PER_DEVICE_BATCH_SIZE = 2
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 8
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 16
DEFAULT_WARMUP_STEPS = 10
DEFAULT_SEED = 3407


# ---------------------------------------------------------------------------
# Modal resources
# ---------------------------------------------------------------------------
app = modal.App(APP_NAME)

hf_cache_volume = modal.Volume.from_name("hf-cache-qwen35-opus", create_if_missing=True)
outputs_volume = modal.Volume.from_name("qwen35-opus-outputs", create_if_missing=True)

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


def make_run_name(max_seq_length: int, epochs: float) -> str:
    return f"qwen35-4b-opus3000x-msl{max_seq_length}-e{str(epochs).replace('.', '_')}"


@app.function(
    image=image,
    gpu="A10G",
    timeout=12 * MINUTES,
    volumes={
        HF_CACHE_DIR: hf_cache_volume,
        OUTPUT_ROOT: outputs_volume,
    },
    secrets=[huggingface_secret],
)
def train(
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    epochs: float = DEFAULT_EPOCHS,
    per_device_train_batch_size: int = DEFAULT_PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps: int = DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    lora_r: int = DEFAULT_LORA_R,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
    warmup_steps: int = DEFAULT_WARMUP_STEPS,
    seed: int = DEFAULT_SEED,
    hf_repo_id: str = "",
    save_merged_16bit: bool = False,
    report_to: str = "none",
):
    import torch
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only

    run_name = make_run_name(max_seq_length=max_seq_length, epochs=epochs)
    run_output_dir = Path(OUTPUT_ROOT) / run_name
    lora_output_dir = run_output_dir / "lora"
    run_output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading base model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        finetune_vision_layers     = False, # False if not finetuning vision layers
        finetune_language_layers   = True, # False if not finetuning language layers
        finetune_attention_modules = True, # False if not finetuning attention layers
        finetune_mlp_modules       = True, # False if not finetuning MLP layers
        lora_alpha=lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
        max_seq_length=max_seq_length,
        use_rslora=False,
        loftq_config=None,
    )

    print("Loading dataset...")
    dataset = load_dataset(
        DATASET_NAME,
        split=DATASET_SPLIT,
        revision=DATASET_REVISION,
    )

    def convert_to_conversation(sample):
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

    converted_dataset = dataset.map(
        convert_to_conversation,
        remove_columns=dataset.column_names,
        desc="Formatting dataset into Qwen chat examples",
    )

    gpu_stats = torch.cuda.get_device_properties(0)
    supports_bf16 = torch.cuda.is_bf16_supported()
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"Dataset rows = {len(dataset)}")
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved before training.")
    print(f"bf16 supported = {supports_bf16}")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=converted_dataset,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            fp16=not supports_bf16,
            bf16=supports_bf16,
            seed=seed,
            output_dir=str(run_output_dir / "trainer"),
            report_to=report_to,
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

    print("Starting training...")
    trainer_stats = trainer.train()

    print("Saving LoRA adapter...")
    model.save_pretrained(str(lora_output_dir))
    tokenizer.save_pretrained(str(lora_output_dir))

    if save_merged_16bit:
        merged_dir = run_output_dir / "merged_16bit"
        print("Saving merged 16-bit model...")
        model.save_pretrained_merged(
            str(merged_dir),
            tokenizer,
            save_method="merged_16bit",
        )

    if hf_repo_id:
        print(f"Uploading LoRA adapter to Hugging Face: {hf_repo_id}")
        model.push_to_hub_merged(
            hf_repo_id,
            tokenizer,
            save_method="lora",
            token=os.environ["HF_TOKEN"],
        )

    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_training = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    training_percentage = round(used_memory_for_training / max_memory * 100, 3)

    summary = {
        "base_model": BASE_MODEL,
        "dataset_name": DATASET_NAME,
        "dataset_split": DATASET_SPLIT,
        "dataset_revision": DATASET_REVISION,
        "dataset_rows": len(dataset),
        "max_seq_length": max_seq_length,
        "epochs": epochs,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": per_device_train_batch_size * gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "output_dir": str(run_output_dir),
        "train_runtime_seconds": trainer_stats.metrics.get("train_runtime"),
        "train_loss": trainer_stats.metrics.get("train_loss"),
        "peak_reserved_memory_gb": used_memory,
        "peak_training_memory_gb": used_memory_for_training,
        "peak_reserved_memory_pct": used_percentage,
        "peak_training_memory_pct": training_percentage,
    }

    summary_path = run_output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    outputs_volume.commit()
    hf_cache_volume.commit()

    print(json.dumps(summary, indent=2))
    return summary


@app.local_entrypoint()
def main(
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    epochs: float = DEFAULT_EPOCHS,
    per_device_train_batch_size: int = DEFAULT_PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps: int = DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    lora_r: int = DEFAULT_LORA_R,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
    warmup_steps: int = DEFAULT_WARMUP_STEPS,
    seed: int = DEFAULT_SEED,
    hf_repo_id: str = "",
    save_merged_16bit: bool = False,
    report_to: str = "none",
):
    train.remote(
        max_seq_length=max_seq_length,
        epochs=epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        warmup_steps=warmup_steps,
        seed=seed,
        hf_repo_id=hf_repo_id,
        save_merged_16bit=save_merged_16bit,
        report_to=report_to,
    )
