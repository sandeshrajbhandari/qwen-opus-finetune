"""Fine-tune the pruned multimodal Qwen3.5-24B-A10B model on the Opus dataset.

Usage:
    modal run modal_qwen35_24b_a10b_opus_train.py
    modal run modal_qwen35_24b_a10b_opus_train.py --per-device-train-batch-size 1 --gradient-accumulation-steps 8
    modal run modal_qwen35_24b_a10b_opus_train.py --hf-repo-id your-name/qwen35-24b-a10b-opus
"""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from pathlib import Path

import modal


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
APP_NAME = "qwen35-24b-a10b-opus-sft"
BASE_MODEL = "sandeshrajx/qwen3.5b-24b-a10b"
DATASET_NAME = "nohurry/Opus-4.6-Reasoning-3000x-filtered"
DATASET_SPLIT = "train"
DATASET_REVISION = "main"
OUTPUT_ROOT = "/outputs"
HF_CACHE_DIR = "/root/.cache/huggingface"
MODEL_SNAPSHOT_DIR = f"{HF_CACHE_DIR}/preloaded/qwen35_24b_a10b"
TEXT_ONLY_MODEL_SNAPSHOT_DIR = f"{HF_CACHE_DIR}/preloaded/qwen35_24b_a10b_text_only"
SKIP_WORKING_COPY_FILENAMES = {
    "model.safetensors",
    "stale_model_stale.safetensors",
}

DEFAULT_MAX_SEQ_LENGTH = 8192
DEFAULT_EPOCHS = 4
DEFAULT_PER_DEVICE_BATCH_SIZE = 2
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 4
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_LORA_R = 32
DEFAULT_LORA_ALPHA = 64 
DEFAULT_WARMUP_STEPS = 2
DEFAULT_SEED = 3407


# ---------------------------------------------------------------------------
# Modal resources
# ---------------------------------------------------------------------------
app = modal.App(APP_NAME)

hf_cache_volume = modal.Volume.from_name("hf-cache-qwen35-24b-a10b", create_if_missing=True)
outputs_volume = modal.Volume.from_name("qwen35-24b-a10b-opus-outputs", create_if_missing=True)

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
        "datasets>=3.6.0,<4.0.0",
        "accelerate>=1.7.0",
        "huggingface-hub>=0.34.0",
        "sentencepiece",
        "protobuf",
        "wandb>=0.21.1",
    )
    .env({
        "HF_XET_HIGH_PERFORMANCE": "1",
        "UNSLOTH_MOE_DISABLE_AUTOTUNE": "1",
    })
)

MINUTES = 60


def make_run_name(max_seq_length: int, epochs: float) -> str:
    return f"qwen35-24b-a10b-opus-msl{max_seq_length}-e{str(epochs).replace('.', '_')}"


def resolve_run_output_dir(
    output_root: Path,
    max_seq_length: int,
    epochs: float,
    output_run_name: str,
) -> Path:
    run_name = output_run_name or make_run_name(max_seq_length=max_seq_length, epochs=epochs)
    return output_root / run_name


def resolve_checkpoint_path(base_dir: Path, resume_from_checkpoint: str) -> str | None:
    if not resume_from_checkpoint:
        return None

    if resume_from_checkpoint == "latest":
        checkpoints = sorted(
            base_dir.glob("checkpoint-*"),
            key=lambda path: int(path.name.split("-")[-1]) if path.name.split("-")[-1].isdigit() else -1,
        )
        return str(checkpoints[-1]) if checkpoints else None

    checkpoint_path = Path(resume_from_checkpoint)
    if checkpoint_path.is_absolute():
        return str(checkpoint_path)
    return str(base_dir / resume_from_checkpoint)


def patch_tokenizer_files(snapshot_path: Path) -> dict:
    tokenizer_config_path = snapshot_path / "tokenizer_config.json"
    special_tokens_map_path = snapshot_path / "special_tokens_map.json"

    result = {
        "tokenizer_config_path": str(tokenizer_config_path),
        "special_tokens_map_path": str(special_tokens_map_path),
        "patched": False,
    }

    if not tokenizer_config_path.exists():
        return result

    tokenizer_config = json.loads(tokenizer_config_path.read_text())

    eos_token = tokenizer_config.get("eos_token")
    if isinstance(eos_token, dict):
        eos_content = eos_token.get("content")
    else:
        eos_content = eos_token

    if eos_content == "<EOS_TOKEN>":
        tokenizer_config["eos_token"] = "<|im_end|>"
        if tokenizer_config.get("pad_token") in (None, "<EOS_TOKEN>"):
            tokenizer_config["pad_token"] = "<|im_end|>"
        tokenizer_config_path.write_text(json.dumps(tokenizer_config, indent=2))
        result["patched"] = True

    if special_tokens_map_path.exists():
        special_tokens_map = json.loads(special_tokens_map_path.read_text())
        if special_tokens_map.get("eos_token") == "<EOS_TOKEN>":
            special_tokens_map["eos_token"] = "<|im_end|>"
            if special_tokens_map.get("pad_token") in (None, "<EOS_TOKEN>"):
                special_tokens_map["pad_token"] = "<|im_end|>"
            special_tokens_map_path.write_text(json.dumps(special_tokens_map, indent=2))
            result["patched"] = True

    return result


def prepare_text_only_model_copy(source_path: Path, target_path: Path) -> dict:
    config_path = source_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json under {source_path}")

    if target_path.exists():
        shutil.rmtree(target_path)
    target_path.mkdir(parents=True, exist_ok=True)
    for item in source_path.iterdir():
        if item.name in SKIP_WORKING_COPY_FILENAMES:
            continue
        destination = target_path / item.name
        if item.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(item, destination)
        else:
            shutil.copy2(item, destination)

    target_config_path = target_path / "config.json"
    source_config = json.loads(target_config_path.read_text(encoding="utf-8"))

    removed_keys = [
        key
        for key in [
            "vision_config",
            "image_token_id",
            "video_token_id",
            "vision_start_token_id",
            "vision_end_token_id",
        ]
        if key in source_config
    ]

    text_config = json.loads(json.dumps(source_config.get("text_config", {})))
    text_config["model_type"] = "qwen3_5_moe"
    text_config["num_experts"] = 39
    text_config["architectures"] = ["Qwen3_5MoeForCausalLM"]
    if "transformers_version" in source_config:
        text_config["transformers_version"] = source_config["transformers_version"]
    text_config["tie_word_embeddings"] = source_config.get("tie_word_embeddings", False)

    target_config_path.write_text(json.dumps(text_config, indent=2) + "\n", encoding="utf-8")
    patch_tokenizer_files(target_path)

    return {
        "source_model_snapshot_dir": str(source_path),
        "text_only_model_snapshot_dir": str(target_path),
        "removed_multimodal_keys": removed_keys,
        "text_architecture": text_config["architectures"][0],
        "num_experts": text_config["num_experts"],
    }


@app.function(
    image=image,
    volumes={HF_CACHE_DIR: hf_cache_volume},
    secrets=[huggingface_secret],
    timeout=20 * MINUTES,
)
def download_model():
    from huggingface_hub import snapshot_download

    snapshot_path = Path(MODEL_SNAPSHOT_DIR)
    config_path = snapshot_path / "config.json"
    if not config_path.exists():
        print(f"Downloading model snapshot for {BASE_MODEL} to {snapshot_path} ...")
        snapshot_download(
            repo_id=BASE_MODEL,
            local_dir=str(snapshot_path),
            local_dir_use_symlinks=False,
        )
        hf_cache_volume.commit()
        print(f"Cached model snapshot at {snapshot_path}")

    tokenizer_patch_summary = patch_tokenizer_files(snapshot_path)
    hf_cache_volume.commit()
    print(
        json.dumps(
            {
                "model_snapshot_dir": str(snapshot_path),
                "base_model": BASE_MODEL,
                **tokenizer_patch_summary,
            },
            indent=2,
        )
    )
    return str(snapshot_path)


@app.function(
    image=image,
    volumes={HF_CACHE_DIR: hf_cache_volume},
    timeout=10 * MINUTES,
)
def prepare_text_only_snapshot():
    source_path = Path(MODEL_SNAPSHOT_DIR)
    if not (source_path / "config.json").exists():
        raise FileNotFoundError(f"Missing config.json under {source_path}")
    result = prepare_text_only_model_copy(
        source_path=source_path,
        target_path=Path(TEXT_ONLY_MODEL_SNAPSHOT_DIR),
    )
    hf_cache_volume.commit()
    print(json.dumps(result, indent=2))
    return result


@app.function(
    image=image,
    volumes={HF_CACHE_DIR: hf_cache_volume},
    timeout=10 * MINUTES,
)
def validate_snapshot():
    snapshot_path = Path(MODEL_SNAPSHOT_DIR)
    if not (snapshot_path / "config.json").exists():
        raise FileNotFoundError(f"Missing config.json under {snapshot_path}")
    tokenizer_patch_summary = patch_tokenizer_files(snapshot_path)
    result = {
        "model_snapshot_dir": str(snapshot_path),
        "base_model": BASE_MODEL,
        "has_config": True,
        "has_preprocessor_config": (snapshot_path / "preprocessor_config.json").exists(),
        "has_chat_template": (snapshot_path / "chat_template.jinja").exists(),
        **tokenizer_patch_summary,
    }
    print(json.dumps(result, indent=2))
    return result


@app.function(
    image=image,
    volumes={HF_CACHE_DIR: hf_cache_volume},
    timeout=10 * MINUTES,
)
def validate_text_only_snapshot():
    snapshot_path = Path(TEXT_ONLY_MODEL_SNAPSHOT_DIR)
    if not (snapshot_path / "config.json").exists():
        raise FileNotFoundError(f"Missing config.json under {snapshot_path}")
    config = json.loads((snapshot_path / "config.json").read_text(encoding="utf-8"))
    result = {
        "model_snapshot_dir": str(snapshot_path),
        "base_model": BASE_MODEL,
        "has_config": True,
        "has_preprocessor_config": (snapshot_path / "preprocessor_config.json").exists(),
        "has_chat_template": (snapshot_path / "chat_template.jinja").exists(),
        "has_vision_config": "vision_config" in config,
        "architectures": config.get("architectures"),
        "text_model_type": config.get("model_type"),
        "num_experts": config.get("num_experts"),
    }
    print(json.dumps(result, indent=2))
    return result


@app.function(
    image=image,
    gpu=["A100-80GB"],
    timeout=180 * MINUTES,
    volumes={
        HF_CACHE_DIR: hf_cache_volume,
        OUTPUT_ROOT: outputs_volume,
    },
    secrets=[huggingface_secret],
)
def train(
    dataset_name: str = DATASET_NAME,
    dataset_split: str = DATASET_SPLIT,
    dataset_revision: str = DATASET_REVISION,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    epochs: float = DEFAULT_EPOCHS,
    per_device_train_batch_size: int = DEFAULT_PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps: int = DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    lora_r: int = DEFAULT_LORA_R,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
    warmup_steps: int = DEFAULT_WARMUP_STEPS,
    seed: int = DEFAULT_SEED,
    max_steps: int = -1,
    resume_from_checkpoint: str = "",
    save_steps: int = 20,
    hf_repo_id: str = "",
    save_merged_16bit: bool = False,
    report_to: str = "none",
    continue_from_run_name: str = "",
    output_run_name: str = "",
    strip_vision_for_training: bool = False,
):
    import torch
    import unsloth
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only

    run_output_dir = resolve_run_output_dir(
        output_root=Path(OUTPUT_ROOT),
        max_seq_length=max_seq_length,
        epochs=epochs,
        output_run_name=output_run_name,
    )
    run_name = run_output_dir.name
    lora_output_dir = run_output_dir / "lora"
    trainer_output_dir = run_output_dir / "trainer"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    continued_lora_dir = (
        Path(OUTPUT_ROOT) / continue_from_run_name / "lora"
        if continue_from_run_name
        else None
    )
    if strip_vision_for_training and continued_lora_dir is None:
        text_only_snapshot_path = Path(TEXT_ONLY_MODEL_SNAPSHOT_DIR)
        if not (text_only_snapshot_path / "config.json").exists():
            print("Preparing text-only snapshot for training...")
            prepare_text_only_model_copy(
                source_path=Path(MODEL_SNAPSHOT_DIR),
                target_path=text_only_snapshot_path,
            )
            hf_cache_volume.commit()

    job_start_time = time.perf_counter()

    model_load_path = (
        str(continued_lora_dir)
        if continued_lora_dir is not None
        else TEXT_ONLY_MODEL_SNAPSHOT_DIR
        if strip_vision_for_training and Path(TEXT_ONLY_MODEL_SNAPSHOT_DIR).exists()
        else MODEL_SNAPSHOT_DIR if Path(MODEL_SNAPSHOT_DIR).exists() else BASE_MODEL
    )
    if continued_lora_dir is not None and not continued_lora_dir.exists():
        raise FileNotFoundError(f"Continuation LoRA directory not found: {continued_lora_dir}")

    print("Loading base model...")
    model, processor = FastLanguageModel.from_pretrained(
        model_load_path,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        fast_inference=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_load_path,
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

    if continued_lora_dir is None:
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_r,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj", "gate_up_proj",
            ],
            lora_alpha=lora_r * 2,
            use_gradient_checkpointing=True,
            random_state=seed,
            bias="none",
        )
    else:
        if hasattr(FastLanguageModel, "for_training"):
            FastLanguageModel.for_training(model)
        if hasattr(model, "enable_adapter_layers"):
            model.enable_adapter_layers()
        print(f"Continuing training from saved adapter at {continued_lora_dir}")

    print("Loading dataset...")
    dataset = load_dataset(
        dataset_name,
        split=dataset_split,
        revision=dataset_revision,
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
    print(f"Run name = {run_name}")
    print(f"Dataset rows = {len(dataset)}")
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved before training.")
    print(f"bf16 supported = {supports_bf16}")

    effective_batch_size = per_device_train_batch_size * gradient_accumulation_steps
    steps_per_epoch = math.ceil(len(dataset) / effective_batch_size)
    target_full_train_steps = math.ceil(steps_per_epoch * epochs)
    use_max_steps = max_steps is not None and max_steps > 0
    actual_train_steps_target = min(max_steps, target_full_train_steps) if use_max_steps else target_full_train_steps
    setup_seconds = time.perf_counter() - job_start_time

    print(f"Effective batch size = {effective_batch_size}")
    print(f"Steps per epoch = {steps_per_epoch}")
    print(f"Target full-train steps = {target_full_train_steps}")
    if use_max_steps:
        print(f"Short run enabled with max_steps = {max_steps}")

    resolved_resume_checkpoint = resolve_checkpoint_path(
        trainer_output_dir,
        resume_from_checkpoint,
    )
    if resume_from_checkpoint and not resolved_resume_checkpoint:
        raise FileNotFoundError(
            f"Could not find checkpoint to resume from under {trainer_output_dir} using '{resume_from_checkpoint}'."
        )
    if resolved_resume_checkpoint:
        print(f"Resuming from checkpoint: {resolved_resume_checkpoint}")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=converted_dataset,
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
            weight_decay=0.001,
            lr_scheduler_type="linear",
            fp16=not supports_bf16,
            bf16=supports_bf16,
            seed=seed,
            output_dir=str(trainer_output_dir),
            report_to=report_to,
            dataset_num_proc=1,
            save_strategy="steps",
            save_steps=save_steps,
            save_total_limit=1,
        ),
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n<think>",
    )

    print("\n" + "=" * 60)
    print("VERIFICATION: Checking that train_on_responses_only worked...")
    print("=" * 60)

    sample_index = 0
    full_input_decoded = tokenizer.decode(trainer.train_dataset[sample_index]["input_ids"])
    print(f"\n[FULL RAW INPUT (Index {sample_index})] - The model sees all of this:")
    print("-" * 40)
    print(full_input_decoded)
    print("-" * 40)

    raw_labels = trainer.train_dataset[sample_index]["labels"]
    masked_labels_decoded = tokenizer.decode(
        [tokenizer.pad_token_id if x == -100 else x for x in raw_labels]
    ).replace(tokenizer.pad_token, "")
    print(f"\n[TRAINABLE LABELS (Index {sample_index})] - The model is ONLY trained on this:")
    print("-" * 40)
    print(masked_labels_decoded)
    print("-" * 40)
    print("If the above block only shows the `<think>...` block and the answer, it worked!\n")

    print("Starting training...")
    trainer_stats = trainer.train(resume_from_checkpoint=resolved_resume_checkpoint)

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
    train_runtime_seconds = trainer_stats.metrics.get("train_runtime")
    observed_steps = trainer_stats.metrics.get("global_step", actual_train_steps_target)
    seconds_per_step = (
        train_runtime_seconds / observed_steps
        if train_runtime_seconds and observed_steps
        else None
    )
    estimated_full_train_runtime_seconds = (
        seconds_per_step * target_full_train_steps
        if seconds_per_step is not None
        else None
    )
    estimated_full_job_runtime_seconds = (
        setup_seconds + estimated_full_train_runtime_seconds
        if estimated_full_train_runtime_seconds is not None
        else None
    )

    summary = {
        "base_model": BASE_MODEL,
        "model_load_path": model_load_path,
        "model_mode": "text_only" if strip_vision_for_training else "multimodal",
        "stripped_vision_for_training": strip_vision_for_training,
        "continue_from_run_name": continue_from_run_name or None,
        "dataset_name": dataset_name,
        "dataset_split": dataset_split,
        "dataset_revision": dataset_revision,
        "dataset_rows": len(dataset),
        "max_seq_length": max_seq_length,
        "epochs": epochs,
        "max_steps": max_steps,
        "resume_from_checkpoint": resolved_resume_checkpoint,
        "save_steps": save_steps,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": effective_batch_size,
        "steps_per_epoch": steps_per_epoch,
        "target_full_train_steps": target_full_train_steps,
        "observed_steps": observed_steps,
        "learning_rate": learning_rate,
        "lora_r": lora_r,
        "lora_alpha": lora_r * 2,
        "load_in_4bit": False,
        "load_in_16bit": True,
        "output_dir": str(run_output_dir),
        "setup_seconds_before_train": setup_seconds,
        "train_runtime_seconds": train_runtime_seconds,
        "train_loss": trainer_stats.metrics.get("train_loss"),
        "seconds_per_step": seconds_per_step,
        "estimated_full_train_runtime_seconds": estimated_full_train_runtime_seconds,
        "estimated_full_job_runtime_seconds": estimated_full_job_runtime_seconds,
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
    dataset_name: str = DATASET_NAME,
    dataset_split: str = DATASET_SPLIT,
    dataset_revision: str = DATASET_REVISION,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    epochs: float = DEFAULT_EPOCHS,
    per_device_train_batch_size: int = DEFAULT_PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps: int = DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    lora_r: int = DEFAULT_LORA_R,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
    warmup_steps: int = DEFAULT_WARMUP_STEPS,
    seed: int = DEFAULT_SEED,
    max_steps: int = -1,
    resume_from_checkpoint: str = "",
    save_steps: int = 20,
    hf_repo_id: str = "",
    save_merged_16bit: bool = False,
    report_to: str = "none",
    continue_from_run_name: str = "",
    output_run_name: str = "",
    strip_vision_for_training: bool = False,
):
    download_model.remote()
    validate_snapshot.remote()
    if strip_vision_for_training:
        prepare_text_only_snapshot.remote()
        validate_text_only_snapshot.remote()
    train.remote(
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        dataset_revision=dataset_revision,
        max_seq_length=max_seq_length,
        epochs=epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        warmup_steps=warmup_steps,
        seed=seed,
        max_steps=max_steps,
        resume_from_checkpoint=resume_from_checkpoint,
        save_steps=save_steps,
        hf_repo_id=hf_repo_id,
        save_merged_16bit=save_merged_16bit,
        report_to=report_to,
        continue_from_run_name=continue_from_run_name,
        output_run_name=output_run_name,
        strip_vision_for_training=strip_vision_for_training,
    )
