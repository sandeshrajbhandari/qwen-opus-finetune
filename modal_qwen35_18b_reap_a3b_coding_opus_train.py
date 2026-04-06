"""Fine-tune Flagstone8878/Qwen3.5-18B-REAP-A3B-Coding on the full Opus dataset.

Usage:
    modal run modal_qwen35_18b_reap_a3b_coding_opus_train.py
    modal run modal_qwen35_18b_reap_a3b_coding_opus_train.py --epochs 2
    modal run modal_qwen35_18b_reap_a3b_coding_opus_train.py --hf-repo-id your-name/qwen35-18b-reap-a3b-coding-opus
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import modal


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
APP_NAME = "qwen35-18b-reap-a3b-coding-opus-sft"
BASE_MODEL = "Flagstone8878/Qwen3.5-18B-REAP-A3B-Coding"
COMPAT_REFERENCE_MODEL = "Qwen/Qwen3.5-35B-A3B"
DATASET_NAME = "nohurry/Opus-4.6-Reasoning-3000x-filtered"
DATASET_SPLIT = "train"
DATASET_REVISION = "main"
OUTPUT_ROOT = "/outputs"
HF_CACHE_DIR = "/root/.cache/huggingface"
MODEL_SNAPSHOT_DIR = f"{HF_CACHE_DIR}/preloaded/flagstone_qwen35_18b_reap_a3b_coding"

DEFAULT_MAX_SEQ_LENGTH = 8192
DEFAULT_EPOCHS = 2
# RECOMMENDED: Change batch size to 1 and gradient accumulation to 6 to avoid OOM
DEFAULT_PER_DEVICE_BATCH_SIZE = 6
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 4
DEFAULT_LEARNING_RATE = 6e-5
DEFAULT_LORA_R = 32
DEFAULT_LORA_ALPHA = 64
DEFAULT_WARMUP_STEPS = 2
DEFAULT_SEED = 3407


# ---------------------------------------------------------------------------
# Modal resources
# ---------------------------------------------------------------------------
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
    return f"qwen35-18b-reap-a3b-coding-opus-msl{max_seq_length}-e{str(epochs).replace('.', '_')}"


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


def patch_model_config(snapshot_path: Path) -> dict:
    config_path = snapshot_path / "config.json"
    original_path = snapshot_path / "config.original.json"
    patched_path = snapshot_path / "config.patched.json"

    config = json.loads(config_path.read_text())
    original_config = json.loads(json.dumps(config))

    if not original_path.exists():
        original_path.write_text(json.dumps(original_config, indent=2))

    if "text_config" not in config:
        text_keys = {
            "attention_bias",
            "attention_dropout",
            "attn_output_gate",
            "dtype",
            "eos_token_id",
            "full_attention_interval",
            "head_dim",
            "hidden_act",
            "hidden_size",
            "initializer_range",
            "layer_types",
            "linear_conv_kernel_dim",
            "linear_key_head_dim",
            "linear_num_key_heads",
            "linear_num_value_heads",
            "linear_value_head_dim",
            "max_position_embeddings",
            "mlp_only_layers",
            "model_type",
            "moe_intermediate_size",
            "mtp_num_hidden_layers",
            "mtp_use_dedicated_embeddings",
            "num_attention_heads",
            "num_experts",
            "num_experts_per_tok",
            "num_hidden_layers",
            "num_key_value_heads",
            "rms_norm_eps",
            "router_aux_loss_coef",
            "shared_expert_intermediate_size",
            "use_cache",
            "vocab_size",
            "mamba_ssm_dtype",
            "rope_parameters",
        }
        text_config = {}
        for key in list(config.keys()):
            if key in text_keys:
                text_config[key] = config.pop(key)
        if text_config:
            config["text_config"] = text_config

    config["model_type"] = "qwen3_5_moe"
    config["architectures"] = ["Qwen3_5MoeForConditionalGeneration"]

    text_config = config.setdefault("text_config", {})
    text_config["model_type"] = "qwen3_5_moe_text"

    if "vision_config" not in config:
        config["vision_config"] = {
            "deepstack_visual_indexes": [],
            "depth": 27,
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_size": 1152,
            "in_channels": 3,
            "initializer_range": 0.02,
            "intermediate_size": 4304,
            "model_type": "qwen3_5_moe",
            "num_heads": 16,
            "num_position_embeddings": 2304,
            "out_hidden_size": text_config.get("hidden_size", 2048),
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        }

    config.setdefault("image_token_id", 248056)
    config.setdefault("video_token_id", 248057)
    config.setdefault("vision_start_token_id", 248053)
    config.setdefault("vision_end_token_id", 248054)
    config.setdefault("tie_word_embeddings", False)

    config_path.write_text(json.dumps(config, indent=2))
    patched_path.write_text(json.dumps(config, indent=2))
    return {
        "config_path": str(config_path),
        "original_config_path": str(original_path),
        "patched_config_path": str(patched_path),
        "model_type": config.get("model_type"),
        "architecture": config.get("architectures", [None])[0],
        "text_model_type": text_config.get("model_type"),
        "has_vision_config": "vision_config" in config,
    }


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
        # The trainer validates eos_token against the tokenizer vocab.
        # Use the real serialized token string that corresponds to the configured ID.
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


@app.function(
    image=image,
    volumes={HF_CACHE_DIR: hf_cache_volume},
    secrets=[huggingface_secret],
    timeout=4 * MINUTES,
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

    patch_summary = patch_model_config(snapshot_path)
    tokenizer_patch_summary = patch_tokenizer_files(snapshot_path)
    hf_cache_volume.commit()
    print(json.dumps({**patch_summary, **tokenizer_patch_summary}, indent=2))
    return str(snapshot_path)


@app.function(
    image=image,
    volumes={HF_CACHE_DIR: hf_cache_volume},
    timeout=4 * MINUTES,
)
def validate_patched_config():
    snapshot_path = Path(MODEL_SNAPSHOT_DIR)
    if not (snapshot_path / "config.json").exists():
        raise FileNotFoundError(f"Missing config.json under {snapshot_path}")
    patch_summary = patch_model_config(snapshot_path)
    tokenizer_patch_summary = patch_tokenizer_files(snapshot_path)
    result = {**patch_summary, **tokenizer_patch_summary}
    print(json.dumps(result, indent=2))
    return result


@app.function(
    image=image,
    gpu=["A100-80GB"],
    timeout=120 * MINUTES,
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
):
    import torch
    import unsloth
    from datasets import load_dataset
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

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

    job_start_time = time.perf_counter()

    model_load_path = (
        str(continued_lora_dir)
        if continued_lora_dir is not None
        else MODEL_SNAPSHOT_DIR if Path(MODEL_SNAPSHOT_DIR).exists() else BASE_MODEL
    )
    if continued_lora_dir is not None and not continued_lora_dir.exists():
        raise FileNotFoundError(f"Continuation LoRA directory not found: {continued_lora_dir}")

    print("Loading base model...")
    # Matches the boolean flags from the reference
    model, processor = FastLanguageModel.from_pretrained(
        model_load_path,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        fast_inference=False, # Not supported for MoE (yet!)
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

    # Matches the precise configuration & structure from the reference 
    if continued_lora_dir is None:
        model = FastLanguageModel.get_peft_model(
            model,
            r = lora_r,
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj", "gate_up_proj", #Enable LoRA on MoE layers
            ],
            lora_alpha = lora_r * 2, # *2 speeds up training as per reference (overriding input alpha if needed)
            use_gradient_checkpointing = True, # Reduces memory usage
            random_state = seed,
            bias = "none",
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
    
    # Matches the exact formatting string targets from the reference notebook
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n<think>",
    )

    # -----------------------------------------------------------------------
    # Verification: Verify the Response-Only Masking worked correctly
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("VERIFICATION: Checking that train_on_responses_only worked...")
    print("="*60)
    
    # Get the very first item in the training dataset
    sample_index = 0
    
    # 1. Decode the full input (what the model sees)
    full_input_decoded = tokenizer.decode(trainer.train_dataset[sample_index]["input_ids"])
    print(f"\n[FULL RAW INPUT (Index {sample_index})] - The model sees all of this:")
    print("-" * 40)
    print(full_input_decoded)
    print("-" * 40)
    
    # 2. Decode the labels (what the model is penalized on)
    # Masked tokens have a value of -100 in PyTorch.
    # We replace -100 with the pad_token_id so the tokenizer can decode it as "empty space"
    raw_labels = trainer.train_dataset[sample_index]["labels"]
    masked_labels_decoded = tokenizer.decode(
        [tokenizer.pad_token_id if x == -100 else x for x in raw_labels]
    ).replace(tokenizer.pad_token, "") # Clean up the output so it's readable
    
    print(f"\n[TRAINABLE LABELS (Index {sample_index})] - The model is ONLY trained on this:")
    print("-" * 40)
    print(masked_labels_decoded)
    print("-" * 40)
    print("If the above block only shows the `<think>...` block and the answer, it worked!\n")
    # -----------------------------------------------------------------------

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
):
    download_model.remote()
    validate_patched_config.remote()
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
    )
