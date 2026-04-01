"""Repair a pruned/custom Qwen3.5 MoE repo using the original Qwen repo metadata/config layout.

Usage:
    python3 -m modal run modal_repair_qwen_repo.py
    python3 -m modal run modal_repair_qwen_repo.py --custom-model your-name/your-pruned-repo
    python3 -m modal run modal_repair_qwen_repo.py --custom-model your-name/your-pruned-repo --hf-repo-id your-name/your-repaired-repo
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import modal


APP_NAME = "repair-qwen-repo"
BASE_MODEL = "Qwen/Qwen3.5-35B-A3B"
CUSTOM_MODEL = "Flagstone8878/Qwen3.5-18B-REAP-A3B-Coding"
WORK_ROOT = "/work"

COPY_FILES = [
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
    "chat_template.json",
    "tokenizer.json",
    "tokenizer.model",
    "merges.txt",
    "vocab.json",
]

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "huggingface-hub>=0.34.0",
        "transformers>=4.57.0",
        "sentencepiece",
        "protobuf",
        "jinja2",
    )
)

huggingface_secret = modal.Secret.from_name(
    "huggingface-secret", required_keys=["HF_TOKEN"]
)


def _snapshot_download(repo_id: str, local_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        ignore_patterns=["*.safetensors", "*.bin", "*.pt"],
    )


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2))


def _repair_config(base_dir: Path, custom_dir: Path) -> dict:
    base_config = _load_json(base_dir / "config.json")
    custom_config = _load_json(custom_dir / "config.json")

    repaired = json.loads(json.dumps(base_config))

    text_config = custom_config.get("text_config")
    if text_config is None:
        text_keys = set(base_config.get("text_config", {}).keys())
        text_keys.update(
            {
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
                "output_router_logits",
                "partial_rotary_factor",
            }
        )
        text_config = {k: v for k, v in custom_config.items() if k in text_keys}

    repaired["text_config"] = text_config
    repaired["text_config"]["model_type"] = "qwen3_5_moe_text"

    if "architectures" in custom_config:
        repaired["architectures"] = ["Qwen3_5MoeForConditionalGeneration"]

    if "transformers_version" in custom_config:
        repaired["transformers_version"] = custom_config["transformers_version"]

    _write_json(custom_dir / "config.original.json", custom_config)
    _write_json(custom_dir / "config.json", repaired)
    _write_json(custom_dir / "config.repaired.json", repaired)

    return {
        "repaired_model_type": repaired.get("model_type"),
        "repaired_architecture": repaired.get("architectures", [None])[0],
        "repaired_text_model_type": repaired.get("text_config", {}).get("model_type"),
        "has_vision_config": "vision_config" in repaired,
    }


def _copy_metadata_files(base_dir: Path, custom_dir: Path) -> dict:
    copied = []
    missing = []
    for name in COPY_FILES:
        src = base_dir / name
        dst = custom_dir / name
        if src.exists():
            shutil.copy2(src, dst)
            copied.append(name)
        else:
            missing.append(name)
    return {"copied_files": copied, "missing_files": missing}


def _validate_repo(path: Path) -> dict:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(path),
        trust_remote_code=True,
        use_fast=False,
    )

    messages = [{"role": "user", "content": "Write a haiku about code."}]
    chat_render_error = ""
    chat_preview = None
    try:
        chat_preview = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )[:500]
    except Exception as exc:
        chat_render_error = str(exc)

    return {
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
        "vocab_size": tokenizer.vocab_size,
        "has_chat_template": bool(getattr(tokenizer, "chat_template", None)),
        "chat_render_error": chat_render_error,
        "chat_preview": chat_preview,
    }


@app.function(
    image=image,
    secrets=[huggingface_secret],
    timeout=60 * 60,
)
def repair_repo(
    base_model: str = BASE_MODEL,
    custom_model: str = CUSTOM_MODEL,
    hf_repo_id: str = "",
    private: bool = False,
):
    from huggingface_hub import HfApi

    work_root = Path(WORK_ROOT)
    work_root.mkdir(parents=True, exist_ok=True)

    base_dir = work_root / "base"
    custom_dir = work_root / "custom"

    _snapshot_download(base_model, base_dir)
    _snapshot_download(custom_model, custom_dir)

    config_summary = _repair_config(base_dir, custom_dir)
    copy_summary = _copy_metadata_files(base_dir, custom_dir)
    validation_summary = _validate_repo(custom_dir)

    result = {
        "base_model": base_model,
        "custom_model": custom_model,
        "repaired_repo_path": str(custom_dir),
        **config_summary,
        **copy_summary,
        **validation_summary,
    }

    if hf_repo_id:
        api = HfApi(token=True)
        api.create_repo(repo_id=hf_repo_id, repo_type="model", private=private, exist_ok=True)
        api.upload_folder(
            folder_path=str(custom_dir),
            repo_id=hf_repo_id,
            repo_type="model",
        )
        result["hf_repo_id"] = hf_repo_id

    print(json.dumps(result, indent=2))
    return result


@app.local_entrypoint()
def main(
    base_model: str = BASE_MODEL,
    custom_model: str = CUSTOM_MODEL,
    hf_repo_id: str = "",
    private: bool = False,
):
    repair_repo.remote(
        base_model=base_model,
        custom_model=custom_model,
        hf_repo_id=hf_repo_id,
        private=private,
    )
