"""Manual config repair helpers for the pruned Qwen3.5-24B-A10B checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

import modal


APP_NAME = "repair-qwen35-24b-config"
REPO_ID = "sandeshrajx/qwen3.5b-24b-a10b"
RESULTS_DIR = "/results"
HF_CACHE_DIR = "/root/.cache/huggingface"
PRUNED_MODEL_DIR = (
    "/results/Qwen3.5-122B-A10B/evol-codealpaca-v1/pruned_models/reap-seed_42-0.85"
)
CACHED_MODEL_DIR = f"{HF_CACHE_DIR}/preloaded/qwen35_24b_a10b"
EXPECTED_NUM_EXPERTS = 39

app = modal.App(APP_NAME)

results_volume = modal.Volume.from_name("reap-results")
hf_cache_volume = modal.Volume.from_name("hf-cache-qwen35-24b-a10b")
hf_secret = modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
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


def read_num_experts(path: Path) -> int:
    data = json.loads(path.read_text(encoding="utf-8"))
    return int(data["text_config"]["num_experts"])


@app.function(
    image=image,
    volumes={
        RESULTS_DIR: results_volume,
        HF_CACHE_DIR: hf_cache_volume,
    },
    timeout=10 * 60,
)
def verify_configs():
    pruned_config = Path(PRUNED_MODEL_DIR) / "config.json"
    cached_config = Path(CACHED_MODEL_DIR) / "config.json"
    result = {
        "pruned_model_dir": PRUNED_MODEL_DIR,
        "cached_model_dir": CACHED_MODEL_DIR,
        "pruned_num_experts": read_num_experts(pruned_config),
        "cached_num_experts": read_num_experts(cached_config),
    }
    print(json.dumps(result, indent=2))
    return result


@app.function(
    image=image,
    volumes={
        RESULTS_DIR: results_volume,
    },
    secrets=[hf_secret],
    timeout=20 * 60,
)
def upload_pruned_config_to_hf():
    from huggingface_hub import HfApi

    config_path = Path(PRUNED_MODEL_DIR) / "config.json"
    config_num_experts = read_num_experts(config_path)
    if config_num_experts != EXPECTED_NUM_EXPERTS:
        raise RuntimeError(
            f"Pruned config has num_experts={config_num_experts}, expected {EXPECTED_NUM_EXPERTS}."
        )

    api = HfApi(token=True)
    api.upload_file(
        path_or_fileobj=str(config_path),
        path_in_repo="config.json",
        repo_id=REPO_ID,
        repo_type="model",
    )
    result = {
        "repo_id": REPO_ID,
        "uploaded_path": "config.json",
        "num_experts": config_num_experts,
    }
    print(json.dumps(result, indent=2))
    return result


@app.function(
    image=image,
    gpu=["A100-80GB"],
    volumes={
        RESULTS_DIR: results_volume,
        HF_CACHE_DIR: hf_cache_volume,
    },
    timeout=60 * 60,
)
def smoke_load(model_source: str):
    import torch
    from unsloth import FastLanguageModel

    if model_source == "pruned_volume":
        model_path = PRUNED_MODEL_DIR
    elif model_source == "hf_cache":
        model_path = CACHED_MODEL_DIR
    else:
        raise ValueError(f"Unsupported model_source: {model_source}")

    model, processor = FastLanguageModel.from_pretrained(
        model_path,
        max_seq_length=8192,
        load_in_4bit=False,
        fast_inference=False,
    )

    result = {
        "model_source": model_source,
        "model_path": model_path,
        "loaded": True,
        "num_experts": read_num_experts(Path(model_path) / "config.json"),
    }
    print(json.dumps(result, indent=2))

    del model
    torch.cuda.empty_cache()
    return result


@app.local_entrypoint()
def main():
    verify_configs.remote()
    upload_pruned_config_to_hf.remote()
    smoke_load.remote("pruned_volume")
    smoke_load.remote("hf_cache")
