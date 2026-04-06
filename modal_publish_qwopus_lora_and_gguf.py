"""Publish a saved PEFT LoRA run and adapter-only GGUF to Hugging Face.

Usage:
    modal run modal_publish_qwopus_lora_and_gguf.py
    modal run modal_publish_qwopus_lora_and_gguf.py --run-name qwen35-18b-reap-a3b-coding-opus-msl8192-e2_0-lr2e4
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import modal


APP_NAME = "publish-qwopus-lora-and-adapter-gguf"
BASE_MODEL = "Flagstone8878/Qwen3.5-18B-REAP-A3B-Coding"
DATASET_NAME = "nohurry/Opus-4.6-Reasoning-3000x-filtered"
HF_CACHE_DIR = "/root/.cache/huggingface"
OUTPUT_ROOT = "/outputs"
MODEL_SNAPSHOT_DIR = f"{HF_CACHE_DIR}/preloaded/flagstone_qwen35_18b_reap_a3b_coding"
DEFAULT_RUN_NAME = "qwen35-18b-reap-a3b-coding-opus-msl8192-e2_0-lr2e4"
DEFAULT_HF_REPO_ID = "sandeshrajx/qwopus-lora-tests"
LLAMA_CPP_DIR = Path("/opt/llama.cpp")
LLAMA_CPP_DIR_STR = "/opt/llama.cpp"

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
    .apt_install("git", "build-essential", "cmake")
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
        "safetensors",
    )
    .run_commands(
        "mkdir -p /opt && "
        f"git clone https://github.com/ggml-org/llama.cpp {LLAMA_CPP_DIR_STR}"
    )
    .env({
        "HF_XET_HIGH_PERFORMANCE": "1",
        "UNSLOTH_MOE_DISABLE_AUTOTUNE": "1",
    })
)


def slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")


def build_gguf_name(run_name: str) -> str:
    base_slug = slugify(BASE_MODEL.replace("/", "-"))
    run_slug = slugify(run_name)
    return f"{base_slug}-{run_slug}-lora-Q8_0.gguf"


def build_model_card(run_name: str, summary: dict, repo_id: str, gguf_name: str) -> str:
    return f"""---
license: apache-2.0
base_model: {summary.get("base_model", BASE_MODEL)}
tags:
  - qwen
  - qwen3
  - qwen3.5
  - lora
  - peft
  - gguf
  - llama.cpp
  - code
  - reasoning
library_name: peft
pipeline_tag: text-generation
---

# {repo_id}

Experimental PEFT LoRA adapter and adapter-only GGUF export for Qwen 3.5 18B REAP-A3B Coding.

## What This Repo Contains

- PEFT LoRA adapter files at repo root
- Tokenizer files copied from the training artifact for convenience
- Adapter-only GGUF for llama.cpp under `gguf-lora/`

This repo does not contain a merged full model.

## Training Run

- Base model: `{summary.get("base_model", BASE_MODEL)}`
- Dataset: `{DATASET_NAME}`
- Run name: `{run_name}`
- Context length: `{summary.get("max_seq_length")}`
- Epochs: `{summary.get("epochs")}`
- Learning rate: `{summary.get("learning_rate")}`
- Effective batch size: `{summary.get("effective_batch_size")}`
- Final reported train loss: `{summary.get("train_loss")}`

## Intent

This is a testing repo for comparing:

- the original base model
- the PEFT LoRA adapter in Transformers / PEFT runtimes
- the adapter-only GGUF in llama.cpp-compatible runtimes

It is intended for evaluation and experimentation, not as a production-quality coding release.

## PEFT Usage

Load the adapter with the original base model:

```python
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM

model = AutoPeftModelForCausalLM.from_pretrained("{repo_id}", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("{repo_id}", trust_remote_code=True, use_fast=False)
```

## llama.cpp Usage

The GGUF in this repo is an adapter-only LoRA GGUF, intended to be used with a separate base-model GGUF:

```bash
./llama-cli -m <base-model>.gguf --lora {gguf_name}
```

The adapter GGUF is located at:

```text
gguf-lora/{gguf_name}
```

## Prompt Format

Training examples followed this assistant structure:

```text
<think>
...reasoning...
</think>

...final answer...
```

Response-only masking started at:

```text
<|im_start|>assistant
<think>
```

## Important Notes

- This repo contains an adapter-only GGUF, not a merged-model GGUF.
- The adapter GGUF must be paired with a compatible base GGUF derived from the same base model family.
- Output quality is still experimental; some prompts are more coherent than the base model, but code quality remains inconsistent.
"""


def patch_lora_converter(convert_script: Path) -> None:
    original = convert_script.read_text(encoding="utf-8")
    old = """                        if ".base_layer.weight" in name:
                            continue
"""
    new = """                        if ".base_layer.weight" in name or ".mlp.experts.weight" in name:
                            continue
"""
    if old not in original:
        raise RuntimeError("Expected patch target not found in convert_lora_to_gguf.py")
    convert_script.write_text(original.replace(old, new, 1), encoding="utf-8")


@app.function(
    image=image,
    gpu=["A100-80GB"],
    volumes={
        HF_CACHE_DIR: hf_cache_volume,
        OUTPUT_ROOT: outputs_volume,
    },
    secrets=[huggingface_secret],
    timeout=3 * 60 * 60,
)
def publish(
    run_name: str = DEFAULT_RUN_NAME,
    hf_repo_id: str = DEFAULT_HF_REPO_ID,
):
    import os
    from huggingface_hub import HfApi

    run_output_dir = Path(OUTPUT_ROOT) / run_name
    lora_dir = run_output_dir / "lora"
    summary_path = run_output_dir / "summary.json"
    gguf_dir = run_output_dir / "gguf-lora"
    publish_dir = Path("/tmp") / f"{run_name}-lora-upload"

    if not lora_dir.exists():
        raise FileNotFoundError(f"LoRA adapter directory not found: {lora_dir}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Training summary not found: {summary_path}")
    if not Path(MODEL_SNAPSHOT_DIR).exists():
        raise FileNotFoundError(f"Base model snapshot dir not found: {MODEL_SNAPSHOT_DIR}")

    summary = json.loads(summary_path.read_text())
    gguf_name = build_gguf_name(run_name)
    gguf_dir.mkdir(parents=True, exist_ok=True)
    gguf_path = gguf_dir / gguf_name

    convert_script = LLAMA_CPP_DIR / "convert_lora_to_gguf.py"
    if not convert_script.exists():
        fallback_script = Path("/root/optllama.cpp/convert_lora_to_gguf.py")
        if fallback_script.exists():
            convert_script = fallback_script
        else:
            raise FileNotFoundError(f"llama.cpp converter not found: {convert_script}")
    patch_lora_converter(convert_script)

    convert_cmd = [
        "python",
        str(convert_script),
        "--base",
        MODEL_SNAPSHOT_DIR,
        "--outfile",
        str(gguf_path),
        "--outtype",
        "q8_0",
        str(lora_dir),
    ]

    print("Running adapter GGUF conversion:")
    print(" ".join(convert_cmd))
    subprocess.run(convert_cmd, check=True)

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo_id=hf_repo_id, repo_type="model", exist_ok=True)

    if publish_dir.exists():
        shutil.rmtree(publish_dir)
    shutil.copytree(lora_dir, publish_dir)
    (publish_dir / "README.md").write_text(
        build_model_card(run_name, summary, hf_repo_id, gguf_name),
        encoding="utf-8",
    )

    print(f"Uploading LoRA adapter files from {publish_dir} to {hf_repo_id}")
    api.upload_folder(
        folder_path=str(publish_dir),
        repo_id=hf_repo_id,
        repo_type="model",
    )

    print(f"Uploading adapter GGUF {gguf_path} to {hf_repo_id}")
    api.upload_file(
        path_or_fileobj=str(gguf_path),
        path_in_repo=f"gguf-lora/{gguf_name}",
        repo_id=hf_repo_id,
        repo_type="model",
    )

    outputs_volume.commit()
    hf_cache_volume.commit()

    result = {
        "run_name": run_name,
        "hf_repo_id": hf_repo_id,
        "lora_dir": str(lora_dir),
        "gguf_path": str(gguf_path),
        "gguf_name": gguf_name,
        "repo_url": f"https://huggingface.co/{hf_repo_id}",
    }
    print(json.dumps(result, indent=2))
    return result


@app.local_entrypoint()
def main(
    run_name: str = DEFAULT_RUN_NAME,
    hf_repo_id: str = DEFAULT_HF_REPO_ID,
):
    publish.remote(
        run_name=run_name,
        hf_repo_id=hf_repo_id,
    )
