"""Quantize the pruned Qwen3.5-24B-A10B model to GGUF Q4_K_M and upload to Hugging Face.

Usage:
    modal run modal_quantize_qwen35_24b_a10b_gguf.py
    modal run modal_quantize_qwen35_24b_a10b_gguf.py --hf-repo-id sandeshrajx/qwen3.5b-24b-a10b
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import modal


APP_NAME = "quantize-qwen35-24b-a10b-gguf"
BASE_MODEL = "sandeshrajx/qwen3.5b-24b-a10b"
HF_CACHE_DIR = "/root/.cache/huggingface"
RESULTS_DIR = "/results"
OUTPUT_ROOT = "/outputs"
MODEL_SNAPSHOT_DIR = f"{HF_CACHE_DIR}/preloaded/qwen35_24b_a10b"
PRUNED_MODEL_DIR = f"{RESULTS_DIR}/Qwen3.5-122B-A10B/evol-codealpaca-v1/pruned_models/reap-seed_42-0.85"
DEFAULT_HF_REPO_ID = "sandeshrajx/qwen3.5b-24b-a10b"
LLAMA_CPP_DIR = "/opt/llama.cpp"
GGUF_OUTPUT_FILENAME = "sandeshrajx-qwen3.5b-24b-a10b-Q4_K_M.gguf"

app = modal.App(APP_NAME)

hf_cache_volume = modal.Volume.from_name("hf-cache-qwen35-24b-a10b", create_if_missing=True)
results_volume = modal.Volume.from_name("reap-results", create_if_missing=True)
outputs_volume = modal.Volume.from_name("qwen35-24b-a10b-opus-outputs", create_if_missing=True)

huggingface_secret = modal.Secret.from_name(
    "huggingface-secret", required_keys=["HF_TOKEN"]
)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .apt_install("git", "wget", "cmake", "build-essential")
    .run_commands("pip install --upgrade -qqq uv")
    .pip_install(
        "huggingface-hub>=0.34.0",
        "sentencepiece",
        "gguf",
        "protobuf",
        "safetensors",
        "transformers>=4.55.0",
        "torch",
    )
    .run_commands(
        f"git clone https://github.com/ggml-org/llama.cpp {LLAMA_CPP_DIR} && "
        f"cmake -S {LLAMA_CPP_DIR} -B {LLAMA_CPP_DIR}/build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=OFF && "
        f"cmake --build {LLAMA_CPP_DIR}/build --config Release -j"
    )
    .env({
        "HF_XET_HIGH_PERFORMANCE": "1",
    })
)


def resolve_source_model_dir(prefer_pruned_volume: bool) -> Path:
    pruned_dir = Path(PRUNED_MODEL_DIR)
    cache_dir = Path(MODEL_SNAPSHOT_DIR)

    if prefer_pruned_volume and (pruned_dir / "config.json").exists():
        return pruned_dir
    if (cache_dir / "config.json").exists():
        return cache_dir
    if (pruned_dir / "config.json").exists():
        return pruned_dir
    raise FileNotFoundError(
        f"Could not find a model snapshot in either {cache_dir} or {pruned_dir}"
    )


def make_working_copy(source_dir: Path, work_dir: Path) -> Path:
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    for item in source_dir.iterdir():
        destination = work_dir / item.name
        if item.is_dir():
            shutil.copytree(item, destination)
        else:
            shutil.copy2(item, destination)

    stale_single_file = work_dir / "model.safetensors"
    if stale_single_file.exists() and (work_dir / "model.safetensors.index.json").exists():
        stale_single_file.unlink()

    return work_dir


@app.function(
    image=image,
    volumes={
        HF_CACHE_DIR: hf_cache_volume,
        RESULTS_DIR: results_volume,
        OUTPUT_ROOT: outputs_volume,
    },
    secrets=[huggingface_secret],
    timeout=8 * 60 * 60,
)
def quantize_to_gguf(
    hf_repo_id: str = DEFAULT_HF_REPO_ID,
    prefer_pruned_volume: bool = False,
):
    from huggingface_hub import HfApi

    source_dir = resolve_source_model_dir(prefer_pruned_volume=prefer_pruned_volume)
    run_dir = Path(OUTPUT_ROOT) / "qwen35-24b-a10b-gguf"
    work_dir = run_dir / "hf-source"
    gguf_dir = run_dir / "gguf"
    f16_gguf_path = gguf_dir / "model-f16.gguf"
    final_gguf_path = gguf_dir / GGUF_OUTPUT_FILENAME

    print(f"Using source model directory: {source_dir}")
    make_working_copy(source_dir=source_dir, work_dir=work_dir)
    gguf_dir.mkdir(parents=True, exist_ok=True)

    if not f16_gguf_path.exists():
        print(f"Converting HF model to F16 GGUF at {f16_gguf_path}")
        subprocess.run(
            [
                "python3",
                f"{LLAMA_CPP_DIR}/convert_hf_to_gguf.py",
                str(work_dir),
                "--outfile",
                str(f16_gguf_path),
                "--outtype",
                "f16",
            ],
            check=True,
        )
        outputs_volume.commit()

    if not final_gguf_path.exists():
        quantize_bin = f"{LLAMA_CPP_DIR}/build/bin/llama-quantize"
        if not os.path.exists(quantize_bin):
            quantize_bin = f"{LLAMA_CPP_DIR}/build/bin/quantize"
        print(f"Quantizing GGUF to {final_gguf_path.name} with Q4_K_M")
        subprocess.run(
            [
                quantize_bin,
                str(f16_gguf_path),
                str(final_gguf_path),
                "Q4_K_M",
            ],
            check=True,
        )
        outputs_volume.commit()

    print(f"Uploading {final_gguf_path.name} to {hf_repo_id}")
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.upload_file(
        path_or_fileobj=str(final_gguf_path),
        path_in_repo=f"gguf/{final_gguf_path.name}",
        repo_id=hf_repo_id,
        repo_type="model",
    )

    result = {
        "base_model": BASE_MODEL,
        "source_dir": str(source_dir),
        "prefer_pruned_volume": prefer_pruned_volume,
        "work_dir": str(work_dir),
        "f16_gguf_path": str(f16_gguf_path),
        "final_gguf_path": str(final_gguf_path),
        "hf_repo_id": hf_repo_id,
        "repo_url": f"https://huggingface.co/{hf_repo_id}",
    }
    print(json.dumps(result, indent=2))
    return result


@app.local_entrypoint()
def main(
    hf_repo_id: str = DEFAULT_HF_REPO_ID,
    prefer_pruned_volume: bool = False,
):
    quantize_to_gguf.remote(
        hf_repo_id=hf_repo_id,
        prefer_pruned_volume=prefer_pruned_volume,
    )
