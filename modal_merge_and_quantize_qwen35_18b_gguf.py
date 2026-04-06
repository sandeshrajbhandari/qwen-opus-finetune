"""Merge a saved Qwen3.5-18B LoRA and quantize it to GGUF.

Usage:
    modal run modal_merge_and_quantize_qwen35_18b_gguf.py
    modal run modal_merge_and_quantize_qwen35_18b_gguf.py --run-name qwen35-18b-reap-a3b-coding-opus-msl8192-e2_0-lr2e4
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import modal


APP_NAME = "merge-and-quantize-qwen35-18b-gguf"
BASE_MODEL = "Flagstone8878/Qwen3.5-18B-REAP-A3B-Coding"
HF_CACHE_DIR = "/root/.cache/huggingface"
OUTPUT_ROOT = "/outputs"
MODEL_SNAPSHOT_DIR = f"{HF_CACHE_DIR}/preloaded/flagstone_qwen35_18b_reap_a3b_coding"
DEFAULT_RUN_NAME = "qwen35-18b-reap-a3b-coding-opus-msl8192-e2_0-lr2e4"
DEFAULT_HF_REPO_ID = "sandeshrajx/qwopus-lora-tests"
DEFAULT_MAX_SEQ_LENGTH = 8192
LLAMA_CPP_DIR = "/opt/llama.cpp"
GGUF_OUTPUT_FILENAME = "Flagstone8878-Qwen3.5-18B-REAP-A3B-Coding-qwopus-merged-Q4_K_M.gguf"

app = modal.App(APP_NAME)

hf_cache_volume = modal.Volume.from_name("hf-cache-qwen35-18b-reap-a3b", create_if_missing=True)
outputs_volume = modal.Volume.from_name("qwen35-18b-reap-a3b-opus-outputs", create_if_missing=True)

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
        "accelerate",
        "huggingface-hub>=0.34.0",
        "numpy",
        "sentencepiece",
        "gguf",
        "protobuf",
        "safetensors",
    )
    .run_commands(
        f"git clone https://github.com/ggml-org/llama.cpp {LLAMA_CPP_DIR} && "
        f"cmake -S {LLAMA_CPP_DIR} -B {LLAMA_CPP_DIR}/build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=OFF && "
        f"cmake --build {LLAMA_CPP_DIR}/build --config Release -j"
    )
    .env({
        "HF_XET_HIGH_PERFORMANCE": "1",
        "UNSLOTH_MOE_DISABLE_AUTOTUNE": "1",
    })
)


@app.function(
    image=image,
    gpu=["A100-80GB"],
    volumes={
        HF_CACHE_DIR: hf_cache_volume,
        OUTPUT_ROOT: outputs_volume,
    },
    secrets=[huggingface_secret],
    timeout=4 * 60 * 60,
)
def merge_and_quantize(
    run_name: str = DEFAULT_RUN_NAME,
    hf_repo_id: str = DEFAULT_HF_REPO_ID,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
):
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer
    from unsloth import FastLanguageModel

    run_output_dir = Path(OUTPUT_ROOT) / run_name
    lora_dir = run_output_dir / "lora"
    merged_dir = run_output_dir / "merged_16bit"
    gguf_dir = run_output_dir / "gguf-merged"
    f16_gguf_path = gguf_dir / "model-f16.gguf"
    final_gguf_path = gguf_dir / GGUF_OUTPUT_FILENAME

    if not lora_dir.exists():
        raise FileNotFoundError(f"LoRA adapter directory not found: {lora_dir}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(lora_dir),
        trust_remote_code=True,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if not merged_dir.exists():
        print(f"Loading LoRA adapter from {lora_dir}")
        model, processor = FastLanguageModel.from_pretrained(
            str(lora_dir),
            max_seq_length=max_seq_length,
            load_in_4bit=False,
            fast_inference=False,
        )
        print(f"Merging adapter into 16-bit model at {merged_dir}")
        model.save_pretrained_merged(
            str(merged_dir),
            tokenizer,
            save_method="merged_16bit",
        )
        outputs_volume.commit()
        hf_cache_volume.commit()

    gguf_dir.mkdir(parents=True, exist_ok=True)

    if not f16_gguf_path.exists():
        config_path = merged_dir / "config.json"
        original_config = json.loads(config_path.read_text(encoding="utf-8"))
        patched_config = json.loads(json.dumps(original_config))
        original_architectures = patched_config.get("architectures", [])
        patched_config["architectures"] = ["Qwen3_5MoeForConditionalGeneration"]
        config_path.write_text(json.dumps(patched_config, indent=2), encoding="utf-8")
        try:
            print(f"Converting merged HF model to F16 GGUF at {f16_gguf_path}")
            subprocess.run(
                [
                    "python3",
                    f"{LLAMA_CPP_DIR}/convert_hf_to_gguf.py",
                    str(merged_dir),
                    "--outfile",
                    str(f16_gguf_path),
                    "--outtype",
                    "f16",
                ],
                check=True,
            )
        finally:
            original_config["architectures"] = original_architectures
            config_path.write_text(json.dumps(original_config, indent=2), encoding="utf-8")
        outputs_volume.commit()

    if not final_gguf_path.exists():
        quantize_bin = f"{LLAMA_CPP_DIR}/build/bin/llama-quantize"
        if not os.path.exists(quantize_bin):
            quantize_bin = f"{LLAMA_CPP_DIR}/build/bin/quantize"
        print(f"Quantizing merged GGUF to {final_gguf_path.name}")
        subprocess.run(
            [
                quantize_bin,
                "--token-embedding-type", "q8_0",
                "--output-tensor-type", "q6_k",
                "--tensor-type", "attn_gate=q8_0",
                "--tensor-type", "attn_qkv=q8_0",
                "--tensor-type", "ffn_down_shexp=q8_0",
                "--tensor-type", "ffn_gate_shexp=q8_0",
                "--tensor-type", "ffn_up_shexp=q8_0",
                "--tensor-type", "ssm_alpha=q8_0",
                "--tensor-type", "ssm_beta=q8_0",
                "--tensor-type", "ssm_out=q8_0",
                "--tensor-type", "ffn_down_exps=q5_k",
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
        path_in_repo=f"gguf-merged/{final_gguf_path.name}",
        repo_id=hf_repo_id,
        repo_type="model",
    )

    result = {
        "base_model": BASE_MODEL,
        "run_name": run_name,
        "lora_dir": str(lora_dir),
        "merged_dir": str(merged_dir),
        "f16_gguf_path": str(f16_gguf_path),
        "final_gguf_path": str(final_gguf_path),
        "hf_repo_id": hf_repo_id,
        "repo_url": f"https://huggingface.co/{hf_repo_id}",
    }
    print(json.dumps(result, indent=2))
    return result


@app.local_entrypoint()
def main(
    run_name: str = DEFAULT_RUN_NAME,
    hf_repo_id: str = DEFAULT_HF_REPO_ID,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
):
    merge_and_quantize.remote(
        run_name=run_name,
        hf_repo_id=hf_repo_id,
        max_seq_length=max_seq_length,
    )
