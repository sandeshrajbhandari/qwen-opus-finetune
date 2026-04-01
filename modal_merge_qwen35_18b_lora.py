"""Merge a saved Qwen3.5-18B LoRA adapter into the base model and optionally upload to Hugging Face.

Usage:
    modal run modal_merge_qwen35_18b_lora.py
    modal run modal_merge_qwen35_18b_lora.py --run-name qwen35-18b-reap-a3b-coding-opus-msl16384-e1_0
    modal run modal_merge_qwen35_18b_lora.py --hf-repo-id your-name/qwen35-18b-reap-a3b-coding-opus-merged
    modal run modal_merge_qwen35_18b_lora.py --skip-merge --run-test-generation
"""

from __future__ import annotations

import json
from pathlib import Path

import modal


APP_NAME = "qwen35-18b-reap-a3b-merge"
BASE_MODEL = "Flagstone8878/Qwen3.5-18B-REAP-A3B-Coding"
HF_CACHE_DIR = "/root/.cache/huggingface"
OUTPUT_ROOT = "/outputs"
MODEL_SNAPSHOT_DIR = f"{HF_CACHE_DIR}/preloaded/flagstone_qwen35_18b_reap_a3b_coding"
DEFAULT_RUN_NAME = "qwen35-18b-reap-a3b-coding-opus-msl8192-e1_0"
DEFAULT_MAX_SEQ_LENGTH = 8192
DEFAULT_TEST_PROMPT = "Write a short Python function that returns the nth Fibonacci number."

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
        "huggingface-hub>=0.34.0",
        "sentencepiece",
        "protobuf",
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
    timeout=60 * 60,
)
def merge_lora(
    run_name: str = DEFAULT_RUN_NAME,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    hf_repo_id: str = "",
    test_prompt: str = DEFAULT_TEST_PROMPT,
    test_max_new_tokens: int = 192,
    skip_merge: bool = False,
    run_test_generation: bool = True,
):
    import torch
    import unsloth
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
    from unsloth import FastLanguageModel

    run_output_dir = Path(OUTPUT_ROOT) / run_name
    lora_dir = run_output_dir / "lora"
    merged_dir = run_output_dir / "merged_16bit"

    if not lora_dir.exists():
        raise FileNotFoundError(f"LoRA adapter directory not found: {lora_dir}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(merged_dir if merged_dir.exists() and skip_merge else lora_dir),
        trust_remote_code=True,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = None
    if not skip_merge or not merged_dir.exists():
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
    else:
        print(f"Skipping merge and using existing merged model at {merged_dir}")

    decoded = ""
    test_error = ""
    if run_test_generation:
        try:
            print(f"Loading merged model from {merged_dir} for smoke test")
            merged_model = AutoModelForCausalLM.from_pretrained(
                str(merged_dir),
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="auto",
            )
            merged_model.eval()
            messages = [
                {"role": "user", "content": test_prompt},
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(text, return_tensors="pt").to(merged_model.device)
            streamer = TextStreamer(tokenizer, skip_prompt=True)
            with torch.inference_mode():
                output = merged_model.generate(
                    **inputs,
                    max_new_tokens=test_max_new_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    use_cache=True,
                    streamer=streamer,
                )
            decoded = tokenizer.decode(output[0], skip_special_tokens=False)
        except Exception as exc:
            test_error = repr(exc)
            print(f"Smoke test generation failed: {test_error}")

    if hf_repo_id:
        if model is None:
            print(f"Reloading merged model from {merged_dir} for upload")
            model, processor = FastLanguageModel.from_pretrained(
                str(merged_dir),
                max_seq_length=max_seq_length,
                load_in_4bit=False,
                fast_inference=False,
            )
        print(f"Uploading merged model to {hf_repo_id}")
        model.push_to_hub_merged(
            hf_repo_id,
            tokenizer,
            save_method="merged_16bit",
            token=True,
        )

    outputs_volume.commit()
    hf_cache_volume.commit()

    result = {
        "base_model": BASE_MODEL,
        "run_name": run_name,
        "lora_dir": str(lora_dir),
        "merged_dir": str(merged_dir),
        "hf_repo_id": hf_repo_id,
        "test_prompt": test_prompt,
        "test_max_new_tokens": test_max_new_tokens,
        "skip_merge": skip_merge,
        "run_test_generation": run_test_generation,
        "test_error": test_error,
        "decoded_output_preview": decoded[-2000:],
    }
    print(json.dumps(result, indent=2))
    return result


@app.local_entrypoint()
def main(
    run_name: str = DEFAULT_RUN_NAME,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    hf_repo_id: str = "",
    test_prompt: str = DEFAULT_TEST_PROMPT,
    test_max_new_tokens: int = 192,
    skip_merge: bool = False,
    run_test_generation: bool = True,
):
    merge_lora.remote(
        run_name=run_name,
        max_seq_length=max_seq_length,
        hf_repo_id=hf_repo_id,
        test_prompt=test_prompt,
        test_max_new_tokens=test_max_new_tokens,
        skip_merge=skip_merge,
        run_test_generation=run_test_generation,
    )
