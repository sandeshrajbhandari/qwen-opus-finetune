"""Analyze and filter the Opus 3000x dataset by token length, then upload to Hugging Face.

Usage:
    modal run modal_filter_opus_dataset.py
    modal run modal_filter_opus_dataset.py --max-length 8192
    modal run modal_filter_opus_dataset.py --repo-id your-name/Opus-4.6-Reasoning-3000x-filtered-8k
"""

from __future__ import annotations

import json
from pathlib import Path

import modal


APP_NAME = "opus-3000x-filter-8k"
BASE_MODEL = "Flagstone8878/Qwen3.5-18B-REAP-A3B-Coding"
SOURCE_DATASET = "nohurry/Opus-4.6-Reasoning-3000x-filtered"
SOURCE_SPLIT = "train"
SOURCE_REVISION = "main"
HF_CACHE_DIR = "/root/.cache/huggingface"
OUTPUT_ROOT = "/outputs"
MODEL_SNAPSHOT_DIR = f"{HF_CACHE_DIR}/preloaded/flagstone_qwen35_18b_reap_a3b_coding"

DEFAULT_MAX_LENGTH = 8192

app = modal.App(APP_NAME)

hf_cache_volume = modal.Volume.from_name("hf-cache-qwen35-18b-reap-a3b", create_if_missing=True)
outputs_volume = modal.Volume.from_name("qwen35-opus-filter-outputs", create_if_missing=True)

huggingface_secret = modal.Secret.from_name(
    "huggingface-secret", required_keys=["HF_TOKEN"]
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "datasets>=3.6.0,<4.0.0",
        "huggingface-hub>=0.34.0",
        "transformers>=4.57.0",
        "sentencepiece",
        "protobuf",
        "numpy",
    )
)


@app.function(
    image=image,
    volumes={
        HF_CACHE_DIR: hf_cache_volume,
        OUTPUT_ROOT: outputs_volume,
    },
    secrets=[huggingface_secret],
    timeout=60 * 60,
)
def filter_dataset(
    max_length: int = DEFAULT_MAX_LENGTH,
    repo_id: str = "",
    private: bool = False,
):
    import numpy as np
    from datasets import load_dataset
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoTokenizer

    output_dir = Path(OUTPUT_ROOT) / f"opus-filter-lte-{max_length}"
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = Path(MODEL_SNAPSHOT_DIR)
    if not (snapshot_path / "tokenizer.json").exists() and not (snapshot_path / "tokenizer_config.json").exists():
        print(f"Downloading tokenizer/model snapshot for {BASE_MODEL} ...")
        snapshot_download(
            repo_id=BASE_MODEL,
            local_dir=str(snapshot_path),
            local_dir_use_symlinks=False,
        )
        hf_cache_volume.commit()

    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot_path) if snapshot_path.exists() else BASE_MODEL,
        trust_remote_code=True,
        use_fast=False,
    )

    dataset = load_dataset(
        SOURCE_DATASET,
        split=SOURCE_SPLIT,
        revision=SOURCE_REVISION,
    )

    def format_text(sample):
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

    formatted = dataset.map(
        format_text,
        remove_columns=[],
        desc="Formatting chat text",
    )

    def add_length(sample):
        token_count = len(
            tokenizer(
                sample["text"],
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]
        )
        return {"token_count": token_count}

    with_lengths = formatted.map(
        add_length,
        desc="Computing token lengths",
    )

    token_counts = with_lengths["token_count"]
    filtered = with_lengths.filter(
        lambda sample: sample["token_count"] <= max_length,
        desc=f"Filtering rows <= {max_length} tokens",
    )

    stats = {
        "source_dataset": SOURCE_DATASET,
        "source_split": SOURCE_SPLIT,
        "source_revision": SOURCE_REVISION,
        "base_model": BASE_MODEL,
        "max_length": max_length,
        "rows_total": len(with_lengths),
        "rows_kept": len(filtered),
        "rows_removed": len(with_lengths) - len(filtered),
        "kept_fraction": len(filtered) / len(with_lengths) if len(with_lengths) else 0.0,
        "token_count_min": int(np.min(token_counts)) if token_counts else None,
        "token_count_p50": float(np.percentile(token_counts, 50)) if token_counts else None,
        "token_count_p90": float(np.percentile(token_counts, 90)) if token_counts else None,
        "token_count_p95": float(np.percentile(token_counts, 95)) if token_counts else None,
        "token_count_p99": float(np.percentile(token_counts, 99)) if token_counts else None,
        "token_count_max": int(np.max(token_counts)) if token_counts else None,
    }

    stats_path = output_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))

    filtered_jsonl_path = output_dir / "filtered.jsonl"
    filtered.to_json(str(filtered_jsonl_path))

    if not repo_id:
        api = HfApi(token=True)
        username = api.whoami()["name"]
        repo_id = f"{username}/Opus-4.6-Reasoning-3000x-filtered-{max_length}"

    print(f"Uploading filtered dataset to {repo_id} ...")
    filtered.push_to_hub(
        repo_id,
        private=private,
        token=True,
    )

    outputs_volume.commit()
    hf_cache_volume.commit()

    result = {
        "repo_id": repo_id,
        "stats_path": str(stats_path),
        "filtered_jsonl_path": str(filtered_jsonl_path),
        **stats,
    }
    print(json.dumps(result, indent=2))
    return result


@app.local_entrypoint()
def main(
    max_length: int = DEFAULT_MAX_LENGTH,
    repo_id: str = "",
    private: bool = False,
):
    filter_dataset.remote(
        max_length=max_length,
        repo_id=repo_id,
        private=private,
    )
