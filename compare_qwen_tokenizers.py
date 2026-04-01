"""Compare tokenizer/config behavior between two Hugging Face model repos or local paths.

Usage:
    python3 compare_qwen_tokenizers.py \
      --base-model Qwen/Qwen3.5-35B-A3B \
      --custom-model Flagstone8878/Qwen3.5-18B-REAP-A3B-Coding
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


COMPARE_FILES = [
    "config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
    "chat_template.json",
    "tokenizer.json",
]

FETCH_PATTERNS = [
    "config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
    "chat_template.json",
    "tokenizer.json",
    "tokenizer.model",
    "merges.txt",
    "vocab.json",
]

SAMPLE_PROMPTS = [
    "Write a Python function that returns the nth Fibonacci number.",
    "Explain what gradient checkpointing does.",
    "Solve: If x + 3 = 11, what is x?",
]


def _load_json_if_exists(base: Path, name: str) -> Any | None:
    path = base / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return {"_unparsed_text_preview": path.read_text()[:2000]}


def _simple_diff(a: Any, b: Any) -> dict[str, Any]:
    if a == b:
        return {"equal": True}
    if isinstance(a, dict) and isinstance(b, dict):
        keys = sorted(set(a.keys()) | set(b.keys()))
        diff = {}
        for key in keys:
            if a.get(key) != b.get(key):
                diff[key] = {"base": a.get(key), "custom": b.get(key)}
        return {"equal": False, "diff": diff}
    return {"equal": False, "base": a, "custom": b}


def compare_models(base_model: str, custom_model: str) -> dict[str, Any]:
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    cache_dir = Path(".hf_compare_cache")
    cache_dir.mkdir(exist_ok=True)

    def resolve(model_ref: str, local_name: str) -> Path:
        path = Path(model_ref)
        if path.exists():
            return path
        local_dir = cache_dir / local_name
        snapshot_download(
            repo_id=model_ref,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            allow_patterns=FETCH_PATTERNS,
            ignore_patterns=["*.safetensors", "*.bin", "*.pt"],
        )
        return local_dir

    base_path = resolve(base_model, "base_model")
    custom_path = resolve(custom_model, "custom_model")

    file_diffs = {}
    for name in COMPARE_FILES:
        base_json = _load_json_if_exists(base_path, name)
        custom_json = _load_json_if_exists(custom_path, name)
        file_diffs[name] = _simple_diff(base_json, custom_json)

    base_tokenizer = AutoTokenizer.from_pretrained(
        str(base_path),
        trust_remote_code=True,
        use_fast=False,
    )
    custom_tokenizer = AutoTokenizer.from_pretrained(
        str(custom_path),
        trust_remote_code=True,
        use_fast=False,
    )

    special_tokens = {
        "base": {
            "eos_token": base_tokenizer.eos_token,
            "eos_token_id": base_tokenizer.eos_token_id,
            "bos_token": base_tokenizer.bos_token,
            "bos_token_id": base_tokenizer.bos_token_id,
            "pad_token": base_tokenizer.pad_token,
            "pad_token_id": base_tokenizer.pad_token_id,
            "unk_token": base_tokenizer.unk_token,
            "unk_token_id": base_tokenizer.unk_token_id,
            "vocab_size": base_tokenizer.vocab_size,
        },
        "custom": {
            "eos_token": custom_tokenizer.eos_token,
            "eos_token_id": custom_tokenizer.eos_token_id,
            "bos_token": custom_tokenizer.bos_token,
            "bos_token_id": custom_tokenizer.bos_token_id,
            "pad_token": custom_tokenizer.pad_token,
            "pad_token_id": custom_tokenizer.pad_token_id,
            "unk_token": custom_tokenizer.unk_token,
            "unk_token_id": custom_tokenizer.unk_token_id,
            "vocab_size": custom_tokenizer.vocab_size,
        },
    }

    prompt_checks = []
    for prompt in SAMPLE_PROMPTS:
        base_ids = base_tokenizer(prompt, add_special_tokens=False)["input_ids"]
        custom_ids = custom_tokenizer(prompt, add_special_tokens=False)["input_ids"]
        prompt_checks.append(
            {
                "prompt": prompt,
                "base_len": len(base_ids),
                "custom_len": len(custom_ids),
                "same_ids": base_ids == custom_ids,
                "base_first_32": base_ids[:32],
                "custom_first_32": custom_ids[:32],
            }
        )

    chat_template_check = {}
    if hasattr(base_tokenizer, "apply_chat_template") and hasattr(custom_tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": "Hello, write a haiku about code."}]
        try:
            base_chat = base_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as exc:
            base_chat = None
            base_chat_error = str(exc)
        else:
            base_chat_error = ""

        try:
            custom_chat = custom_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as exc:
            custom_chat = None
            custom_chat_error = str(exc)
        else:
            custom_chat_error = ""

        chat_template_check = {
            "same_chat_template_render": base_chat == custom_chat,
            "base_chat_preview": base_chat[:500] if isinstance(base_chat, str) else None,
            "custom_chat_preview": custom_chat[:500] if isinstance(custom_chat, str) else None,
            "base_chat_error": base_chat_error,
            "custom_chat_error": custom_chat_error,
        }

    return {
        "base_model": base_model,
        "custom_model": custom_model,
        "base_path": str(base_path),
        "custom_path": str(custom_path),
        "special_tokens": special_tokens,
        "file_diffs": file_diffs,
        "prompt_checks": prompt_checks,
        "chat_template_check": chat_template_check,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--custom-model", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    result = compare_models(args.base_model, args.custom_model)
    rendered = json.dumps(result, indent=2)
    print(rendered)

    if args.output:
        Path(args.output).write_text(rendered)


if __name__ == "__main__":
    main()
