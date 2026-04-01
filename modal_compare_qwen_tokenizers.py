"""Run tokenizer/config comparison on Modal without downloading model weights locally."""

from __future__ import annotations

import modal

app = modal.App("compare-qwen-tokenizers")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "huggingface-hub>=0.34.0",
        "transformers>=4.57.0",
        "sentencepiece",
        "protobuf",
        "jinja2",
    )
    .add_local_file(
        "/Users/sandeshrajbhandari/Documents/gen-ai/qwen-opus-finetune/compare_qwen_tokenizers.py",
        "/root/compare_qwen_tokenizers.py",
    )
)


@app.function(
    image=image,
    timeout=30 * 60,
)
def compare(
    base_model: str = "Qwen/Qwen3.5-35B-A3B",
    custom_model: str = "Flagstone8878/Qwen3.5-18B-REAP-A3B-Coding",
):
    from compare_qwen_tokenizers import compare_models

    result = compare_models(base_model, custom_model)
    import json

    print(json.dumps(result, indent=2))
    return result


@app.local_entrypoint()
def main(
    base_model: str = "Qwen/Qwen3.5-35B-A3B",
    custom_model: str = "Flagstone8878/Qwen3.5-18B-REAP-A3B-Coding",
):
    compare.remote(base_model=base_model, custom_model=custom_model)
