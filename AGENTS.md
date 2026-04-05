# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

This is a collection of Python scripts for fine-tuning Qwen3.5 LLMs on reasoning-distilled datasets using [Modal](https://modal.com/) for remote GPU compute. There are no local services, web servers, or databases. All heavy compute (training, merging, pruning) runs remotely on Modal's GPU containers.

### Local development dependencies

Only `modal`, `huggingface-hub`, `transformers`, `sentencepiece`, `protobuf`, and `ruff` are needed locally. Install with:

```
pip install modal huggingface-hub transformers sentencepiece protobuf ruff
```

### Required secrets (environment variables)

- `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` — needed to run any `modal run` / `modal deploy` command. Create via `modal token new`.
- `HF_TOKEN` — needed by Modal containers to download models/datasets from Hugging Face. Configured as a Modal secret: `modal secret create huggingface-secret HF_TOKEN=<token>`.

### Linting

```bash
ruff check .
```

The codebase has pre-existing lint warnings (F401 unused imports, etc.) that are intentional side-effect imports for `unsloth`.

### Running scripts

All `modal_*.py` and `prune_qwen.py` / `upload_to_hf.py` scripts run remotely via Modal:

```bash
modal run modal_qwen35_18b_reap_a3b_coding_opus_train.py
```

The only script designed to run locally is `compare_qwen_tokenizers.py`:

```bash
python3 compare_qwen_tokenizers.py --base-model <model> --custom-model <model>
```

### Key caveats

- PyTorch is **not** required locally — it runs inside Modal containers. The `transformers` import warning about PyTorch missing is expected and harmless.
- The `qwen3_5_(4b)_vision.py` and `qwen3_5_moe.py` files are reference Colab notebooks saved as `.py`; they are not meant to be run directly.
- See `progress.md` for the detailed development log and `MODAL_AGENT_GUIDE.md` for Modal usage patterns.
