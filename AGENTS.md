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

### Verifying scripts without Modal auth

All scripts can be syntax-checked and import-verified locally without valid Modal credentials:

```bash
python3 -m py_compile modal_qwen35_18b_reap_a3b_coding_opus_train.py
python3 -c "import importlib.util; spec = importlib.util.spec_from_file_location('m', 'modal_qwen35_18b_reap_a3b_coding_opus_train.py'); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); print(hasattr(mod, 'app'))"
```

### Key caveats

- PyTorch is **not** required locally — it runs inside Modal containers. The `transformers` import warning about PyTorch missing is expected and harmless.
- Modal tokens (`MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET`) are validated against `api.modal.com` on every `modal run` or `modal token set` call. If tokens are expired or invalid, use `modal token new` on a machine with browser access to regenerate them.
- `modal_compare_qwen_tokenizers.py` has a hardcoded local path in `.add_local_file()`; it will fail unless the path is updated to match the current workspace.
- The `qwen3_5_(4b)_vision.py` and `qwen3_5_moe.py` files are reference Colab notebooks saved as `.py`; they are not meant to be run directly.
- See `progress.md` for the detailed development log and `MODAL_AGENT_GUIDE.md` for Modal usage patterns.
