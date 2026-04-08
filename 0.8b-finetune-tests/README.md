# 0.8B Finetune Tests

Local Windows finetune workflow for `Qwen3.5-0.8B` using a dedicated `uv` virtual environment.

## Environment

Create and use a clean env in repo root:

```bash
uv venv .venv-unsloth --python 3.12
source .venv-unsloth/Scripts/activate
```

Install Torch from local CUDA 12.8 wheels, then install project requirements:

```bash
uv pip install --python ".venv-unsloth/Scripts/python.exe" --no-index --find-links "F:/code-hdd/ComfyUI/torch_cu128_wheels" torch torchvision torchaudio
uv pip install --python ".venv-unsloth/Scripts/python.exe" -r 0.8b-finetune-tests/requirements.txt
```

Why this is set up this way:

- Keeps all packages isolated from other local envs.
- Avoids cross-env `pip` installs by pinning `uv pip` to the env Python path.
- Uses `triton-windows` automatically on Windows via requirement markers.

Quick verify:

```bash
".venv-unsloth/Scripts/python.exe" -c "import torch, triton, unsloth; print(torch.__version__, triton.__version__, unsloth.__version__)"
```

## Run

Smoke test:

```bash
".venv-unsloth/Scripts/python.exe" 0.8b-finetune-tests/finetune_qwen35_08b_local.py --model-path 0.8b-finetune-tests/Qwen3.5-0.8B --output-dir 0.8b-finetune-tests/outputs/qwen35_08b_smoke --epochs 3 --learning-rate 2e-4 --batch-size 6 --grad-accum 4 --save-steps 49 --max-steps 2
```

Full run:

```bash
".venv-unsloth/Scripts/python.exe" 0.8b-finetune-tests/finetune_qwen35_08b_local.py --model-path 0.8b-finetune-tests/Qwen3.5-0.8B --output-dir 0.8b-finetune-tests/outputs/qwen35_08b_full_e3_lr2e4 --epochs 3 --learning-rate 2e-4 --batch-size 6 --grad-accum 4 --save-steps 49 --load-in-4bit
```

For multi-line versions of these commands, see `0.8b-finetune-tests/finetune_commands.md`.
