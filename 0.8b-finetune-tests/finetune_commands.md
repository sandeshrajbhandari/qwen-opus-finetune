# Qwen3.5-0.8B Local Finetune Commands

This uses the locally downloaded model path in this subfolder:

- `0.8b-finetune-tests/Qwen3.5-0.8B`

Requested training settings:

- `epochs = 3`
- `learning_rate = 2e-4`
- save every half epoch

For the Opus dataset currently used in this repo (`2326` rows) and:

- `batch_size = 6`
- `grad_accum = 4`

effective batch size is `24`, so steps per epoch is about `97`, and half epoch is about `49` steps.
Use `--save-steps 49`.

## 1) One-time setup (new `uv` env)

```bash
uv venv .venv-unsloth --python 3.12
source .venv-unsloth/Scripts/activate
uv pip install --python ".venv-unsloth/Scripts/python.exe" --no-index --find-links "F:/code-hdd/ComfyUI/torch_cu128_wheels" torch torchvision torchaudio
uv pip install --python ".venv-unsloth/Scripts/python.exe" -r 0.8b-finetune-tests/requirements.txt
```

Notes:

- `requirements.txt` uses OS markers so Windows gets `triton-windows` and Linux/macOS gets `triton`.
- Always target the env interpreter explicitly with `uv pip --python ".venv-unsloth/Scripts/python.exe"` to avoid accidentally installing into another active env.
- Quick check:

```bash
".venv-unsloth/Scripts/python.exe" -c "import torch, triton, unsloth; print(torch.__version__, triton.__version__)"
```

If the model is not already present:

```bash
".venv-unsloth/Scripts/python.exe" 0.8b-finetune-tests/download_qwen35_08b.py --output-dir 0.8b-finetune-tests
```

## 2) Smoke test (short run)

This keeps your requested core hyperparams but limits runtime with `--max-steps`.

```bash
".venv-unsloth/Scripts/python.exe" 0.8b-finetune-tests/finetune_qwen35_08b_local.py \
  --model-path 0.8b-finetune-tests/Qwen3.5-0.8B \
  --output-dir 0.8b-finetune-tests/outputs/qwen35_08b_smoke \
  --epochs 3 \
  --learning-rate 2e-4 \
  --batch-size 6 \
  --grad-accum 4 \
  --save-steps 49 \
  --max-steps -1
```

## 3) Full finetune run

```bash
".venv-unsloth/Scripts/python.exe" 0.8b-finetune-tests/finetune_qwen35_08b_local.py \
  --model-path 0.8b-finetune-tests/Qwen3.5-0.8B \
  --output-dir 0.8b-finetune-tests/outputs/qwen35_08b_full_e3_lr2e4 \
  --epochs 3 \
  --learning-rate 2e-4 \
  --batch-size 6 \
  --grad-accum 4 \
  --save-steps 49 \
  --load-in-4bit
```

## Optional: save merged model at the end

Add this flag to either command:

```bash
--save-merged-16bit
```
