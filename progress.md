# Progress Log

## April 1, 2026

1. Started from the goal of fine-tuning Qwen 3.5 on the Opus-distilled `3000x` dataset using Modal, with an initial preference for `2048` context and at least `1` epoch.

2. Inspected the workspace and found:
   - [qwen3_5_(4b)_vision.py](/Users/sandeshrajbhandari/Documents/gen-ai/qwen-opus-finetune/qwen3_5_(4b)_vision.py)
   - [qwen3_5_moe.py](/Users/sandeshrajbhandari/Documents/gen-ai/qwen-opus-finetune/qwen3_5_moe.py)
   - existing Modal helper scripts like [prune_qwen.py](/Users/sandeshrajbhandari/Documents/gen-ai/qwen-opus-finetune/prune_qwen.py) and [upload_to_hf.py](/Users/sandeshrajbhandari/Documents/gen-ai/qwen-opus-finetune/upload_to_hf.py)

3. Verified that the current `main` split of `nohurry/Opus-4.6-Reasoning-3000x-filtered` contains `2326` rows and has `problem`, `thinking`, and `solution`-style fields.

4. Created an initial Modal training script for `unsloth/Qwen3.5-4B`:
   - [modal_qwen35_4b_opus_train.py](/Users/sandeshrajbhandari/Documents/gen-ai/qwen-opus-finetune/modal_qwen35_4b_opus_train.py)
   - mirrored the style of the Unsloth reference notebook
   - formatted samples into Qwen chat format with `<think> ... </think>`

5. Added a second training script targeting `Flagstone8878/Qwen3.5-18B-REAP-A3B-Coding`:
   - [modal_qwen35_18b_reap_a3b_coding_opus_train.py](/Users/sandeshrajbhandari/Documents/gen-ai/qwen-opus-finetune/modal_qwen35_18b_reap_a3b_coding_opus_train.py)

6. Tuned the `18B` script for long-context training:
   - moved to `16384` default context
   - set conservative batch settings
   - discussed cheaper GPU options versus A100

7. Switched the `18B` script to `16-bit` LoRA loading to align with the reference notebook recommendation rather than `4-bit`.

8. Added short-run benchmarking support:
   - `max_steps`
   - automatic estimates for `seconds_per_step`
   - estimated full-train runtime

9. Added response-only training support to both scripts using Unsloth’s masking helper.

10. Hit an Unsloth loader failure for the Flagstone MoE checkpoint:
    - `Qwen3_5MoeTextConfig`
    - routed incorrectly to `AutoModelForImageTextToText`

11. Investigated the MoE reference notebook [qwen3_5_moe.py](/Users/sandeshrajbhandari/Documents/gen-ai/qwen-opus-finetune/qwen3_5_moe.py) and aligned the `18B` script more closely with it:
    - `fast_inference=False`
    - MoE LoRA targets including `gate_up_proj`
    - `train_on_responses_only(...)`
    - `UNSLOTH_MOE_DISABLE_AUTOTUNE=1`

12. Discovered the Flagstone checkpoint’s config shape likely caused the loader issue:
    - top-level/text config shape did not match what Unsloth expected from the official Qwen MoE checkpoint
    - user noted the difference between `qwen3_5_moe_text` and the official multimodal-style wrapper config

13. Implemented local snapshot patching for the Flagstone checkpoint:
    - patched `config.json` into a Qwen-compatible structure
    - preserved the original config as `config.original.json`
    - wrote patched metadata as `config.patched.json`

14. Added local tokenizer metadata patching:
    - patched `tokenizer_config.json`
    - patched `special_tokens_map.json`
    - replaced placeholder `"<EOS_TOKEN>"` metadata with `"<|im_end|>"`

15. Added a dedicated CPU pre-download step so the expensive GPU container does not waste time redownloading the model:
    - `download_model()`
    - `validate_patched_config()`

16. Worked through multiple trainer compatibility issues caused by the pinned reference stack:
    - `SFTConfig` uses `max_length` instead of `max_seq_length`
    - `SFTTrainer` expects `processing_class=` instead of `tokenizer=`
    - `assistant_only_loss=True` was invalid for the rendered `text` dataset format

17. Settled on the same pattern as the reference notebook:
    - rendered `text` field
    - `SFTTrainer`
    - `train_on_responses_only(...)`

18. Fixed tokenizer handling in the fine-tune script:
    - stopped using raw tokenizer backend objects
    - restored `AutoTokenizer.from_pretrained(..., trust_remote_code=True, use_fast=False)`
    - ensured `apply_chat_template(...)` works

19. Verified that a `20-step` short run on the `18B` setup completed and produced an ETA estimate.

20. Ran a successful full `1` epoch training job with a faster and more practical config:
    - dataset: `nohurry/Opus-4.6-Reasoning-3000x-filtered`
    - rows: `2326`
    - `max_seq_length=8192`
    - `per_device_train_batch_size=2`
    - `gradient_accumulation_steps=16`
    - `learning_rate=2e-5`
    - `lora_r=16`
    - `lora_alpha=32`
    - run name: `qwen35-18b-reap-a3b-coding-opus-msl8192-e1_0`
    - train loss: about `0.7515`
    - runtime: about `25.9 minutes`
    - peak reserved memory: about `56.7 GB`

21. Added checkpoint resume support to the training script:
    - `save_steps`
    - `resume_from_checkpoint`
    - `latest` checkpoint resolution

22. Added a dataset analysis and filter pipeline:
    - [modal_filter_opus_dataset.py](/Users/sandeshrajbhandari/Documents/gen-ai/qwen-opus-finetune/modal_filter_opus_dataset.py)
    - computes token length stats
    - filters rows by max token length
    - uploads filtered dataset to HF
    - training script now supports `dataset_name`, `dataset_split`, and `dataset_revision` overrides

23. Added a merge script for the finished LoRA:
    - [modal_merge_qwen35_18b_lora.py](/Users/sandeshrajbhandari/Documents/gen-ai/qwen-opus-finetune/modal_merge_qwen35_18b_lora.py)
    - merges LoRA to `merged_16bit`
    - optionally uploads to Hugging Face
    - includes a smoke-test generation step

24. Hit a Triton MoE inference shared-memory error during smoke-test generation on L40S:
    - `OutOfResources: shared memory`
    - required shared memory exceeded L40S hardware limit
    - concluded this is a kernel-path issue, not a merge failure

25. Updated the merge script so the smoke test:
    - loads the saved merged model from disk with plain `transformers`
    - records generation errors without failing the whole merge

26. Added a way to test an already merged model without re-merging:
    - `--skip-merge`
    - `--run-test-generation`

27. Current known state:
    - the `18B` finetune script works for training
    - the `1` epoch `8192`-context run completed successfully
    - local config/tokenizer patching is required for the Flagstone checkpoint
    - merged-model smoke testing may still hit MoE kernel limits on L40S depending on inference path

## Current Main Files

- [modal_qwen35_18b_reap_a3b_coding_opus_train.py](/Users/sandeshrajbhandari/Documents/gen-ai/qwen-opus-finetune/modal_qwen35_18b_reap_a3b_coding_opus_train.py)
- [modal_merge_qwen35_18b_lora.py](/Users/sandeshrajbhandari/Documents/gen-ai/qwen-opus-finetune/modal_merge_qwen35_18b_lora.py)
- [modal_filter_opus_dataset.py](/Users/sandeshrajbhandari/Documents/gen-ai/qwen-opus-finetune/modal_filter_opus_dataset.py)
- [modal_qwen35_4b_opus_train.py](/Users/sandeshrajbhandari/Documents/gen-ai/qwen-opus-finetune/modal_qwen35_4b_opus_train.py)

## Current Best Known Training Command

```bash
modal run /Users/sandeshrajbhandari/Documents/gen-ai/qwen-opus-finetune/modal_qwen35_18b_reap_a3b_coding_opus_train.py --max-seq-length 8192 --per-device-train-batch-size 2 --gradient-accumulation-steps 16 --learning-rate 2e-5 --lora-r 16 --lora-alpha 32 --epochs 1 --max-steps -1
```

## Current Best Known Resume Command

```bash
modal run /Users/sandeshrajbhandari/Documents/gen-ai/qwen-opus-finetune/modal_qwen35_18b_reap_a3b_coding_opus_train.py --max-seq-length 8192 --per-device-train-batch-size 2 --gradient-accumulation-steps 16 --learning-rate 2e-5 --lora-r 16 --lora-alpha 32 --resume-from-checkpoint latest
```

## Current Best Known Merge Command

```bash
python3 -m modal run /Users/sandeshrajbhandari/Documents/gen-ai/qwen-opus-finetune/modal_merge_qwen35_18b_lora.py
```

## Current Best Known Smoke-Test-Only Command

```bash
python3 -m modal run /Users/sandeshrajbhandari/Documents/gen-ai/qwen-opus-finetune/modal_merge_qwen35_18b_lora.py --skip-merge --run-test-generation --test-prompt "Write a Python function that validates an email address."
```
