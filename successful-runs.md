https://modal.com/apps/sandeshrb87/main/ap-T2xnaBiA2yBtgqaRF6Rpx4?start=1775004929.2&end=1775008529.2&live=true&fcId=fc-01KN3BEPK6YRR7BRZFN4F4BSW5&inputId=in-01KN3BEPKQ5R0H09B952GB1H9Q&activeTab=functions&functionId=fu-Nu42J7CQqi2WgbIYgz8Tbw&functionSection=calls&limit=100&includeLogContext=false&useInputsTable=true

this run 2 epochs, good memory util, but low lr so didn't quickly reduce the loss. try with 2e-4 instead of 6e-5

## 2e-4 run
2 epochs
got upto 0.5748 epochs

'train_runtime': '2719', 'train_samples_per_second': '1.711', 'train_steps_per_second': '0.071', 'train_loss': '0.5748', 'epoch': '2'}

100%|██████████| 194/194 [45:18<00:00, 12.26s/it]
100%|██████████| 194/194 [45:18<00:00, 14.02s/it]
Saving LoRA adapter...
{
  "base_model": "Flagstone8878/Qwen3.5-18B-REAP-A3B-Coding",
  "model_load_path": "/root/.cache/huggingface/preloaded/flagstone_qwen35_18b_reap_a3b_coding",
  "continue_from_run_name": null,
  "dataset_name": "nohurry/Opus-4.6-Reasoning-3000x-filtered",
  "dataset_split": "train",
  "dataset_revision": "main",
  "dataset_rows": 2326,
  "max_seq_length": 8192,
  "epochs": 2.0,
  "max_steps": -1,
  "resume_from_checkpoint": null,
  "save_steps": 20,
  "per_device_train_batch_size": 6,
  "gradient_accumulation_steps": 4,
  "effective_batch_size": 24,
  "steps_per_epoch": 97,
  "target_full_train_steps": 194,
  "observed_steps": 194,
  "learning_rate": 0.0002,
  "lora_r": 32,
  "lora_alpha": 64,
  "load_in_4bit": false,
  "load_in_16bit": true,
  "output_dir": "/outputs/qwen35-18b-reap-a3b-coding-opus-msl8192-e2_0-lr2e4",
  "setup_seconds_before_train": 66.054580499,
  "train_runtime_seconds": 2718.9457,
  "train_loss": 0.5748463515768346,
  "seconds_per_step": 14.015184020618557,
  "estimated_full_train_runtime_seconds": 2718.9457,
  "estimated_full_job_runtime_seconds": 2785.0002804990004,
  "peak_reserved_memory_gb": 71.09,
  "peak_training_memory_gb": 32.07,
  "peak_reserved_memory_pct": 89.702,
  "peak_training_memory_pct": 40.466
}
https://modal.com/apps/sandeshrb87/main/ap-4HQxT95cyu4j1aX4b54NyX
