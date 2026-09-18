# Running angel-ai on Google Colab (or a similar cloud GPU notebook)

This isn't a separate codebase or branch — Colab's NVIDIA GPU is reached
through the same `backend=cuda` config path used by any local NVIDIA box.
The only Colab-specific concerns are: `uv` isn't preinstalled, and the
filesystem is ephemeral (wiped when the session recycles), so checkpoints
and MLflow run data need to live on mounted Google Drive to survive that.

This section is written from how Colab behaves at the time of writing;
Colab's default Python/CUDA versions change periodically, so if `uv sync`
fails, check what Python/CUDA version the runtime currently ships and adjust
`requires-python` / the `cuda` extra's torch pin accordingly — treat this as
a starting point to verify, not a guarantee.

## One-line bootstrap

In a notebook cell:

```python
!git clone <your-repo-url> angel-ai
%cd angel-ai
!bash scripts/colab_bootstrap.sh
```

`scripts/colab_bootstrap.sh` does three things:
1. Installs `uv` (`pip install uv`, since Colab doesn't ship it).
2. Runs `uv sync --extra cuda` to install the CUDA-enabled dependency set.
3. Mounts Google Drive at `/content/drive` (interactive auth prompt appears
   in the notebook — approve it there).

## Persisting checkpoints and MLflow data across sessions

Colab sessions are ephemeral: anything under `/content` that isn't on Drive
disappears when the runtime recycles. Point `output_dir` at a Drive path via
a config override so checkpoints and MLflow's local store survive:

```bash
uv run python -m angel_ai.training \
  backend=cuda \
  training=qlora \
  output_dir=/content/drive/MyDrive/angel-ai-runs/run-001
```

Resuming after a session reset (re-run the bootstrap cell first, then):

```bash
uv run python -m angel_ai.training \
  backend=cuda \
  output_dir=/content/drive/MyDrive/angel-ai-runs/run-001 \
  training.resume_from_checkpoint=latest
```

## Notes specific to the cloud/CUDA path

- QLoRA (`training=qlora`) is available here (unlike the local DirectML
  backend), since `bitsandbytes` is CUDA-only — this is the natural place to
  run it if you want to compare against the DirectML LoRA results from local
  hardware.
- Colab GPU availability and VRAM vary by tier/session; if you hit an
  out-of-memory error, lower `training.per_device_train_batch_size` and
  raise `training.gradient_accumulation_steps` to compensate, same as on the
  local iGPU.
- `mlflow ui --backend-store-uri sqlite:///<output_dir>/mlruns/mlflow.db` can
  be run locally after copying (or `rclone`-ing) the Drive-persisted
  `mlruns/` folder back down, since Colab doesn't conveniently expose a
  long-running UI port.
