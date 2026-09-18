# Running angel-ai on Google Colab (or a similar cloud GPU notebook)

**Quickest path**: open [`notebooks/colab_train.ipynb`](../notebooks/colab_train.ipynb)
directly in Colab — [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ianktoo/angel-ai/blob/main/notebooks/colab_train.ipynb) —
which walks through every step below as runnable cells (bootstrap, train,
resume, evaluate, export, view results). The rest of this doc explains what
those cells do and why, for anyone scripting it outside a notebook.

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

## Bootstrap

In a notebook cell:

```python
!git clone https://github.com/ianktoo/angel-ai angel-ai
%cd angel-ai
!bash scripts/colab_bootstrap.sh
```

`scripts/colab_bootstrap.sh` installs `uv` (`pip install uv`, since Colab
doesn't ship it) and runs `uv sync --extra cuda` to install the CUDA-enabled
dependency set.

**Mount Google Drive in a separate cell, directly** (not via `!bash`):

```python
from google.colab import drive
drive.mount("/content/drive")
```

This has to run as an actual notebook cell rather than inside the bootstrap
script. `drive.mount()` talks to the Colab frontend through the notebook's
own IPython kernel (via `get_ipython()`); a subprocess spawned by `!bash
colab_bootstrap.sh` has no such kernel, so calling it from there fails with
`AttributeError: 'NoneType' object has no attribute 'kernel'` (confirmed by
running it that way first).

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

Resuming after a session reset (re-run the clone, bootstrap, and Drive-mount
cells first — the VM and its filesystem are wiped, only Drive survives —
then):

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
