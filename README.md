# angel-ai

**angel-ai** is a research project exploring how far LLM finetuning,
optimization, and evaluation can go on hardware you already own, instead of
defaulting to a rented CUDA GPU. The starting hypothesis: consumer AMD/Intel
integrated GPUs (via DirectML) and NPUs (via ONNX Runtime execution
providers) are underused for this, and a config-driven pipeline can target
them, a CUDA box, or CPU-only, without rewriting code per environment.

This is exploratory, not a polished product. Some paths here are
well-trodden (CUDA training, CPU eval); others — training on an AMD iGPU via
DirectML, exporting a finetuned model for an NPU's Vitis AI execution
provider — are newer, less standardized, and more likely to need
environment-specific tweaking (driver versions, SDK quirks) than a
guaranteed one-command flow. Treat results and instructions here as findings
from an ongoing investigation, and expect the DirectML/NPU paths in
particular to need debugging on hardware other than the one this was
developed against.

## Why "angel-ai"

Leverages the hardware you already have ("existing angels"), rather than
chasing expensive GPUs.

## Architecture at a glance

- **Backend abstraction** (`angel_ai/backends/`): a `Backend` interface with
  `directml`, `cuda`, and `cpu` implementations. Every pipeline stage depends
  on this interface, never on a concrete backend — hardware is a config
  choice (`backend: directml|cuda|cpu|auto`), not a code fork.
- **Config-driven** (Hydra, `configs/`): backend, model, training strategy
  (LoRA vs QLoRA), dataset, and optimization target are all config groups,
  composable and overridable from the CLI.
- **Pipeline stages** (`angel_ai/{training,optimize,evaluation}/`):
  finetuning (LoRA/PEFT), optimization (Optuna hyperparameter sweeps +
  ONNX export/quantization), and evaluation (loss/perplexity), each backend-
  agnostic except where hardware genuinely constrains it (e.g. QLoRA is
  CUDA-only).
- **Local experiment tracking** (`angel_ai/tracking/`): MLflow with a local
  file store — no server, no account.

See [`docs/architecture.md`](docs/architecture.md) if present, or the
module docstrings in `src/angel_ai/`, for more detail.

## Hardware this was developed against

- AMD Ryzen AI 5 340 (6C/12T) with a Radeon 840M integrated GPU, 15GB shared
  system RAM, and an XDNA NPU.
- Training runs on the iGPU via DirectML or on CPU. **The NPU is
  inference-only** — AMD's current Ryzen AI NPU acceleration path
  (ONNX Runtime GenAI + Vitis AI execution provider) does not support
  training. The intended flow is: train with LoRA on the iGPU, then export +
  quantize to ONNX and run inference on the NPU.
- 15GB of *shared* RAM (CPU and iGPU draw from the same pool) is the binding
  constraint here, not raw compute — default configs target small models
  (~0.5B–1.5B parameters) with LoRA, gradient checkpointing, and small batch
  sizes plus gradient accumulation.

If you're running this on different hardware, the config system should
adapt, but the DirectML/NPU-specific code paths were only verified against
the above — expect to file/fix issues.

## Setup

Requires [`uv`](https://docs.astral.sh/uv/). `torch-directml` only supports
Python 3.8–3.12, so this project pins its own interpreter independent of
whatever Python is installed system-wide:

```bash
uv python install 3.11
uv sync --extra directml   # this machine (AMD iGPU via DirectML)
# or: uv sync --extra cuda   # NVIDIA GPU, local or cloud (see docs/colab.md)
# or: uv sync --extra cpu    # CPU-only fallback
```

Optionally install [`just`](https://github.com/casey/just) as a task runner
(`uv tool install rust-just` — **not** `uv tool install just`, which installs
an unrelated PyPI package of the same name) — every `just <task>` below has a
plain-script equivalent under `scripts/` (both `.sh` and `.ps1`) if you'd
rather not.

## Running the pipeline

```bash
just setup                 # install Python 3.11 + sync deps for the detected backend
just check                 # lint + validate configs (no model loading)
just dry-run                # 1-step sanity pass through the full pipeline, tiny model/dataset
just train backend=directml model=qwen2_5_0_5b training=lora data=tiny
just infer backend=directml                       # evaluate the latest checkpoint
just export target=npu backend=directml           # export + quantize for the NPU
just sweep                                          # Optuna hyperparameter search
just test                                          # fast unit tests (no model loading)
just test-slow                                      # + real tiny-model smoke tests
```

Without `just`, the equivalent commands are `uv run python -m
angel_ai.training [overrides...]`, `angel_ai.evaluation`,
`angel_ai.optimize.export`, `angel_ai.optimize.sweep`, or use the scripts in
`scripts/`. All Hydra config overrides (`backend=...`, `model=...`,
`training=...`, `data=...`) work identically either way.

Compare runs with `mlflow ui --backend-store-uri sqlite:///<output_dir>/mlruns/mlflow.db`
(MLflow's plain-folder file store is in maintenance mode as of MLflow 2.x, so
tracking data lives in a local SQLite file instead — still local, still free,
no server or account).

## Dataset format

Datasets are JSONL, one chat-style record per line:

```jsonl
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is LoRA?"}, {"role": "assistant", "content": "..."}]}
```

- `messages` is required; `system` is optional per record and falls back to
  `data.default_system_prompt` from config.
- Files live at `data/<dataset-name>/{train,validation,test}.jsonl` (`test`
  optional). Point `configs/data/*.yaml`'s `dataset_dir` at that folder —
  copy `configs/data/example.yaml` as a starting template.
- Alpaca-style records (`instruction`/`input`/`output`, no `messages`) are
  also accepted and converted automatically — see `angel_ai/data/converters.py`.
- A bundled 4-record example lives at `src/angel_ai/data/examples/tiny/`,
  used by `just dry-run` and the fast test suite.

## Backend capability matrix

| Backend    | Training | QLoRA (bitsandbytes) | Notes |
|------------|----------|-----------------------|-------|
| `directml` | Yes (iGPU/dGPU) | No | This machine's Radeon 840M target |
| `cuda`     | Yes | Yes | Local NVIDIA GPU, or cloud (Colab, see `docs/colab.md`) |
| `cpu`      | Yes (slow) | No | Always available; used for tests/dry-runs |
| NPU (Ryzen AI) | No (inference only) | n/a | Reached via `optimize.export target=npu`, not `backend=` |

Requesting an unsupported combination (e.g. `training=qlora backend=directml`)
raises `angel_ai.errors.UnsupportedConfigError` with the incompatible field
and a suggested fix, rather than failing deep inside a third-party library.

## Running on Google Colab / cloud GPUs

See [`docs/colab.md`](docs/colab.md). In short: `backend=cuda` is the same
config path a Colab GPU runtime needs — no separate branch, just
`uv sync --extra cuda` in the notebook and pointing checkpoints/MLflow at
Google Drive so they survive session resets.

## Development workflow

- `main` stays hardware-agnostic and config-driven; branches are for actual
  code/experiment divergence (a new training loop, a different data
  pipeline), not for switching hardware targets — that's a config/extras
  concern (`uv sync --extra ...`).
- Changes land via feature branches + pull requests, not direct pushes to
  `main`.
- `just check` (lint + config validation) and `just test` (fast unit tests)
  should pass before opening a PR; `just test-slow` before merging anything
  that touches training/eval/export internals.
- Versioning follows `pyproject.toml`'s `[project].version`; a merge that
  bumps it gets tagged `vX.Y.Z` on `main` (see [`CHANGELOG.md`](CHANGELOG.md)).
  Pre-1.0, expect breaking config changes between minor versions.

## Troubleshooting

- **`torch-directml` import fails / ImportError**: confirm you're on Python
  3.10–3.12 (`uv run python --version`) and ran `uv sync --extra directml`,
  not `--extra cpu`.
- **`UnsupportedConfigError` mentioning quantization**: QLoRA needs
  `backend=cuda`; DirectML/CPU only support plain LoRA.
- **Out-of-memory on the iGPU**: lower `training.per_device_train_batch_size`,
  raise `training.gradient_accumulation_steps` to compensate, and confirm
  `training.gradient_checkpointing: true`.
- **NPU export fails to load a provider**: the Vitis AI execution provider
  requires AMD's Ryzen AI SDK/driver to be installed separately — this repo
  only handles the ONNX export/quantization side. Fall back to
  `optimize.export target=igpu` (DirectML EP) or `target=cpu` if it isn't
  installed.

## License

MIT — see [`LICENSE`](LICENSE).
