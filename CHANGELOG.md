# Changelog

Versions follow `pyproject.toml`'s `[project].version`, tagged on `main` as
`vX.Y.Z` once a PR merges. Pre-1.0, minor bumps may include breaking config
changes -- this is a research pipeline, not a stable API yet.

## [0.1.0] - 2026-09-17

Initial pipeline scaffold.

- Backend abstraction (`directml`/`cuda`/`cpu`) behind a common `Backend`
  interface, resolved from Hydra config so hardware is a config choice.
- LoRA/QLoRA finetuning loop with checkpoint save/resume, `rich` progress
  bars, and typed config/environment error handling.
- Optuna hyperparameter sweeps and ONNX export/quantization targeting
  NPU (Vitis AI EP), iGPU (DirectML EP), or CPU execution providers.
- Local, no-account MLflow experiment tracking (SQLite-backed).
- Canonical chat-style JSONL dataset format, Alpaca-format converter, and
  early schema validation with actionable errors.
- `just`-based task runner with bash/PowerShell script equivalents.
- Test suite (fast unit + config-composition + a slow-marked real
  train-then-evaluate smoke test), verified end-to-end on the CPU backend.
- DirectML (iGPU) and NPU export paths are implemented but not yet verified
  against real hardware -- see README's research framing.
