#!/usr/bin/env bash
# 1-step sanity pass through the full pipeline (data -> model -> backend ->
# checkpoint -> tracking) using a tiny model and the bundled tiny dataset.
set -euo pipefail
uv run python -m angel_ai.training model=tiny_test backend=cpu training.max_steps=1 training.save_steps=1 \
  training.logging_steps=1 training.gradient_accumulation_steps=1 "training.lora.target_modules=[c_attn]" "$@"
