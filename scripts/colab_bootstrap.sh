#!/usr/bin/env bash
# One-line dependency bootstrap for a Colab (or similar cloud GPU) notebook
# cell. See docs/colab.md for the full walkthrough.
#
# NOTE: this deliberately does NOT mount Google Drive. `google.colab.drive
# .mount()` talks to the Colab frontend over the notebook's own IPython
# kernel (via `get_ipython()`); called from a subprocess spawned by
# `!bash colab_bootstrap.sh`, there is no such kernel and it fails with
# `AttributeError: 'NoneType' object has no attribute 'kernel'` (confirmed
# by running it). Mount Drive in its own notebook cell instead -- see
# notebooks/colab_train.ipynb.
set -euo pipefail

pip install -q uv
uv python install 3.11
uv sync --extra cuda

echo "Bootstrap complete. Mount Drive in a notebook cell (see notebooks/colab_train.ipynb), then example run:"
echo "  uv run python -m angel_ai.training backend=cuda output_dir=/content/drive/MyDrive/angel-ai-runs/run-001"
