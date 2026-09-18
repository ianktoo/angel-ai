#!/usr/bin/env bash
# One-line bootstrap for a Colab (or similar cloud GPU) notebook cell.
# See docs/colab.md for the full walkthrough.
set -euo pipefail

pip install -q uv
uv python install 3.11
uv sync --extra cuda

python - <<'PY'
try:
    from google.colab import drive
    drive.mount("/content/drive")
except ImportError:
    print("Not running in Colab; skipping Drive mount.")
PY

echo "Bootstrap complete. Example run:"
echo "  uv run python -m angel_ai.training backend=cuda output_dir=/content/drive/MyDrive/angel-ai-runs/run-001"
