#!/usr/bin/env bash
# Usage: scripts/setup.sh [directml|cuda|cpu]
set -euo pipefail
BACKEND="${1:-cpu}"

uv python install 3.11
uv sync --extra "$BACKEND" --group dev
