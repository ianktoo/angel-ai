#!/usr/bin/env bash
set -euo pipefail
uv run python -m angel_ai.evaluation "$@"
