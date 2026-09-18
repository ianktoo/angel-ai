#!/usr/bin/env bash
set -euo pipefail
uv run ruff check src tests
uv run python scripts/check_configs.py
