#!/usr/bin/env bash
# Usage: scripts/test.sh          -> fast unit tests only
#        scripts/test.sh --slow   -> include marked slow end-to-end smoke tests
set -euo pipefail
if [[ "${1:-}" == "--slow" ]]; then
    uv run pytest
else
    uv run pytest -m "not slow"
fi
