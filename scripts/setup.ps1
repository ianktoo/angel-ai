# Usage: scripts/setup.ps1 [directml|cuda|cpu]
param([string]$Backend = "cpu")

uv python install 3.11
if ($?) { uv sync --extra $Backend --group dev }
