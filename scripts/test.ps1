# Usage: scripts/test.ps1          -> fast unit tests only
#        scripts/test.ps1 -Slow    -> include marked slow end-to-end smoke tests
param([switch]$Slow)

if ($Slow) { uv run pytest } else { uv run pytest -m "not slow" }
