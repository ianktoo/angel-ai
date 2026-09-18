uv run ruff check src tests
if ($?) { uv run python scripts/check_configs.py }
