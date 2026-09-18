# angel-ai task runner. Works identically in bash and PowerShell via `just`.
# No `just`? Every recipe here has a plain-script equivalent under scripts/
# (both .sh and .ps1) that runs the same underlying `uv run` command.

# "auto" is a runtime backend.name value, not a uv extras group -- pick the
# real extras group for this machine (directml/cuda/cpu) for `just setup`.
backend := "cpu"

setup:
    uv python install 3.11
    uv sync --extra {{backend}} --group dev

check:
    uv run ruff check src tests
    uv run python scripts/check_configs.py

dry-run:
    uv run python -m angel_ai.training model=tiny_test backend=cpu training.max_steps=1 training.save_steps=1 training.logging_steps=1 training.gradient_accumulation_steps=1 "training.lora.target_modules=[c_attn]"

test:
    uv run pytest -m "not slow"

test-slow:
    uv run pytest

train *ARGS:
    uv run python -m angel_ai.training {{ARGS}}

infer *ARGS:
    uv run python -m angel_ai.evaluation {{ARGS}}

export *ARGS:
    uv run python -m angel_ai.optimize.export {{ARGS}}

sweep *ARGS:
    uv run python -m angel_ai.optimize.sweep {{ARGS}}
