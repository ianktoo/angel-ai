"""`angel-ai <stage> [hydra overrides...]` -- thin dispatch to each pipeline stage.

Equivalent to calling `python -m angel_ai.<stage>` directly; this just saves
typing the module path. All Hydra config overrides pass through unchanged.
"""

from __future__ import annotations

import sys

_STAGE_ENTRYPOINTS = {
    "train": "angel_ai.training.train",
    "evaluate": "angel_ai.evaluation.evaluate",
    "export": "angel_ai.optimize.export",
    "sweep": "angel_ai.optimize.sweep",
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in _STAGE_ENTRYPOINTS:
        print(f"Usage: angel-ai <{'|'.join(_STAGE_ENTRYPOINTS)}> [config overrides...]")
        raise SystemExit(1)

    stage, overrides = sys.argv[1], sys.argv[2:]
    sys.argv = [stage, *overrides]

    import importlib

    module = importlib.import_module(_STAGE_ENTRYPOINTS[stage])
    module.main()


if __name__ == "__main__":
    main()
