"""Resolves the repo's `configs/` directory as an absolute path.

Hydra treats a relative `config_path` in `@hydra.main` as a dotted Python
package path (requiring `__init__.py`), not a filesystem traversal -- since
`configs/` is a plain YAML directory outside the installed package, every
stage entrypoint needs an absolute path instead.
"""

from __future__ import annotations

from pathlib import Path

CONFIG_DIR = str((Path(__file__).parent.parent.parent / "configs").resolve())
