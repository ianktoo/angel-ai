"""Hardware backends (cpu/cuda/directml) behind a common interface.

Use `angel_ai.backends.registry.get_backend` to resolve the active backend
from config rather than importing a concrete backend directly.
"""

from angel_ai.backends.base import Backend
from angel_ai.backends.registry import available_backends, get_backend

__all__ = ["Backend", "available_backends", "get_backend"]
