"""Resolves the active Backend from config.

Every pipeline stage should call `get_backend(cfg)` instead of importing a
backend module directly, so hardware selection stays a config concern.
"""

from __future__ import annotations

from angel_ai.backends.base import Backend
from angel_ai.backends.cpu import CPUBackend
from angel_ai.backends.cuda import CUDABackend
from angel_ai.backends.directml import DirectMLBackend
from angel_ai.errors import BackendUnavailableError, UnsupportedConfigError

_BACKENDS: dict[str, type[Backend]] = {
    "cpu": CPUBackend,
    "cuda": CUDABackend,
    "directml": DirectMLBackend,
}


def available_backends() -> list[str]:
    """Names of backends whose runtime dependency is actually usable here."""
    return [name for name, cls in _BACKENDS.items() if cls().is_available()]


def get_backend(name: str, *, require_available: bool = True) -> Backend:
    """Instantiate the named backend.

    Args:
        name: one of "cpu", "cuda", "directml", or "auto" to pick the first
            available backend in priority order (directml, cuda, cpu).
        require_available: if True, raises when the backend's dependency
            isn't importable/usable instead of returning a dead instance.
    """
    if name == "auto":
        for candidate in ("directml", "cuda", "cpu"):
            backend = _BACKENDS[candidate]()
            if backend.is_available():
                return backend
        # CPU backend is always available, so this is unreachable in practice.
        return CPUBackend()

    backend_cls = _BACKENDS.get(name)
    if backend_cls is None:
        raise UnsupportedConfigError(
            f"Unknown backend '{name}'.",
            field="backend",
            suggestion=f"Choose one of: {', '.join(sorted(_BACKENDS))}, or 'auto'.",
        )

    backend = backend_cls()
    if require_available and not backend.is_available():
        raise BackendUnavailableError(
            f"Backend '{name}' is not available in this environment. "
            f"Available backends here: {available_backends() or 'none (only cpu should ever be empty-safe)'}."
        )
    return backend
