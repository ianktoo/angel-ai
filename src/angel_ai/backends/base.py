"""The Backend interface every hardware target implements.

Pipeline stages (training/optimize/evaluation) depend only on this interface,
never on torch-directml/cuda specifics directly, so swapping hardware is a
config change (`backend: directml|cuda|cpu`) rather than a code change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from typing import Any, Literal

QuantizationKind = Literal["qlora", "int8", "gptq"]


class Backend(ABC):
    """Hardware-specific operations required by the pipeline stages."""

    name: str

    @abstractmethod
    def is_available(self) -> bool:
        """Whether this backend's runtime dependency is importable/usable here."""

    @abstractmethod
    def device(self) -> Any:
        """Return the torch device (or device-like object) to place tensors on."""

    @abstractmethod
    def prepare_model(self, model: Any, cfg: Any) -> Any:
        """Move/prepare a model for training or inference on this backend."""

    @abstractmethod
    def supports_quantization(self, kind: QuantizationKind) -> bool:
        """Whether this backend can run the given quantization scheme.

        Used by the training stage to fall back gracefully (e.g. QLoRA is
        CUDA-only via bitsandbytes; DirectML/CPU report False and the caller
        falls back to plain LoRA instead of crashing deep in bitsandbytes).
        """

    @abstractmethod
    def autocast_context(self) -> AbstractContextManager:
        """Context manager enabling mixed precision if this backend supports it."""

    def empty_cache(self) -> None:
        """Best-effort release of cached device memory. No-op by default."""
        return
