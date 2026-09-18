"""CPU backend — always available, used as the safe fallback and for tests."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from typing import Any

from angel_ai.backends.base import Backend, QuantizationKind


class CPUBackend(Backend):
    name = "cpu"

    def is_available(self) -> bool:
        return True

    def device(self) -> Any:
        import torch

        return torch.device("cpu")

    def prepare_model(self, model: Any, cfg: Any) -> Any:
        return model.to(self.device())

    def supports_quantization(self, kind: QuantizationKind) -> bool:
        return False

    def autocast_context(self) -> AbstractContextManager:
        return nullcontext()
