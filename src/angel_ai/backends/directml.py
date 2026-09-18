"""DirectML backend — targets an integrated/discrete GPU via torch-directml.

This is the training backend for this machine's Radeon 840M iGPU.
QLoRA (bitsandbytes) is CUDA-only, so `supports_quantization` reports False
here and the training stage falls back to plain LoRA.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from typing import Any

from angel_ai.backends.base import Backend, QuantizationKind


class DirectMLBackend(Backend):
    name = "directml"

    def is_available(self) -> bool:
        try:
            import torch_directml  # noqa: F401
        except ImportError:
            return False
        return True

    def _torch_directml(self):
        try:
            import torch_directml
        except ImportError as exc:
            from angel_ai.errors import BackendUnavailableError

            raise BackendUnavailableError(
                "torch-directml is not installed. Install it with "
                "`uv sync --extra directml` (requires Python 3.10-3.12)."
            ) from exc
        return torch_directml

    def device(self) -> Any:
        return self._torch_directml().device()

    def prepare_model(self, model: Any, cfg: Any) -> Any:
        return model.to(self.device())

    def supports_quantization(self, kind: QuantizationKind) -> bool:
        # bitsandbytes (QLoRA) and GPTQ kernels are CUDA-only; DirectML has no
        # equivalent quantized-training kernel today.
        return False

    def autocast_context(self) -> AbstractContextManager:
        # torch-directml does not yet support torch.autocast; run in full/half
        # precision as configured by the model dtype instead.
        return nullcontext()
