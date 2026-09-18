"""CUDA backend — used on cloud/Colab GPUs or a local NVIDIA card.

Supports QLoRA via bitsandbytes, which is CUDA-only.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any

from angel_ai.backends.base import Backend, QuantizationKind


class CUDABackend(Backend):
    name = "cuda"

    def is_available(self) -> bool:
        try:
            import torch
        except ImportError:
            return False
        return torch.cuda.is_available()

    def device(self) -> Any:
        import torch

        return torch.device("cuda")

    def prepare_model(self, model: Any, cfg: Any) -> Any:
        return model.to(self.device())

    def supports_quantization(self, kind: QuantizationKind) -> bool:
        if kind == "qlora":
            try:
                import bitsandbytes  # noqa: F401
            except ImportError:
                return False
            return True
        return kind in ("int8", "gptq")

    def autocast_context(self) -> AbstractContextManager:
        import torch

        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    def empty_cache(self) -> None:
        import torch

        torch.cuda.empty_cache()
