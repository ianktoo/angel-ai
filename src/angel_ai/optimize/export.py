"""Export/quantize a trained checkpoint for fast local inference.

`target` selects the ONNX Runtime execution provider the exported model is
meant to run under:
  - "npu"  -> Vitis AI EP (this machine's Ryzen AI NPU; inference-only, no
              training support exists for it)
  - "igpu" -> DirectML EP
  - "cpu"  -> CPU EP

This is a research/experimentation pipeline: NPU export via Vitis AI is
newer, less standardized tooling than CUDA/CPU export paths, so treat the
"npu" target as the one most likely to need environment-specific tweaking
(driver/SDK versions) rather than a guaranteed one-command path.
"""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from angel_ai.backends.registry import get_backend
from angel_ai.checkpointing import find_latest_checkpoint
from angel_ai.errors import UnsupportedConfigError
from angel_ai.hydra_utils import CONFIG_DIR
from angel_ai.progress import log, status
from angel_ai.training.prepare import apply_lora, load_base_model, load_tokenizer

_EXECUTION_PROVIDERS = {
    "npu": "VitisAIExecutionProvider",
    "igpu": "DmlExecutionProvider",
    "cpu": "CPUExecutionProvider",
}


def run_export(cfg: DictConfig) -> Path:
    target = cfg.optimize.target
    if target not in _EXECUTION_PROVIDERS:
        raise UnsupportedConfigError(
            f"Unknown export target '{target}'.",
            field="optimize.target",
            suggestion=f"Choose one of: {', '.join(_EXECUTION_PROVIDERS)}.",
        )

    checkpoint_path = Path(cfg.optimize.checkpoint) if cfg.optimize.checkpoint else find_latest_checkpoint(
        Path(cfg.output_dir)
    )
    if checkpoint_path is None:
        raise UnsupportedConfigError(
            "No checkpoint found to export.",
            field="optimize.checkpoint",
            suggestion="Run training first, or pass optimize.checkpoint=<path to a checkpoint's adapter dir>.",
        )

    backend = get_backend(cfg.backend.name)
    tokenizer = load_tokenizer(cfg.model)
    # "eager" attention: the legacy TorchScript-based ONNX exporter can't
    # trace newer transformers' default SDPA/masking code path (fails with
    # `UnsupportedOperatorError: aten::__ior_`) -- eager attention uses a
    # simpler, ONNX-export-friendly implementation. Training/eval don't need
    # this override; only tracing for export does.
    model = load_base_model(cfg.model, backend, attn_implementation="eager")
    model = apply_lora(model, cfg.training, backend)
    model.load_adapter(checkpoint_path / "adapter", adapter_name="default")
    merged_model = model.merge_and_unload()

    export_dir = Path(cfg.optimize.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = export_dir / "model.onnx"

    with status(f"Exporting to ONNX for execution provider {_EXECUTION_PROVIDERS[target]}"):
        _export_to_onnx(merged_model, tokenizer, onnx_path, quantization=cfg.optimize.quantization)

    tokenizer.save_pretrained(export_dir)
    log(f"Exported model for target='{target}' -> {export_dir}", style="bold green")
    return export_dir


def _wrap_for_tracing(model):
    """Wraps a causal LM so tracing only sees a plain tensor output.

    Modern `transformers` forward passes return a `DynamicCache` object for
    `past_key_values` even when unused, and the legacy `torch.onnx.export`
    tracer can't flatten a non-tuple/list/tensor object -- it fails with
    "Only tuples, lists and Variables are supported as JIT inputs/outputs."
    Forcing `use_cache=False` and returning only `.logits` sidesteps this.
    """
    from torch import nn

    class _LogitsOnly(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, attention_mask):
            return self.inner(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False
            ).logits

    return _LogitsOnly(model)


def _export_to_onnx(model, tokenizer, onnx_path: Path, *, quantization: str | None) -> None:
    import torch

    model.eval()
    wrapped = _wrap_for_tracing(model)
    dummy_input = tokenizer("angel-ai export dry run", return_tensors="pt")

    torch.onnx.export(
        wrapped,
        (dummy_input["input_ids"], dummy_input["attention_mask"]),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch", 1: "sequence"},
        },
        opset_version=17,
    )

    if quantization == "int8":
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantized_path = onnx_path.with_name("model.int8.onnx")
        quantize_dynamic(str(onnx_path), str(quantized_path), weight_type=QuantType.QInt8)
        onnx_path.unlink()
        quantized_path.rename(onnx_path)


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="export")
def main(cfg: DictConfig) -> None:
    run_export(cfg)


if __name__ == "__main__":
    main()
