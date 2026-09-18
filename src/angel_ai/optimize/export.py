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
    model = load_base_model(cfg.model, backend)
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


def _export_to_onnx(model, tokenizer, onnx_path: Path, *, quantization: str | None) -> None:
    import torch

    model.eval()
    dummy_input = tokenizer("angel-ai export dry run", return_tensors="pt")

    torch.onnx.export(
        model,
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
