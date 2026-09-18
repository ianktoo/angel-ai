"""Checkpoint save/load/resume helpers shared by training and sweep stages.

A checkpoint directory contains:
  adapter/            - PEFT adapter weights (or full model, if not using LoRA)
  optimizer.pt         - optimizer state dict
  rng_state.pt          - torch/python/numpy RNG state, for reproducible resume
  trainer_state.json   - step, epoch, best_metric, etc.
"""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from angel_ai.errors import CheckpointError

CHECKPOINT_PREFIX = "checkpoint-step-"


@dataclass
class TrainerState:
    step: int = 0
    epoch: float = 0.0
    best_metric: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainerState:
        return cls(**data)


def _checkpoint_dir(output_dir: Path, step: int) -> Path:
    return output_dir / f"{CHECKPOINT_PREFIX}{step}"


def save_checkpoint(
    output_dir: Path,
    *,
    model: Any,
    optimizer: Any,
    state: TrainerState,
    keep_last_n: int = 3,
) -> Path:
    """Save model/optimizer/RNG/trainer state for later resume.

    Older checkpoints beyond `keep_last_n` are pruned to bound disk usage,
    since this machine has limited local storage relative to model sizes.
    """
    import torch

    output_dir = Path(output_dir)
    ckpt_dir = _checkpoint_dir(output_dir, state.step)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    try:
        model.save_pretrained(ckpt_dir / "adapter")
        torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
        torch.save(
            {
                "torch": torch.get_rng_state(),
                "python": random.getstate(),
            },
            ckpt_dir / "rng_state.pt",
        )
        (ckpt_dir / "trainer_state.json").write_text(json.dumps(state.to_dict(), indent=2))
    except OSError as exc:
        raise CheckpointError(f"Failed writing checkpoint to {ckpt_dir}: {exc}") from exc

    _prune_old_checkpoints(output_dir, keep_last_n=keep_last_n)
    return ckpt_dir


def _prune_old_checkpoints(output_dir: Path, *, keep_last_n: int) -> None:
    checkpoints = sorted(
        output_dir.glob(f"{CHECKPOINT_PREFIX}*"),
        key=lambda p: int(p.name.removeprefix(CHECKPOINT_PREFIX)),
    )
    for stale in checkpoints[:-keep_last_n] if keep_last_n > 0 else []:
        shutil.rmtree(stale, ignore_errors=True)


def find_latest_checkpoint(output_dir: Path) -> Path | None:
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return None
    checkpoints = sorted(
        output_dir.glob(f"{CHECKPOINT_PREFIX}*"),
        key=lambda p: int(p.name.removeprefix(CHECKPOINT_PREFIX)),
    )
    return checkpoints[-1] if checkpoints else None


def load_checkpoint(
    checkpoint_dir: Path,
    *,
    model: Any,
    optimizer: Any | None = None,
) -> TrainerState:
    """Resume model/optimizer/RNG/trainer state from a checkpoint directory."""
    import torch

    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        raise CheckpointError(f"Checkpoint directory does not exist: {checkpoint_dir}")

    adapter_dir = checkpoint_dir / "adapter"
    if hasattr(model, "load_adapter"):
        model.load_adapter(adapter_dir, adapter_name="default")
    else:
        from safetensors.torch import load_file

        state_dict = load_file(adapter_dir / "model.safetensors")
        model.load_state_dict(state_dict, strict=False)

    if optimizer is not None:
        optimizer_path = checkpoint_dir / "optimizer.pt"
        if optimizer_path.exists():
            optimizer.load_state_dict(
                torch.load(optimizer_path, map_location="cpu", weights_only=True)
            )

    rng_path = checkpoint_dir / "rng_state.pt"
    if rng_path.exists():
        rng = torch.load(rng_path, map_location="cpu", weights_only=True)
        torch.set_rng_state(rng["torch"])
        random.setstate(rng["python"])

    state_path = checkpoint_dir / "trainer_state.json"
    if not state_path.exists():
        raise CheckpointError(f"Missing trainer_state.json in {checkpoint_dir}")
    return TrainerState.from_dict(json.loads(state_path.read_text()))


def resolve_resume_path(output_dir: Path, resume_from_checkpoint: str | None) -> Path | None:
    """Interpret the `resume_from_checkpoint: latest|<path>|null` config field."""
    if not resume_from_checkpoint:
        return None
    if resume_from_checkpoint == "latest":
        return find_latest_checkpoint(Path(output_dir))
    path = Path(resume_from_checkpoint)
    if not path.exists():
        raise CheckpointError(f"resume_from_checkpoint path does not exist: {path}")
    return path
