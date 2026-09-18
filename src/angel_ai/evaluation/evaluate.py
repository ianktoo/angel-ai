"""Evaluation stage: `uv run python -m angel_ai.evaluation [overrides...]`.

Runs a forward-pass-only eval loop (loss/perplexity) against a checkpoint or
an exported model. This is intentionally backend-agnostic: eval never needs
gradients, so it runs the same way regardless of which backend produced the
checkpoint.
"""

from __future__ import annotations

import math
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from angel_ai import tracking
from angel_ai.backends.registry import get_backend
from angel_ai.checkpointing import find_latest_checkpoint
from angel_ai.data.loader import load_dataset
from angel_ai.hydra_utils import CONFIG_DIR
from angel_ai.progress import log, stage_progress
from angel_ai.training.prepare import apply_lora, load_base_model, load_tokenizer
from angel_ai.training.tokenize import collate, tokenize_dataset


def run_evaluation(cfg: DictConfig) -> dict[str, float]:
    backend = get_backend(cfg.backend.name)
    tokenizer = load_tokenizer(cfg.model)
    model = load_base_model(cfg.model, backend)
    model = apply_lora(model, cfg.training, backend)

    output_dir = Path(cfg.output_dir)
    checkpoint = find_latest_checkpoint(output_dir)
    if checkpoint is not None:
        model.load_adapter(checkpoint / "adapter", adapter_name="default")
        log(f"Loaded adapter from {checkpoint}")
    else:
        log("No checkpoint found; evaluating the base model.", style="yellow")

    dataset = load_dataset(cfg.data)
    split = "validation" if "validation" in dataset else "train"
    eval_split = dataset[split]
    max_eval_samples = cfg.eval.get("max_eval_samples")
    if max_eval_samples:
        eval_split = eval_split.select(range(min(len(eval_split), max_eval_samples)))
    eval_dataset = tokenize_dataset(
        eval_split,
        tokenizer,
        max_seq_length=cfg.eval.max_seq_length,
        default_system_prompt=cfg.data.default_system_prompt,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=cfg.eval.batch_size,
        collate_fn=lambda batch: collate(batch, tokenizer),
    )

    model.eval()
    total_loss, total_batches = 0.0, 0
    with tracking.run(output_dir / "mlruns", experiment_name="angel-ai-eval"):
        with stage_progress("Evaluating", total=len(eval_loader)) as progress:
            task = progress.add_task("batch", total=len(eval_loader))
            with torch.no_grad():
                for batch in eval_loader:
                    batch = {k: v.to(backend.device()) for k, v in batch.items()}
                    with backend.autocast_context():
                        loss = model(**batch).loss
                    total_loss += loss.item()
                    total_batches += 1
                    progress.update(task, advance=1)

        mean_loss = total_loss / max(total_batches, 1)
        metrics = {"eval_loss": mean_loss, "eval_perplexity": math.exp(mean_loss)}
        tracking.log_metrics(metrics)

    log(f"Eval results: {metrics}", style="bold green")
    return metrics


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    run_evaluation(cfg)


if __name__ == "__main__":
    main()
