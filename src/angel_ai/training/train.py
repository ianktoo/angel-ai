"""The finetuning stage: `uv run python -m angel_ai.training [overrides...]`.

Wires together: backend resolution -> dataset loading/tokenizing -> LoRA model
setup -> a plain PyTorch training loop -> checkpointing -> MLflow tracking ->
rich progress bars. Kept as an explicit loop (not transformers.Trainer) so the
backend abstraction and checkpoint/resume logic stay simple to follow.
"""

from __future__ import annotations

import math
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from angel_ai import checkpointing, tracking
from angel_ai.backends.registry import get_backend
from angel_ai.data.loader import load_dataset
from angel_ai.hydra_utils import CONFIG_DIR
from angel_ai.progress import log, stage_progress
from angel_ai.training.prepare import apply_lora, load_base_model, load_tokenizer
from angel_ai.training.tokenize import collate, tokenize_dataset


def run_training(cfg: DictConfig) -> Path:
    torch.manual_seed(cfg.seed)

    backend = get_backend(cfg.backend.name)
    log(f"Resolved backend: {backend.name}")

    tokenizer = load_tokenizer(cfg.model)
    model = load_base_model(cfg.model, backend)
    model = apply_lora(model, cfg.training, backend)

    dataset = load_dataset(cfg.data)
    train_split = dataset["train"]
    max_train_samples = cfg.data.get("max_train_samples")
    if max_train_samples:
        train_split = train_split.select(range(min(len(train_split), max_train_samples)))
    train_dataset = tokenize_dataset(
        train_split,
        tokenizer,
        max_seq_length=cfg.data.max_seq_length,
        default_system_prompt=cfg.data.default_system_prompt,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.per_device_train_batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate(batch, tokenizer),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state = checkpointing.TrainerState()
    resume_path = checkpointing.resolve_resume_path(output_dir, cfg.training.resume_from_checkpoint)
    if resume_path is not None:
        state = checkpointing.load_checkpoint(resume_path, model=model, optimizer=optimizer)
        log(f"Resumed from checkpoint at step {state.step}: {resume_path}")

    steps_per_epoch = math.ceil(len(train_loader) / cfg.training.gradient_accumulation_steps)
    total_steps = cfg.training.max_steps or steps_per_epoch * cfg.training.num_epochs

    with tracking.run(output_dir / "mlruns", experiment_name="angel-ai-training"):
        tracking.log_params(OmegaConf.to_container(cfg, resolve=True))

        model.train()
        with stage_progress("Training", total=total_steps) as progress:
            task = progress.add_task("step 0", total=total_steps)
            optimizer.zero_grad()

            done = False
            for epoch in range(cfg.training.num_epochs):
                if done:
                    break
                for micro_step, batch in enumerate(train_loader):
                    batch = {k: v.to(backend.device()) for k, v in batch.items()}
                    with backend.autocast_context():
                        loss = model(**batch).loss / cfg.training.gradient_accumulation_steps
                    loss.backward()

                    if (micro_step + 1) % cfg.training.gradient_accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                        state.step += 1
                        state.epoch = epoch + micro_step / len(train_loader)

                        step_loss = loss.item() * cfg.training.gradient_accumulation_steps
                        if state.step % cfg.training.logging_steps == 0:
                            tracking.log_metrics({"train_loss": step_loss}, step=state.step)
                        progress.update(task, advance=1, description=f"step {state.step} loss={step_loss:.4f}")

                        if state.step % cfg.training.save_steps == 0:
                            checkpointing.save_checkpoint(
                                output_dir,
                                model=model,
                                optimizer=optimizer,
                                state=state,
                                keep_last_n=cfg.training.keep_last_n_checkpoints,
                            )

                        if cfg.training.max_steps and state.step >= cfg.training.max_steps:
                            done = True
                            break

        final_checkpoint = checkpointing.save_checkpoint(
            output_dir,
            model=model,
            optimizer=optimizer,
            state=state,
            keep_last_n=cfg.training.keep_last_n_checkpoints,
        )
        tracking.log_artifact(final_checkpoint)

    log(f"Training complete. Final checkpoint: {final_checkpoint}", style="bold green")
    backend.empty_cache()
    return final_checkpoint


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    run_training(cfg)


if __name__ == "__main__":
    main()
