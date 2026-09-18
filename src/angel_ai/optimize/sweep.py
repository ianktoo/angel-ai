"""Hyperparameter sweep stage: `uv run python -m angel_ai.optimize.sweep`.

Runs `n_trials` short training+eval runs with Optuna-suggested hyperparameters
(learning rate, LoRA rank, ...) and reports the best trial. Each trial gets
its own output_dir so per-trial checkpoints survive an interrupted sweep.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import optuna
from omegaconf import DictConfig, OmegaConf

from angel_ai.evaluation.evaluate import run_evaluation
from angel_ai.hydra_utils import CONFIG_DIR
from angel_ai.progress import log
from angel_ai.training.train import run_training


def _suggest(trial: optuna.Trial, name: str, spec: DictConfig):
    if spec.type == "loguniform":
        return trial.suggest_float(name, spec.low, spec.high, log=True)
    if spec.type == "uniform":
        return trial.suggest_float(name, spec.low, spec.high)
    if spec.type == "categorical":
        return trial.suggest_categorical(name, list(spec.choices))
    raise ValueError(f"Unknown search space type '{spec.type}' for '{name}'")


def _set_nested(cfg: DictConfig, dotted_key: str, value) -> None:
    OmegaConf.update(cfg, dotted_key, value, merge=True)


def run_sweep(cfg: DictConfig) -> optuna.Study:
    base_cfg = cfg
    sweep_cfg = cfg.optimize

    def objective(trial: optuna.Trial) -> float:
        trial_cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
        for name, spec in sweep_cfg.search_space.items():
            _set_nested(trial_cfg, f"training.{name}", _suggest(trial, name, spec))

        trial_output_dir = Path(sweep_cfg.sweep_output_dir) / f"trial-{trial.number}"
        trial_cfg.output_dir = str(trial_output_dir)
        trial_cfg.training.max_steps = trial_cfg.training.max_steps or 20

        run_training(trial_cfg)
        metrics = run_evaluation(trial_cfg)
        return metrics["eval_loss"]

    study = optuna.create_study(direction=sweep_cfg.direction)
    study.optimize(objective, n_trials=sweep_cfg.n_trials)

    log(f"Best trial: {study.best_trial.number} params={study.best_trial.params} "
        f"eval_loss={study.best_value:.4f}", style="bold green")
    return study


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="sweep")
def main(cfg: DictConfig) -> None:
    run_sweep(cfg)


if __name__ == "__main__":
    main()
