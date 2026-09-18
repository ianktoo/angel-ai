"""End-to-end smoke test: train a 1-step LoRA adapter on the tiny bundled
dataset with a tiny public model, then evaluate the resulting checkpoint.

Marked slow because it downloads a (tiny) model from the Hugging Face Hub
and exercises the real training/eval loops -- skipped by default
(`pytest -m "not slow"`), included by `just test-slow`.
"""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

CONFIG_DIR = str((Path(__file__).parents[1] / "configs").resolve())

DRY_RUN_OVERRIDES = [
    "model=tiny_test",
    "backend=cpu",
    "training.max_steps=1",
    "training.save_steps=1",
    "training.logging_steps=1",
    "training.gradient_accumulation_steps=1",
    "training.lora.target_modules=[c_attn]",
]


@pytest.mark.slow
def test_train_then_evaluate_round_trip(tmp_path):
    from angel_ai.evaluation.evaluate import run_evaluation
    from angel_ai.training.train import run_training

    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config", overrides=[*DRY_RUN_OVERRIDES, f"output_dir={tmp_path}"])

    checkpoint = run_training(cfg)
    assert checkpoint.exists()
    assert (checkpoint / "adapter").exists()
    assert (checkpoint / "trainer_state.json").exists()

    metrics = run_evaluation(cfg)
    assert "eval_loss" in metrics
    assert metrics["eval_loss"] > 0
