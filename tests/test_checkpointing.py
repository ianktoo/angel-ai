from typing import ClassVar

import torch
from torch import nn

from angel_ai.checkpointing import (
    TrainerState,
    find_latest_checkpoint,
    load_checkpoint,
    resolve_resume_path,
    save_checkpoint,
)


class _TinyModel(nn.Module):
    """A stand-in for a PEFT model: exposes save_pretrained/load_adapter."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def save_pretrained(self, path, state_dict=None):
        from pathlib import Path

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(state_dict if state_dict is not None else self.state_dict(), path / "weights.pt")

    def load_adapter(self, path, adapter_name=None):
        from pathlib import Path

        self.load_state_dict(torch.load(Path(path) / "weights.pt", weights_only=True))


def test_save_and_resume_round_trip(tmp_path):
    model = _TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    state = TrainerState(step=5, epoch=0.5)

    ckpt_dir = save_checkpoint(tmp_path, model=model, optimizer=optimizer, state=state)
    assert ckpt_dir.exists()

    restored_model = _TinyModel()
    restored_optimizer = torch.optim.SGD(restored_model.parameters(), lr=0.1)
    restored_state = load_checkpoint(ckpt_dir, model=restored_model, optimizer=restored_optimizer)

    assert restored_state.step == 5
    assert restored_state.epoch == 0.5
    assert torch.allclose(model.linear.weight, restored_model.linear.weight)


def test_find_latest_checkpoint_picks_highest_step(tmp_path):
    model = _TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for step in (10, 30, 20):
        save_checkpoint(tmp_path, model=model, optimizer=optimizer, state=TrainerState(step=step))

    latest = find_latest_checkpoint(tmp_path)
    assert latest.name.endswith("30")


def test_find_latest_checkpoint_returns_none_when_empty(tmp_path):
    assert find_latest_checkpoint(tmp_path / "does-not-exist") is None


def test_prune_keeps_only_last_n(tmp_path):
    model = _TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for step in range(1, 6):
        save_checkpoint(
            tmp_path, model=model, optimizer=optimizer, state=TrainerState(step=step), keep_last_n=2
        )

    remaining = sorted(p.name for p in tmp_path.glob("checkpoint-step-*"))
    assert len(remaining) == 2
    assert remaining[-1] == "checkpoint-step-5"


def test_resolve_resume_path_none_when_not_set(tmp_path):
    assert resolve_resume_path(tmp_path, None) is None


def test_resolve_resume_path_latest(tmp_path):
    model = _TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    save_checkpoint(tmp_path, model=model, optimizer=optimizer, state=TrainerState(step=1))

    resolved = resolve_resume_path(tmp_path, "latest")
    assert resolved.name == "checkpoint-step-1"


class _FakePeftModel(_TinyModel):
    """Stand-in for a PEFT model: has `peft_config`, so save_checkpoint should
    take the get_peft_model_state_dict path instead of plain state_dict()."""

    peft_config: ClassVar[dict] = {"default": object()}


def test_save_checkpoint_uses_peft_state_dict_for_peft_models(tmp_path, mocker):
    model = _FakePeftModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    fake_state = {"lora.weight": model.linear.weight.detach().clone()}
    mocked = mocker.patch("peft.get_peft_model_state_dict", return_value=fake_state)

    save_checkpoint(tmp_path, model=model, optimizer=optimizer, state=TrainerState(step=1))

    mocked.assert_called_once_with(model)
