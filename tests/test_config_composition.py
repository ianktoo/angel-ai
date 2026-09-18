from pathlib import Path

from hydra import compose, initialize_config_dir

CONFIG_DIR = str((Path(__file__).parents[1] / "configs").resolve())


def test_default_training_config_composes():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config")
    assert cfg.backend.name == "auto"
    assert cfg.training.method == "lora"
    assert cfg.data.name == "tiny"


def test_backend_override_applies():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config", overrides=["backend=cpu"])
    assert cfg.backend.name == "cpu"


def test_qlora_training_override_applies():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config", overrides=["training=qlora"])
    assert cfg.training.method == "qlora"
    assert cfg.training.quantization == "qlora"


def test_export_config_composes():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="export")
    assert cfg.optimize.target == "npu"


def test_sweep_config_composes():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="sweep")
    assert cfg.optimize.n_trials == 10
    assert "learning_rate" in cfg.optimize.search_space
