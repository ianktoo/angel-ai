"""Dry-parses every Hydra entrypoint config to catch composition errors fast,
without loading any model. Used by `just check` / scripts/check.{sh,ps1}.
"""

from pathlib import Path

from hydra import compose, initialize_config_dir

CONFIG_DIR = str((Path(__file__).parents[1] / "configs").resolve())
ENTRYPOINTS = ["config", "export", "sweep"]

if __name__ == "__main__":
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        for name in ENTRYPOINTS:
            compose(config_name=name)
            print(f"{name}.yaml composes OK")
