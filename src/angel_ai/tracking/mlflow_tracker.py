"""Thin MLflow wrapper used by training/sweep/eval stages.

Uses a local file-store backend (no server, no account) so run comparison
works out of the box: `mlflow ui --backend-store-uri <output_dir>/mlruns`.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow


@contextmanager
def run(tracking_dir: str | Path, experiment_name: str, run_name: str | None = None) -> Iterator[None]:
    """Local, file-based MLflow run -- no server, no account.

    MLflow's plain directory-based file store is in maintenance mode as of
    2.x, so this uses a local SQLite database instead
    (`sqlite:///<tracking_dir>/mlflow.db`), which is still entirely local and
    free, just not the plain-folder store. Browse with:
      mlflow ui --backend-store-uri sqlite:///<tracking_dir>/mlflow.db
    """
    tracking_dir = Path(tracking_dir)
    tracking_dir.mkdir(parents=True, exist_ok=True)
    db_path = (tracking_dir / "mlflow.db").resolve()
    mlflow.set_tracking_uri(f"sqlite:///{db_path.as_posix()}")
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        yield


def log_params(params: dict[str, Any]) -> None:
    # MLflow rejects nested/complex values; flatten to strings defensively.
    mlflow.log_params({k: str(v) for k, v in params.items()})


def log_metrics(metrics: dict[str, float], *, step: int | None = None) -> None:
    mlflow.log_metrics(metrics, step=step)


def log_artifact(path: str | Path) -> None:
    mlflow.log_artifact(str(path))
