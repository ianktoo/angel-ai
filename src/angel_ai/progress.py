"""Consistent rich-based progress/logging helpers shared across all stages."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

# On Windows, redirecting stdout to a file/pipe (rather than an interactive
# terminal) makes Python fall back to the system locale's codec (cp1252),
# which can't encode Rich's spinner's Unicode braille characters and crashes
# the whole process with UnicodeEncodeError. Force UTF-8 so piping output to
# a log file (a normal thing to do with a training run) doesn't crash.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except (ValueError, OSError):
            pass

console = Console()


def stage_progress(description: str, total: int) -> Progress:
    """A Progress bar configured consistently for every pipeline stage.

    Usage:
        with stage_progress("Training", total=num_steps) as progress:
            task = progress.add_task("step", total=num_steps)
            for step in range(num_steps):
                ...
                progress.update(task, advance=1, description=f"loss={loss:.4f}")
    """
    progress = Progress(
        SpinnerColumn(),
        TextColumn(f"[bold blue]{description}[/] [progress.description]{{task.description}}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )
    return progress


@contextmanager
def status(message: str) -> Iterator[None]:
    """Spinner for indeterminate-length steps (loading a model, exporting)."""
    with console.status(f"[bold blue]{message}[/]"):
        yield


def log(message: str, *, style: str = "") -> None:
    console.log(message, style=style)
