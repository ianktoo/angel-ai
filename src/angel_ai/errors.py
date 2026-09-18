"""Typed exceptions raised at config/environment boundaries.

These carry an actionable message so failures surface as a short, specific
error instead of a stack trace from deep inside a third-party library.
"""

from __future__ import annotations


class AngelAIError(Exception):
    """Base class for all angel-ai errors."""


class UnsupportedConfigError(AngelAIError):
    """A requested config combination isn't supported by the resolved backend.

    Example: ``training=qlora`` on a backend whose
    ``supports_quantization("qlora")`` is ``False``.
    """

    def __init__(self, message: str, *, field: str | None = None, suggestion: str | None = None):
        parts = [message]
        if field:
            parts.append(f"(field: {field})")
        if suggestion:
            parts.append(f"Suggestion: {suggestion}")
        super().__init__(" ".join(parts))
        self.field = field
        self.suggestion = suggestion


class BackendUnavailableError(AngelAIError):
    """The requested backend's runtime dependency isn't importable/usable."""


class DatasetFormatError(AngelAIError):
    """A dataset file doesn't match the expected schema.

    Carries the offending line number (1-indexed) so the user can jump
    straight to the bad record instead of guessing.
    """

    def __init__(self, message: str, *, path: str | None = None, line_number: int | None = None):
        parts = [message]
        if path:
            parts.append(f"(file: {path})")
        if line_number is not None:
            parts.append(f"(line: {line_number})")
        super().__init__(" ".join(parts))
        self.path = path
        self.line_number = line_number


class CheckpointError(AngelAIError):
    """Saving or resuming from a checkpoint failed."""
