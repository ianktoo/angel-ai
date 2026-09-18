"""Validates dataset JSONL records against the expected chat schema.

See README/docs for the full dataset format spec. This runs before training
starts so a malformed dataset fails fast with a line number, not a cryptic
error deep inside the tokenizer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from angel_ai.errors import DatasetFormatError

VALID_ROLES = {"system", "user", "assistant"}


def validate_record(record: dict[str, Any], *, path: str, line_number: int) -> None:
    if "messages" not in record:
        raise DatasetFormatError(
            "Record is missing required 'messages' field.", path=path, line_number=line_number
        )
    messages = record["messages"]
    if not isinstance(messages, list) or not messages:
        raise DatasetFormatError(
            "'messages' must be a non-empty list.", path=path, line_number=line_number
        )
    for message in messages:
        if not isinstance(message, dict) or "role" not in message or "content" not in message:
            raise DatasetFormatError(
                "Each message needs 'role' and 'content' fields.",
                path=path,
                line_number=line_number,
            )
        if message["role"] not in VALID_ROLES:
            raise DatasetFormatError(
                f"Unknown role '{message['role']}'. Expected one of {sorted(VALID_ROLES)}.",
                path=path,
                line_number=line_number,
            )
        if not str(message["content"]).strip():
            raise DatasetFormatError(
                f"Message with role '{message['role']}' has empty content.",
                path=path,
                line_number=line_number,
            )
    non_system_roles = [m["role"] for m in messages if m["role"] != "system"]
    if not non_system_roles or non_system_roles[-1] != "assistant":
        raise DatasetFormatError(
            "The last non-system message must have role 'assistant' "
            "(the target the model is trained to produce).",
            path=path,
            line_number=line_number,
        )


def validate_file(path: str | Path) -> int:
    """Validate every record in a JSONL file. Returns the record count."""
    path = Path(path)
    if not path.exists():
        raise DatasetFormatError(f"Dataset file not found: {path}", path=str(path))

    count = 0
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetFormatError(
                    f"Invalid JSON: {exc}", path=str(path), line_number=line_number
                ) from exc
            validate_record(record, path=str(path), line_number=line_number)
            count += 1

    if count == 0:
        raise DatasetFormatError(f"Dataset file has no records: {path}", path=str(path))
    return count
