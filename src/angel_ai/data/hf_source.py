"""Ingests a dataset directly from the Hugging Face Hub, converting it into
the canonical chat schema instead of requiring a pre-converted local JSONL
file.

Handles three input shapes on the hub, tried in this order per row:
  1. Already has a `messages` column (chat-formatted) -> used as-is.
  2. Alpaca-style `instruction`/`input`/`output` columns -> converted via the
     same converter local Alpaca-format files use.
  3. A generic two-column mapping (`prompt_column`/`response_column` from
     config) -> converted into a single user/assistant turn.
"""

from __future__ import annotations

from typing import Any

from angel_ai.data.converters import alpaca_to_chat, is_alpaca_style
from angel_ai.data.validation import validate_record
from angel_ai.errors import DatasetFormatError


def _row_to_canonical(
    row: dict[str, Any], *, prompt_column: str | None, response_column: str | None
) -> dict[str, Any]:
    if row.get("messages"):
        return {"messages": row["messages"]}
    if is_alpaca_style(row):
        return alpaca_to_chat(row)
    if prompt_column and response_column:
        if prompt_column not in row or response_column not in row:
            raise DatasetFormatError(
                f"Configured prompt_column={prompt_column!r}/"
                f"response_column={response_column!r} not found in this "
                f"dataset's columns: {sorted(row.keys())}"
            )
        return {
            "messages": [
                {"role": "user", "content": str(row[prompt_column]).strip()},
                {"role": "assistant", "content": str(row[response_column]).strip()},
            ]
        }
    raise DatasetFormatError(
        "Could not infer chat format from this dataset's columns "
        f"({sorted(row.keys())}). Set data.hf.prompt_column and "
        "data.hf.response_column in config to map two columns into a "
        "user/assistant turn."
    )


def load_from_hub(
    repo_id: str,
    *,
    config_name: str | None = None,
    split_mapping: dict[str, str] | None = None,
    prompt_column: str | None = None,
    response_column: str | None = None,
    validate: bool = True,
):
    """Load a Hugging Face Hub dataset and convert it to the canonical schema.

    `split_mapping` maps our split names (train/validation/test) to the
    dataset's actual split names on the hub, e.g. `{"validation": "test"}`
    for a dataset that only has train/test splits. Splits absent both on the
    hub and in `split_mapping` are silently skipped (only `train` is
    required).
    """
    from datasets import Dataset, DatasetDict, load_dataset

    split_mapping = split_mapping or {}
    raw = load_dataset(repo_id, config_name)
    raw_splits = raw if isinstance(raw, DatasetDict) else {"train": raw}

    splits = {}
    for our_split in ("train", "validation", "test"):
        hub_split = split_mapping.get(our_split, our_split)
        if hub_split not in raw_splits:
            continue

        converted = [
            _row_to_canonical(dict(row), prompt_column=prompt_column, response_column=response_column)
            for row in raw_splits[hub_split]
        ]
        if validate:
            for line_number, record in enumerate(converted, start=1):
                validate_record(record, path=f"{repo_id}#{hub_split}", line_number=line_number)
        splits[our_split] = Dataset.from_list(converted)

    if "train" not in splits:
        raise DatasetFormatError(
            f"No train split found for '{repo_id}' "
            f"(hub splits: {sorted(raw_splits.keys())}, split_mapping: {split_mapping})."
        )
    return DatasetDict(splits)
