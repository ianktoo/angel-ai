"""Loads a JSONL chat/Alpaca dataset into a Hugging Face `datasets.Dataset`.

Expected on-disk layout (see README for the full spec):

    data/<dataset-name>/train.jsonl
    data/<dataset-name>/validation.jsonl
    data/<dataset-name>/test.jsonl   (optional)
"""

from __future__ import annotations

import json
from pathlib import Path

from angel_ai.data.converters import to_canonical
from angel_ai.data.validation import validate_file
from angel_ai.errors import DatasetFormatError


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(to_canonical(json.loads(line)))
    return records


def load_split(dataset_dir: str | Path, split: str, *, validate: bool = True):
    """Load one split (train/validation/test) as a `datasets.Dataset`.

    Raises DatasetFormatError (with file/line context) if `validate=True` and
    the file doesn't conform to the canonical chat schema after conversion.
    """
    from datasets import Dataset

    dataset_dir = Path(dataset_dir)
    split_path = dataset_dir / f"{split}.jsonl"
    if not split_path.exists():
        raise DatasetFormatError(f"Split file not found: {split_path}", path=str(split_path))

    records = _read_jsonl(split_path)
    dataset = Dataset.from_list(records)

    if validate:
        # Validate the canonicalized records, not the raw file, so Alpaca-style
        # input still gets checked against the schema it converts to.
        for line_number, record in enumerate(records, start=1):
            from angel_ai.data.validation import validate_record

            validate_record(record, path=str(split_path), line_number=line_number)

    return dataset


def load_dataset_dict(dataset_dir: str | Path, *, validate: bool = True):
    """Load train/validation (and test, if present) into a `DatasetDict`."""
    from datasets import DatasetDict

    dataset_dir = Path(dataset_dir)
    splits = {}
    for split in ("train", "validation", "test"):
        if (dataset_dir / f"{split}.jsonl").exists():
            splits[split] = load_split(dataset_dir, split, validate=validate)

    if "train" not in splits:
        raise DatasetFormatError(
            f"No train.jsonl found under {dataset_dir}", path=str(dataset_dir)
        )
    return DatasetDict(splits)


def load_dataset(data_cfg, *, validate: bool = True):
    """Dispatches to a local JSONL directory or the Hugging Face Hub based on
    `data_cfg.source` ("local", the default, or "huggingface").

    This is what training/evaluation should call -- it hides the local vs.
    hub distinction behind one config field (`data.source`).
    """
    from angel_ai.errors import UnsupportedConfigError

    source = data_cfg.get("source", "local")
    if source == "local":
        return load_dataset_dict(data_cfg.dataset_dir, validate=validate)
    if source == "huggingface":
        from angel_ai.data.hf_source import load_from_hub

        hf_cfg = data_cfg.hf
        return load_from_hub(
            hf_cfg.repo_id,
            config_name=hf_cfg.get("config_name"),
            split_mapping=hf_cfg.get("split_mapping") or {},
            prompt_column=hf_cfg.get("prompt_column"),
            response_column=hf_cfg.get("response_column"),
            validate=validate,
        )
    raise UnsupportedConfigError(
        f"Unknown data.source '{source}'.",
        field="data.source",
        suggestion="Use 'local' (a JSONL directory) or 'huggingface' (a Hub dataset repo).",
    )


__all__ = ["load_dataset", "load_dataset_dict", "load_split", "validate_file"]
