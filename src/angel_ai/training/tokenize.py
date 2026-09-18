"""Turns canonical chat records into tokenized, label-masked training examples.

Labels are masked (-100) on everything except the final assistant turn, so
loss is only computed on the text the model should learn to produce.
"""

from __future__ import annotations

from typing import Any

IGNORE_INDEX = -100


def tokenize_example(record: dict, tokenizer, *, max_seq_length: int, default_system_prompt: str) -> dict[str, Any]:
    messages = list(record["messages"])
    if messages[0]["role"] != "system":
        messages = [{"role": "system", "content": default_system_prompt}, *messages]

    prompt_messages = messages[:-1]
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )

    full_ids = tokenizer(full_text, truncation=True, max_length=max_seq_length)["input_ids"]
    prompt_ids = tokenizer(prompt_text, truncation=True, max_length=max_seq_length)["input_ids"]

    labels = list(full_ids)
    prompt_len = min(len(prompt_ids), len(labels))
    for i in range(prompt_len):
        labels[i] = IGNORE_INDEX

    return {"input_ids": full_ids, "labels": labels, "attention_mask": [1] * len(full_ids)}


def tokenize_dataset(dataset, tokenizer, *, max_seq_length: int, default_system_prompt: str):
    return dataset.map(
        lambda record: tokenize_example(
            record,
            tokenizer,
            max_seq_length=max_seq_length,
            default_system_prompt=default_system_prompt,
        ),
        remove_columns=dataset.column_names,
    )


def collate(batch: list[dict], tokenizer):
    import torch

    max_len = max(len(ex["input_ids"]) for ex in batch)
    pad_id = tokenizer.pad_token_id

    input_ids, attention_mask, labels = [], [], []
    for ex in batch:
        pad_len = max_len - len(ex["input_ids"])
        input_ids.append(ex["input_ids"] + [pad_id] * pad_len)
        attention_mask.append(ex["attention_mask"] + [0] * pad_len)
        labels.append(ex["labels"] + [IGNORE_INDEX] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
