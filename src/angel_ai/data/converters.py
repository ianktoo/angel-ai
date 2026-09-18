"""Converts convenience dataset formats into the canonical chat schema.

Canonical schema: {"messages": [{"role": ..., "content": ...}, ...]}
"""

from __future__ import annotations

from typing import Any


def is_alpaca_style(record: dict[str, Any]) -> bool:
    return "messages" not in record and "instruction" in record


def alpaca_to_chat(record: dict[str, Any]) -> dict[str, Any]:
    """Convert an Alpaca-style {instruction, input, output} record to chat form."""
    instruction = record["instruction"].strip()
    extra_input = record.get("input", "").strip()
    user_content = f"{instruction}\n\n{extra_input}" if extra_input else instruction

    messages = [{"role": "user", "content": user_content}]
    if record.get("output"):
        messages.append({"role": "assistant", "content": record["output"].strip()})
    return {"messages": messages}


def to_canonical(record: dict[str, Any]) -> dict[str, Any]:
    """Normalize any supported input record into the canonical chat schema."""
    if is_alpaca_style(record):
        return alpaca_to_chat(record)
    return record
