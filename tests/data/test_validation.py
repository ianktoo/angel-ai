import json

import pytest

from angel_ai.data.converters import alpaca_to_chat, is_alpaca_style, to_canonical
from angel_ai.data.validation import validate_file, validate_record
from angel_ai.errors import DatasetFormatError


def test_valid_record_passes():
    record = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}
    validate_record(record, path="test", line_number=1)


def test_missing_messages_field_raises():
    with pytest.raises(DatasetFormatError):
        validate_record({}, path="test", line_number=1)


def test_empty_messages_raises():
    with pytest.raises(DatasetFormatError):
        validate_record({"messages": []}, path="test", line_number=1)


def test_unknown_role_raises():
    record = {"messages": [{"role": "bot", "content": "hi"}]}
    with pytest.raises(DatasetFormatError):
        validate_record(record, path="test", line_number=1)


def test_empty_content_raises():
    record = {"messages": [{"role": "user", "content": "   "}]}
    with pytest.raises(DatasetFormatError):
        validate_record(record, path="test", line_number=1)


def test_last_message_must_be_assistant():
    record = {"messages": [{"role": "assistant", "content": "hi"}, {"role": "user", "content": "bye"}]}
    with pytest.raises(DatasetFormatError):
        validate_record(record, path="test", line_number=1)


def test_error_includes_line_number():
    with pytest.raises(DatasetFormatError) as exc_info:
        validate_record({}, path="test.jsonl", line_number=42)
    assert "42" in str(exc_info.value)


def test_validate_file_bundled_tiny_dataset():
    from pathlib import Path

    tiny_dir = Path(__file__).parents[2] / "src" / "angel_ai" / "data" / "examples" / "tiny"
    count = validate_file(tiny_dir / "train.jsonl")
    assert count == 4


def test_alpaca_conversion():
    record = {"instruction": "Say hi", "input": "", "output": "Hi there!"}
    assert is_alpaca_style(record)
    converted = alpaca_to_chat(record)
    assert converted["messages"][-1]["role"] == "assistant"
    assert converted["messages"][-1]["content"] == "Hi there!"


def test_to_canonical_passes_through_chat_records():
    record = {"messages": [{"role": "user", "content": "hi"}]}
    assert to_canonical(record) == record


def test_validate_file_missing_raises():
    with pytest.raises(DatasetFormatError):
        validate_file("does/not/exist.jsonl")


def test_validate_file_bad_json_raises(tmp_path):
    bad_file = tmp_path / "bad.jsonl"
    bad_file.write_text("{not valid json}\n")
    with pytest.raises(DatasetFormatError):
        validate_file(bad_file)
