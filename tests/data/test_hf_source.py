"""Tests for Hugging Face Hub dataset ingestion. Mocks `datasets.load_dataset`
so these stay fast/offline -- no network access needed.
"""

import pytest
from datasets import Dataset, DatasetDict

from angel_ai.data.hf_source import load_from_hub
from angel_ai.errors import DatasetFormatError


def test_load_from_hub_messages_format(mocker):
    fake = DatasetDict(
        {
            "train": Dataset.from_list(
                [{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}]
            )
        }
    )
    mocker.patch("datasets.load_dataset", return_value=fake)

    result = load_from_hub("fake/repo")
    assert "train" in result
    assert result["train"][0]["messages"][-1]["content"] == "hello"


def test_load_from_hub_alpaca_format(mocker):
    fake = DatasetDict(
        {
            "train": Dataset.from_list(
                [{"instruction": "Say hi", "input": "", "output": "Hi there!"}]
            )
        }
    )
    mocker.patch("datasets.load_dataset", return_value=fake)

    result = load_from_hub("fake/repo")
    assert result["train"][0]["messages"][-1]["role"] == "assistant"
    assert result["train"][0]["messages"][-1]["content"] == "Hi there!"


def test_load_from_hub_generic_column_mapping(mocker):
    fake = DatasetDict(
        {"train": Dataset.from_list([{"question": "2+2?", "answer": "4"}])}
    )
    mocker.patch("datasets.load_dataset", return_value=fake)

    result = load_from_hub("fake/repo", prompt_column="question", response_column="answer")
    messages = result["train"][0]["messages"]
    assert messages[0] == {"role": "user", "content": "2+2?"}
    assert messages[1] == {"role": "assistant", "content": "4"}


def test_load_from_hub_unmappable_columns_raises(mocker):
    fake = DatasetDict({"train": Dataset.from_list([{"foo": "bar"}])})
    mocker.patch("datasets.load_dataset", return_value=fake)

    with pytest.raises(DatasetFormatError):
        load_from_hub("fake/repo")


def test_load_from_hub_split_mapping(mocker):
    fake = DatasetDict(
        {
            "train": Dataset.from_list(
                [{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hey"}]}]
            ),
            "holdout": Dataset.from_list(
                [{"messages": [{"role": "user", "content": "bye"}, {"role": "assistant", "content": "bye!"}]}]
            ),
        }
    )
    mocker.patch("datasets.load_dataset", return_value=fake)

    result = load_from_hub("fake/repo", split_mapping={"validation": "holdout"})
    assert "validation" in result
    assert result["validation"][0]["messages"][-1]["content"] == "bye!"
    assert "holdout" not in result  # renamed to our canonical split name, not a hub split name


def test_load_from_hub_missing_train_raises(mocker):
    fake = DatasetDict(
        {
            "test": Dataset.from_list(
                [{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hey"}]}]
            )
        }
    )
    mocker.patch("datasets.load_dataset", return_value=fake)

    with pytest.raises(DatasetFormatError):
        load_from_hub("fake/repo")


def test_load_from_hub_single_dataset_treated_as_train(mocker):
    """A hub dataset with no configs/splits returns a plain Dataset, not a
    DatasetDict -- should be treated as the train split."""
    fake = Dataset.from_list(
        [{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hey"}]}]
    )
    mocker.patch("datasets.load_dataset", return_value=fake)

    result = load_from_hub("fake/repo")
    assert "train" in result
