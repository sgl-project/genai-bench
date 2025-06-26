"""Tests for data configuration models."""

import pytest
from pydantic import ValidationError

from genai_bench.data.config import DatasetConfig, DatasetSourceConfig


def test_dataset_source_config_file():
    """Test file-based dataset source configuration."""
    config = DatasetSourceConfig(
        type="file", path="/path/to/file.txt", file_format="txt"
    )
    assert config.type == "file"
    assert config.path == "/path/to/file.txt"
    assert config.file_format == "txt"


def test_dataset_source_config_huggingface():
    """Test HuggingFace dataset source configuration."""
    config = DatasetSourceConfig(
        type="huggingface",
        path="dataset/name",
        huggingface_kwargs={"subset": "subset", "split": "train", "revision": "main"},
    )
    assert config.type == "huggingface"
    assert config.path == "dataset/name"
    assert config.huggingface_kwargs["subset"] == "subset"
    assert config.huggingface_kwargs["split"] == "train"
    assert config.huggingface_kwargs["revision"] == "main"


def test_dataset_source_config_csv():
    """Test CSV dataset source configuration."""
    config = DatasetSourceConfig(
        type="file", path="/path/to/file.csv", file_format="csv"
    )
    assert config.type == "file"
    assert config.file_format == "csv"


def test_dataset_config():
    """Test complete dataset configuration."""
    source_config = DatasetSourceConfig(
        type="file", path="/path/to/file.txt", file_format="txt"
    )
    config = DatasetConfig(
        source=source_config, prompt_column="question", image_column="img"
    )
    assert config.source == source_config
    assert config.prompt_column == "question"
    assert config.image_column == "img"


def test_dataset_config_defaults():
    """Test dataset configuration with default values."""
    source_config = DatasetSourceConfig(
        type="file", path="/path/to/file.txt", file_format="txt"
    )
    config = DatasetConfig(source=source_config)
    assert config.prompt_column == "prompt"
    assert config.image_column == "image"


def test_invalid_dataset_source_type():
    """Test validation of invalid dataset source type."""
    with pytest.raises(ValidationError):
        DatasetSourceConfig(type="invalid_type", path="/some/path")


def test_dataset_config_from_dict():
    """Test creating dataset config from dictionary."""
    config_dict = {
        "source": {
            "type": "huggingface",
            "path": "test/dataset",
            "huggingface_kwargs": {"split": "train"},
        },
        "prompt_column": "text",
        "image_column": "image",
    }
    config = DatasetConfig(**config_dict)
    assert config.source.type == "huggingface"
    assert config.source.path == "test/dataset"
    assert config.source.huggingface_kwargs["split"] == "train"
    assert config.prompt_column == "text"
