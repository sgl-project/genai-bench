"""Tests for dataset source implementations."""

import pytest

from genai_bench.data.config import DatasetSourceConfig
from genai_bench.data.sources import (
    CustomDatasetSource,
    DatasetSourceFactory,
    FileDatasetSource,
    HuggingFaceDatasetSource,
)


def test_dataset_source_factory_file():
    """Test factory creates file dataset source."""
    config = DatasetSourceConfig(
        type="file", path="/path/to/file.txt", file_format="txt"
    )
    source = DatasetSourceFactory.create(config)
    assert isinstance(source, FileDatasetSource)


def test_dataset_source_factory_huggingface():
    """Test factory creates HuggingFace dataset source."""
    config = DatasetSourceConfig(type="huggingface", path="dataset/name")
    source = DatasetSourceFactory.create(config)
    assert isinstance(source, HuggingFaceDatasetSource)


def test_dataset_source_factory_custom():
    """Test factory creates custom dataset source."""
    config = DatasetSourceConfig(
        type="custom", path="custom/path", loader_class="CustomLoader"
    )
    source = DatasetSourceFactory.create(config)
    assert isinstance(source, CustomDatasetSource)


def test_dataset_source_factory_invalid():
    """Test factory raises error for invalid source type."""
    # The validation happens at the config level, not the factory level
    with pytest.raises(ValueError, match="Dataset source type must be one of"):
        DatasetSourceConfig(type="invalid", path="/some/path")


# NOTE: Implementation tests for actual data sources are covered in integration tests
# The factory tests above verify that the correct source types are created


def test_custom_dataset_source_no_loader():
    """Test custom dataset source without loader class."""
    config = DatasetSourceConfig(type="custom", path="custom/path")
    source = CustomDatasetSource(config)

    with pytest.raises(
        ValueError, match="Loader class is required for custom dataset source"
    ):
        source.load()
