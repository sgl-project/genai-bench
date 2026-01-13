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


def test_huggingface_dataset_source_local_path_skips_dataset_info(
    monkeypatch, tmp_path
):
    """When path is a local directory, dataset_info should not be called."""
    # Create a valid local directory
    local_dir = tmp_path / "local_dataset"
    local_dir.mkdir()

    called = {"dataset_info": False, "load_dataset": False}

    def fake_dataset_info(*args, **kwargs):  # pragma: no cover - should not be called
        called["dataset_info"] = True
        raise RuntimeError("dataset_info should not be called for local paths")

    def fake_load_dataset(path, **kwargs):
        called["load_dataset"] = True
        return {"ok": True}

    monkeypatch.setattr("genai_bench.data.sources.dataset_info", fake_dataset_info)
    monkeypatch.setattr("genai_bench.data.sources.load_dataset", fake_load_dataset)

    config = DatasetSourceConfig(
        type="huggingface",
        path=str(local_dir),
        huggingface_kwargs={"split": "train"},
    )
    source = HuggingFaceDatasetSource(config)

    result = source.load()
    assert result == {"ok": True}
    assert not called["dataset_info"]
    assert called["load_dataset"]


def test_huggingface_dataset_source_local_path_not_exists(tmp_path):
    """Test that non-existent local path raises ValueError."""
    non_existent_path = tmp_path / "non_existent_dataset"

    config = DatasetSourceConfig(
        type="huggingface",
        path=str(non_existent_path),
    )
    source = HuggingFaceDatasetSource(config)

    with pytest.raises(ValueError, match="Repo id must be in the form"):
        source.load()


def test_huggingface_dataset_source_local_path_is_file(tmp_path):
    """Test that local path pointing to a file raises ValueError."""
    # Create a file instead of directory
    local_file = tmp_path / "dataset.txt"
    local_file.write_text("some content")

    config = DatasetSourceConfig(
        type="huggingface",
        path=str(local_file),
    )
    source = HuggingFaceDatasetSource(config)

    with pytest.raises(ValueError, match="Repo id must be in the form"):
        source.load()


def test_huggingface_dataset_source_remote_not_found(monkeypatch):
    """Test that remote dataset not found raises ValueError."""
    from datasets.exceptions import DatasetNotFoundError

    def fake_dataset_info(*args, **kwargs):
        raise DatasetNotFoundError("Dataset not found")

    monkeypatch.setattr("genai_bench.data.sources.dataset_info", fake_dataset_info)

    config = DatasetSourceConfig(
        type="huggingface",
        path="non-existent/dataset",
    )
    source = HuggingFaceDatasetSource(config)

    with pytest.raises(
        ValueError,
        match="Dataset 'non-existent/dataset' not found on HuggingFace Hub",
    ):
        source.load()
