"""Tests for dataset source implementations."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from datasets.exceptions import DatasetNotFoundError

from genai_bench.data.config import DatasetSourceConfig
from genai_bench.data.sources import (
    CustomDatasetSource,
    DatasetSourceFactory,
    FileDatasetSource,
    HuggingFaceDatasetSource,
)


class TestFileDatasetSource:
    """Test FileDatasetSource implementation."""

    def test_load_txt_file(self):
        """Test loading text file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Line 1\nLine 2\nLine 3")
            f.flush()

            config = DatasetSourceConfig(type="file", path=f.name)
            source = FileDatasetSource(config)

            result = source.load()

            assert result == ["Line 1", "Line 2", "Line 3"]

        Path(f.name).unlink()

    def test_load_csv_file(self):
        """Test loading CSV file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("text,label\n")
            f.write("Hello,positive\n")
            f.write("World,negative\n")
            f.flush()

            config = DatasetSourceConfig(
                type="file", path=f.name, text_column="text", label_column="label"
            )
            source = FileDatasetSource(config)

            result = source.load()

            # CSV files return dict-like structure
            assert isinstance(result, dict)
            assert "text" in result
            assert "label" in result
            assert result["text"] == ["Hello", "World"]
            assert result["label"] == ["positive", "negative"]

        Path(f.name).unlink()

    def test_load_json_file(self):
        """Test loading JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            data = [
                {"text": "First item", "metadata": "meta1"},
                {"text": "Second item", "metadata": "meta2"},
            ]
            json.dump(data, f)
            f.flush()

            config = DatasetSourceConfig(type="file", path=f.name, text_column="text")
            source = FileDatasetSource(config)

            result = source.load()

            # JSON files return list structure
            assert isinstance(result, list)
            assert len(result) == 2
            assert result[0]["text"] == "First item"
            assert result[1]["text"] == "Second item"

        Path(f.name).unlink()

    def test_load_missing_file(self):
        """Test loading missing file."""
        config = DatasetSourceConfig(type="file", path="/nonexistent/file.txt")
        source = FileDatasetSource(config)

        with pytest.raises(FileNotFoundError, match="Dataset file not found"):
            source.load()

    def test_load_no_path(self):
        """Test loading without path."""
        config = DatasetSourceConfig(type="file")
        source = FileDatasetSource(config)

        with pytest.raises(ValueError, match="File path is required"):
            source.load()

    def test_load_unsupported_format(self):
        """Test loading unsupported file format."""
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            config = DatasetSourceConfig(type="file", path=f.name)
            source = FileDatasetSource(config)

            with pytest.raises(ValueError, match="Unsupported file format"):
                source.load()

        Path(f.name).unlink()


class TestHuggingFaceDatasetSource:
    """Test HuggingFaceDatasetSource implementation."""

    @patch("genai_bench.data.sources.load_dataset")
    @patch("genai_bench.data.sources.dataset_info")
    def test_load_dataset(self, mock_dataset_info, mock_load_dataset):
        """Test loading HuggingFace dataset."""
        mock_dataset = MagicMock()
        mock_dataset.__getitem__.return_value = {
            "train": pd.DataFrame({"text": ["Sample 1", "Sample 2"], "label": [0, 1]})
        }
        mock_load_dataset.return_value = mock_dataset

        config = DatasetSourceConfig(
            type="huggingface",
            path="test_dataset",
            huggingface_kwargs={"name": "default", "split": "train"},
            text_column="text",
        )
        source = HuggingFaceDatasetSource(config)

        result = source.load()

        mock_load_dataset.assert_called_once_with(
            "test_dataset", name="default", split="train"
        )
        assert result == mock_dataset

    @patch("genai_bench.data.sources.load_dataset")
    @patch("genai_bench.data.sources.dataset_info")
    def test_load_dataset_with_labels(self, mock_dataset_info, mock_load_dataset):
        """Test loading HuggingFace dataset with labels."""
        mock_data = pd.DataFrame(
            {"text": ["Sample 1", "Sample 2"], "label": ["pos", "neg"]}
        )
        mock_load_dataset.return_value = mock_data

        config = DatasetSourceConfig(
            type="huggingface",
            path="test_dataset",
            text_column="text",
            label_column="label",
        )
        source = HuggingFaceDatasetSource(config)

        result = source.load()

        pd.testing.assert_frame_equal(result, mock_data)

    @patch("genai_bench.data.sources.dataset_info")
    def test_load_dataset_not_found(self, mock_dataset_info):
        """Test loading non-existent dataset."""
        mock_dataset_info.side_effect = DatasetNotFoundError("Dataset not found")

        config = DatasetSourceConfig(type="huggingface", path="nonexistent_dataset")
        source = HuggingFaceDatasetSource(config)

        with pytest.raises(ValueError, match="Dataset 'nonexistent_dataset' not found"):
            source.load()

    def test_load_no_name(self):
        """Test loading without dataset name."""
        config = DatasetSourceConfig(type="huggingface")
        source = HuggingFaceDatasetSource(config)

        with pytest.raises(ValueError, match="Dataset ID is required"):
            source.load()


class TestCustomDatasetSource:
    """Test CustomDatasetSource implementation."""

    def test_load_custom_loader(self):
        """Test loading dataset from custom loader class."""
        # Create a mock module with a loader class
        with patch("genai_bench.data.sources.importlib.import_module") as mock_import:
            mock_module = MagicMock()
            mock_loader_class = MagicMock()
            mock_loader_instance = MagicMock()
            mock_loader_instance.load.return_value = ["Custom data 1", "Custom data 2"]
            mock_loader_class.return_value = mock_loader_instance

            mock_module.MyLoader = mock_loader_class
            mock_import.return_value = mock_module

            config = DatasetSourceConfig(
                type="custom",
                loader_class="my_module.MyLoader",
                loader_kwargs={"param": "value"},
            )
            source = CustomDatasetSource(config)

            result = source.load()

            assert result == ["Custom data 1", "Custom data 2"]
            mock_import.assert_called_once_with("my_module")
            mock_loader_class.assert_called_once_with(param="value")

    def test_load_no_loader_class(self):
        """Test loading without loader class."""
        config = DatasetSourceConfig(type="custom")
        source = CustomDatasetSource(config)

        with pytest.raises(ValueError, match="Loader class is required"):
            source.load()

    def test_load_loader_without_load_method(self):
        """Test loading with loader missing load method."""
        with patch("genai_bench.data.sources.importlib.import_module") as mock_import:
            mock_module = MagicMock()
            mock_loader_class = MagicMock()
            mock_loader_instance = MagicMock()
            del mock_loader_instance.load  # Remove load method
            mock_loader_class.return_value = mock_loader_instance

            mock_module.MyLoader = mock_loader_class
            mock_import.return_value = mock_module

            config = DatasetSourceConfig(
                type="custom", loader_class="my_module.MyLoader"
            )
            source = CustomDatasetSource(config)

            with pytest.raises(AttributeError, match="must have a 'load' method"):
                source.load()

    def test_load_loader_not_found(self):
        """Test loading with non-existent loader."""
        config = DatasetSourceConfig(
            type="custom", loader_class="nonexistent.module.Loader"
        )
        source = CustomDatasetSource(config)

        with pytest.raises(ImportError, match="Failed to import custom loader class"):
            source.load()


class TestDatasetSourceFactory:
    """Test DatasetSourceFactory implementation."""

    def test_create_file_source(self):
        """Test creating file dataset source."""
        config = DatasetSourceConfig(type="file", path="test.txt")
        source = DatasetSourceFactory.create(config)
        assert isinstance(source, FileDatasetSource)

    def test_create_huggingface_source(self):
        """Test creating HuggingFace dataset source."""
        config = DatasetSourceConfig(type="huggingface", path="dataset/name")
        source = DatasetSourceFactory.create(config)
        assert isinstance(source, HuggingFaceDatasetSource)

    def test_create_custom_source(self):
        """Test creating custom dataset source."""
        config = DatasetSourceConfig(type="custom", loader_class="module.Loader")
        source = DatasetSourceFactory.create(config)
        assert isinstance(source, CustomDatasetSource)

    def test_create_unknown_source(self):
        """Test creating unknown dataset source."""
        # Test that unknown type is caught at config validation
        with pytest.raises(ValueError, match="Dataset source type must be one of"):
            DatasetSourceConfig(type="unknown")

    def test_register_source(self):
        """Test registering new dataset source."""

        class TestSource:
            pass

        DatasetSourceFactory.register_source("test", TestSource)
        assert DatasetSourceFactory._sources["test"] == TestSource

        # Clean up
        del DatasetSourceFactory._sources["test"]
