"""Tests for data loader factory."""

from unittest.mock import MagicMock, patch

import pytest

from genai_bench.data.config import DatasetConfig, DatasetSourceConfig
from genai_bench.data.loaders.factory import DataLoaderFactory


def test_load_data_for_text_task():
    """Test loading data for text-to-text task."""
    config = DatasetConfig(
        source=DatasetSourceConfig(
            type="file", path="/path/to/file.txt", file_format="txt"
        )
    )

    with patch(
        "genai_bench.data.loaders.factory.TextDatasetLoader"
    ) as mock_loader_class:
        mock_loader = MagicMock()
        mock_loader.load_request.return_value = ["text1", "text2"]
        mock_loader_class.return_value = mock_loader

        data, use_scenario = DataLoaderFactory.load_data_for_task(
            "text-to-text", config
        )

        assert data == ["text1", "text2"]
        assert use_scenario is True  # txt files use scenario-based sampling


def test_load_data_for_embeddings_with_non_sonnet():
    """Test loading data for embeddings task with non-sonnet dataset."""
    config = DatasetConfig(
        source=DatasetSourceConfig(type="huggingface", path="other/dataset")
    )

    with patch(
        "genai_bench.data.loaders.factory.TextDatasetLoader"
    ) as mock_loader_class:
        mock_loader = MagicMock()
        mock_loader.load_request.return_value = ["text1", "text2"]
        mock_loader_class.return_value = mock_loader

        data, use_scenario = DataLoaderFactory.load_data_for_task(
            "text-to-embeddings", config
        )

        # Should use sonnet.txt for embeddings
        assert mock_loader_class.call_count == 1
        created_config = mock_loader_class.call_args[0][0]
        assert "sonnet.txt" in created_config.source.path


def test_load_data_for_image_task():
    """Test loading data for image-to-text task."""
    config = DatasetConfig(
        source=DatasetSourceConfig(type="huggingface", path="image/dataset")
    )

    with patch(
        "genai_bench.data.loaders.factory.ImageDatasetLoader"
    ) as mock_loader_class:
        mock_loader = MagicMock()
        mock_loader.load_request.return_value = [
            ("prompt1", "image1"),
            ("prompt2", "image2"),
        ]
        mock_loader_class.return_value = mock_loader

        data, use_scenario = DataLoaderFactory.load_data_for_task(
            "image-to-text", config
        )

        assert data == [("prompt1", "image1"), ("prompt2", "image2")]
        assert use_scenario is False  # images don't use scenario-based sampling


def test_load_data_for_invalid_task():
    """Test loading data for unsupported task."""
    config = DatasetConfig(
        source=DatasetSourceConfig(
            type="file", path="/path/to/file.txt", file_format="txt"
        )
    )

    with pytest.raises(ValueError, match="Unsupported input modality: video"):
        DataLoaderFactory.load_data_for_task("video-to-text", config)
