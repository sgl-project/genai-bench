from unittest.mock import MagicMock, patch

import pytest

from genai_bench.data.config import DatasetConfig, DatasetSourceConfig
from genai_bench.data.loaders.image import ImageDatasetLoader


@pytest.fixture
def mock_dataset():
    mock_dataset = MagicMock()
    mock_dataset.__len__.return_value = 2
    mock_dataset.features = ["image_column", "prompt_column"]
    mock_dataset.__iter__.return_value = iter(
        [
            {"image_column": "image_data_1", "prompt_column": "prompt_1"},
            {"image_column": "image_data_2", "prompt_column": "prompt_2"},
        ]
    )
    return mock_dataset


@pytest.fixture
def mock_empty_dataset():
    return []


@pytest.fixture
def dataset_config():
    return DatasetConfig(
        source=DatasetSourceConfig(
            type="huggingface",
            path="test/dataset",
            huggingface_kwargs={"subset": None, "split": None, "revision": None},
        ),
        prompt_column="prompt_column",
        image_column="image_column",
    )


@patch("genai_bench.data.sources.DatasetSourceFactory.create")
def test_load_requests_success(mock_factory, mock_dataset, dataset_config):
    """Test if load_requests successfully loads image dataset"""
    mock_source = MagicMock()
    mock_source.load.return_value = mock_dataset
    mock_factory.return_value = mock_source

    results = ImageDatasetLoader(dataset_config).load_request()

    expected_results = [("prompt_1", "image_data_1"), ("prompt_2", "image_data_2")]
    assert results == expected_results


@patch("genai_bench.data.sources.DatasetSourceFactory.create")
def test_load_requests_missing_prompt(mock_factory, mock_dataset, dataset_config):
    """Test if load_requests handles missing prompt column"""
    mock_dataset.__iter__.return_value = [
        {"image_column": "image_data_1"},
        {"image_column": "image_data_2"},
    ]
    mock_source = MagicMock()
    mock_source.load.return_value = mock_dataset
    mock_factory.return_value = mock_source
    dataset_config.prompt_column = None

    results = ImageDatasetLoader(dataset_config).load_request()

    expected_results = [("", "image_data_1"), ("", "image_data_2")]
    assert results == expected_results


@patch("genai_bench.data.sources.DatasetSourceFactory.create")
def test_load_requests_empty_dataset(mock_factory, mock_empty_dataset, dataset_config):
    """Test if load_requests handles empty dataset gracefully"""
    mock_source = MagicMock()
    mock_source.load.return_value = mock_empty_dataset
    mock_factory.return_value = mock_source

    results = ImageDatasetLoader(dataset_config).load_request()
    assert results == []
