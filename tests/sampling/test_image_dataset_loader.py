from unittest.mock import MagicMock, patch

import pytest

from genai_bench.sampling.dataset_loader import (
    DatasetConfig,
    DatasetFormat,
    DatasetPath,
)
from genai_bench.sampling.image_dataset_loader import ImageDatasetLoader


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
    dataset_path = DatasetPath()
    dataset_path.type = DatasetFormat.HUGGINGFACE_HUB
    return DatasetConfig(
        dataset_path=dataset_path,
        hf_prompt_column_name="prompt_column",
        hf_image_column_name="image_column",
        hf_subset=None,
        hf_split=None,
        hf_revision=None,
        dataset_prompt_column_index=0,
    )


@patch("genai_bench.sampling.dataset_loader.load_dataset")
def test_load_requests_success(mock_load_dataset, mock_dataset, dataset_config):
    """Test if load_requests successfully loads image dataset"""
    mock_load_dataset.return_value = mock_dataset

    results = ImageDatasetLoader(dataset_config).load_request()

    expected_results = [("prompt_1", "image_data_1"), ("prompt_2", "image_data_2")]
    assert results == expected_results


@patch("genai_bench.sampling.dataset_loader.load_dataset")
def test_load_requests_missing_prompt(mock_load_dataset, mock_dataset, dataset_config):
    """Test if load_requests handles missing prompt column"""
    mock_dataset.__iter__.return_value = [
        {"image_column": "image_data_1"},
        {"image_column": "image_data_2"},
    ]
    mock_load_dataset.return_value = mock_dataset
    dataset_config.hf_prompt_column_name = None

    results = ImageDatasetLoader(dataset_config).load_request()

    expected_results = [("", "image_data_1"), ("", "image_data_2")]
    assert results == expected_results


@patch("genai_bench.sampling.dataset_loader.load_dataset")
def test_load_requests_empty_dataset(
    mock_load_dataset, mock_empty_dataset, dataset_config
):
    """Test if load_requests raises an assertion error when dataset is empty"""
    mock_load_dataset.return_value = mock_empty_dataset
    with pytest.raises(AssertionError):
        ImageDatasetLoader(dataset_config).load_request()
