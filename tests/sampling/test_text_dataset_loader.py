from unittest.mock import MagicMock, patch

import pytest

from genai_bench.sampling.dataset_loader import (
    DatasetConfig,
    DatasetFormat,
    DatasetPath,
)
from genai_bench.sampling.text_dataset_loader import TextDatasetLoader


@pytest.fixture
def mock_txt_file(tmp_path):
    file_path = tmp_path / "test.txt"
    file_path.write_text("Line 1\nLine 2\nLine 3")
    return file_path


@pytest.fixture
def mock_csv_file(tmp_path):
    file_path = tmp_path / "test.csv"
    file_path.write_text("col1,col2\nval1,val2\nval3,val4")
    return file_path


@pytest.fixture
def mock_dataset_config_txt(mock_txt_file):
    return DatasetConfig(
        dataset_path=DatasetPath(type=DatasetFormat.TEXT, path=str(mock_txt_file)),
        hf_prompt_column_name=None,
        hf_image_column_name=None,
        hf_subset=None,
        hf_split=None,
        hf_revision=None,
        dataset_prompt_column_index=0,
    )


@pytest.fixture
def mock_dataset_config_csv(mock_csv_file):
    return DatasetConfig(
        dataset_path=DatasetPath(type=DatasetFormat.CSV, path=str(mock_csv_file)),
        hf_prompt_column_name=None,
        hf_image_column_name=None,
        hf_subset=None,
        hf_split=None,
        hf_revision=None,
        dataset_prompt_column_index=0,
    )


@pytest.fixture
def mock_dataset_config_hf():
    return DatasetConfig(
        dataset_path=DatasetPath(
            type=DatasetFormat.HUGGINGFACE_HUB, path="huggingface/dataset"
        ),
        hf_prompt_column_name="text",
        hf_image_column_name=None,
        hf_subset=None,
        hf_split="train",
        hf_revision=None,
        dataset_prompt_column_index=0,
    )


@patch("genai_bench.sampling.text_dataset_loader.load_dataset")
def test_load_txt_file(mock_load_dataset, mock_dataset_config_txt):
    mock_load_dataset.return_value = {"train": {"text": ["Line 1", "Line 2", "Line 3"]}}

    result = TextDatasetLoader(mock_dataset_config_txt).load_request()

    assert result == ["Line 1", "Line 2", "Line 3"]
    mock_load_dataset.assert_called_once_with(
        "text", data_files=mock_dataset_config_txt.dataset_path.path
    )


@patch("genai_bench.sampling.text_dataset_loader.load_dataset")
def test_load_csv_file(mock_load_dataset, mock_dataset_config_csv):
    mock_dataset = MagicMock()
    mock_dataset.__len__.return_value = 2
    train = MagicMock()
    train.__getitem__.return_value = ["val1", "val3"]
    train.features.__len__.return_value = 1
    mock_dataset.__getitem__.return_value = train
    mock_load_dataset.return_value = mock_dataset

    result = TextDatasetLoader(mock_dataset_config_csv).load_request()

    assert result == ["val1", "val3"]
    mock_load_dataset.assert_called_once_with(
        "csv", data_files=mock_dataset_config_csv.dataset_path.path, header=None
    )


@patch("genai_bench.sampling.dataset_loader.load_dataset")
def test_load_huggingface_hub(mock_load_dataset, mock_dataset_config_hf):
    mock_dataset = MagicMock()
    mock_dataset.__getitem__.return_value = ["sample1", "sample2"]
    mock_dataset.features = {"text": "text"}
    mock_dataset.__len__.return_value = 2
    mock_load_dataset.return_value = mock_dataset

    result = TextDatasetLoader(mock_dataset_config_hf).load_request()

    assert result == ["sample1", "sample2"]
    mock_load_dataset.assert_called_once_with(
        "huggingface/dataset", name=None, split="train", revision=None
    )


@patch("genai_bench.sampling.text_dataset_loader.load_dataset")
def test_load_csv_invalid_column(mock_load_dataset, mock_dataset_config_csv):
    mock_dataset = MagicMock()
    mock_dataset.__len__.return_value = 2
    train = MagicMock()
    train.__getitem__.return_value = ["val1", "val3"]
    train.features.__len__.return_value = 1
    mock_dataset.__getitem__.return_value = train
    mock_load_dataset.return_value = mock_dataset
    mock_dataset_config_csv.dataset_prompt_column_index = 5  # Invalid column index

    with pytest.raises(ValueError, match="Column index '5' is out of bounds"):
        TextDatasetLoader(mock_dataset_config_csv).load_request()
