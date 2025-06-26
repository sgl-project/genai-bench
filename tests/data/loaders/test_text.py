from unittest.mock import MagicMock, patch

import pytest

from genai_bench.data.config import DatasetConfig, DatasetSourceConfig
from genai_bench.data.loaders.text import TextDatasetLoader


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
        source=DatasetSourceConfig(
            type="file", path=str(mock_txt_file), file_format="txt"
        ),
        prompt_column="prompt",
        image_column="image",
    )


@pytest.fixture
def mock_dataset_config_csv(mock_csv_file):
    return DatasetConfig(
        source=DatasetSourceConfig(
            type="file", path=str(mock_csv_file), file_format="csv"
        ),
        prompt_column="prompt",
        image_column="image",
    )


@pytest.fixture
def mock_dataset_config_hf():
    return DatasetConfig(
        source=DatasetSourceConfig(
            type="huggingface",
            path="huggingface/dataset",
            huggingface_kwargs={"split": "train"},
        ),
        prompt_column="text",
        image_column="image",
    )


@patch("genai_bench.data.sources.FileDatasetSource.load")
def test_load_txt_file(mock_load, mock_dataset_config_txt):
    mock_load.return_value = ["Line 1", "Line 2", "Line 3"]

    result = TextDatasetLoader(mock_dataset_config_txt).load_request()

    assert result == ["Line 1", "Line 2", "Line 3"]
    mock_load.assert_called_once()


@patch("genai_bench.data.sources.DatasetSourceFactory.create")
def test_load_csv_file(mock_factory, mock_dataset_config_csv):
    """Test loading CSV file with column names."""
    mock_source = MagicMock()
    # Mock CSV data as a dictionary (like pandas returns)
    mock_source.load.return_value = {
        "prompt": ["val1", "val3"],
        "other_col": ["x", "y"],
    }
    mock_factory.return_value = mock_source

    result = TextDatasetLoader(mock_dataset_config_csv).load_request()

    assert result == ["val1", "val3"]
    mock_factory.assert_called_once()
    mock_source.load.assert_called_once()


@patch("genai_bench.data.sources.DatasetSourceFactory.create")
def test_load_huggingface_hub(mock_factory, mock_dataset_config_hf):
    mock_source = MagicMock()
    mock_source.load.return_value = ["sample1", "sample2"]
    mock_factory.return_value = mock_source

    result = TextDatasetLoader(mock_dataset_config_hf).load_request()

    assert result == ["sample1", "sample2"]
    mock_factory.assert_called_once()
    mock_source.load.assert_called_once()


@patch("genai_bench.data.sources.DatasetSourceFactory.create")
def test_load_csv_invalid_column(mock_factory, mock_dataset_config_csv):
    """Test handling of invalid column name in CSV file."""
    # Mock CSV data as a dictionary (like pandas returns)
    mock_source = MagicMock()
    mock_source.load.return_value = {"text": ["value1", "value2"], "label": ["A", "B"]}
    mock_factory.return_value = mock_source

    # Set an invalid column name that doesn't exist in the data
    mock_dataset_config_csv.prompt_column = "invalid_column"

    # Should raise ValueError with helpful message about available columns
    with pytest.raises(
        ValueError, match="Column 'invalid_column' not found in CSV file"
    ):
        TextDatasetLoader(mock_dataset_config_csv).load_request()
