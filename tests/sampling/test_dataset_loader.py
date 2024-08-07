from unittest.mock import MagicMock, patch

import pytest
from datasets.exceptions import DatasetNotFoundError

from genai_bench.sampling.dataset_loader import (
    DatasetConfig,
    DatasetFormat,
    DatasetLoader,
    DatasetPath,
)


def test_default_initialization():
    dataset_path = DatasetPath()
    assert dataset_path.type == DatasetFormat.TEXT
    assert dataset_path.path.endswith("sonnet.txt")


@patch("genai_bench.sampling.dataset_loader.Path.is_file", return_value=True)
@patch("genai_bench.sampling.dataset_loader.DatasetFormat")
def test_from_value_with_local_file(mock_dataset_format, mock_is_file):
    mock_dataset_format.return_value = DatasetFormat.TEXT
    dataset_path = DatasetPath.from_value("/path/to/file.txt")
    assert dataset_path.path == "/path/to/file.txt"
    assert dataset_path.type == DatasetFormat.TEXT


@patch("genai_bench.sampling.dataset_loader.Path.is_file", return_value=False)
@patch("genai_bench.sampling.dataset_loader.dataset_info")
def test_from_value_with_huggingface_dataset(mock_dataset_info, mock_is_file):
    mock_dataset_info.return_value = MagicMock()
    dataset_path = DatasetPath.from_value("huggingface/dataset")
    assert dataset_path.path == "huggingface/dataset"
    assert dataset_path.type == DatasetFormat.HUGGINGFACE_HUB
    mock_dataset_info.assert_called_once()


@patch("genai_bench.sampling.dataset_loader.Path.is_file", return_value=False)
@patch(
    "genai_bench.sampling.dataset_loader.dataset_info",
    side_effect=DatasetNotFoundError("Invalid dataset"),
)
def test_from_value_with_invalid_path(file_mock, dataset_info_mock):
    with pytest.raises(
        ValueError,
        match=r"Provided `--dataset-path` .* nether a local file nor an accessible"
        r" dataset. If its gated repo, please set HUGGINGFACE_API_KEY "
        r"environment variable.",
    ):
        DatasetPath.from_value("invalid/path")


@patch.multiple(DatasetLoader, __abstractmethods__=set())
def test_load_request():
    dataset_path = DatasetPath()
    dataset_path.type = DatasetFormat.TEXT
    mock_dataset_config = MagicMock(spec=DatasetConfig)
    mock_dataset_config.dataset_path = dataset_path
    dataset_loader = DatasetLoader(mock_dataset_config)
    dataset_loader.media_type = "Image"
    dataset_loader.supported_formats = set([DatasetFormat.HUGGINGFACE_HUB])

    with pytest.raises(
        ValueError, match=f"{DatasetFormat.TEXT} is unsupported for Image"
    ):
        dataset_loader.load_request()
