from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict
from datasets import IterableDataset as HFIterableDataset

from genai_bench.data.config import DatasetConfig, DatasetSourceConfig
from genai_bench.data.loaders.image import ImageDatasetLoader


@pytest.fixture
def dataset_config():
    return DatasetConfig(
        source=DatasetSourceConfig(type="huggingface", path="test/dataset"),
        prompt_column="prompt_column",
        image_column="image_column",
    )


@patch("genai_bench.data.sources.DatasetSourceFactory.create")
def test_returns_hf_dataset_as_is(mock_factory, dataset_config):
    ds = HFDataset.from_dict({"image_column": [1, 2], "prompt_column": ["a", "b"]})
    mock_source = MagicMock()
    mock_source.load.return_value = ds
    mock_factory.return_value = mock_source

    results = ImageDatasetLoader(dataset_config).load_request()
    assert isinstance(results, HFDataset)
    assert len(results) == 2


@patch("genai_bench.data.sources.DatasetSourceFactory.create")
def test_selects_train_split_from_datasetdict(mock_factory, dataset_config):
    train = HFDataset.from_dict({"image_column": [1], "prompt_column": ["t"]})
    valid = HFDataset.from_dict({"image_column": [2], "prompt_column": ["v"]})
    dd = HFDatasetDict({"train": train, "validation": valid})
    mock_source = MagicMock()
    mock_source.load.return_value = dd
    mock_factory.return_value = mock_source

    results = ImageDatasetLoader(dataset_config).load_request()
    assert isinstance(results, HFDataset)
    assert len(results) == 1
    assert results[0]["prompt_column"] == "t"


@patch("genai_bench.data.sources.DatasetSourceFactory.create")
def test_datasetdict_no_splits_raises(mock_factory, dataset_config):
    dd = HFDatasetDict({})
    mock_source = MagicMock()
    mock_source.load.return_value = dd
    mock_factory.return_value = mock_source

    with pytest.raises(ValueError, match="no splits to select"):
        ImageDatasetLoader(dataset_config).load_request()


@patch("genai_bench.data.sources.DatasetSourceFactory.create")
def test_streaming_dataset_raises(mock_factory, dataset_config):
    # Create a minimal IterableDataset if available; otherwise skip
    if not hasattr(HFIterableDataset, "from_generator"):
        pytest.skip(
            "IterableDataset.from_generator not available in this datasets version"
        )

    ds = HFIterableDataset.from_generator(lambda: iter([]))
    mock_source = MagicMock()
    mock_source.load.return_value = ds
    mock_factory.return_value = mock_source

    with pytest.raises(ValueError, match="Streaming datasets are not supported"):
        ImageDatasetLoader(dataset_config).load_request()
