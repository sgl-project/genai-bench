import pytest

from genai_bench.data.config import DatasetConfig, DatasetSourceConfig
from genai_bench.data.loaders.base import DatasetFormat, DatasetLoader


def test_default_source_config():
    config = DatasetSourceConfig(type="file", path="test.txt", file_format="txt")
    assert config.type == "file"
    assert config.path == "test.txt"


def test_file_source_config():
    config = DatasetSourceConfig(
        type="file", path="/path/to/file.txt", file_format="txt"
    )
    assert config.path == "/path/to/file.txt"
    assert config.type == "file"


def test_huggingface_source_config():
    config = DatasetSourceConfig(type="huggingface", path="huggingface/dataset")
    assert config.path == "huggingface/dataset"
    assert config.type == "huggingface"


def test_invalid_source_config():
    with pytest.raises(ValueError):
        DatasetSourceConfig(type="invalid", path="some/path")


def test_load_request():
    """Test that DatasetLoader validates file format support"""

    # Create a mock loader class that has the required attributes
    class MockDatasetLoader(DatasetLoader):
        media_type = "Image"
        supported_formats = {DatasetFormat.HUGGINGFACE_HUB}

        def _process_loaded_data(self, data):
            return []

    config = DatasetConfig(
        source=DatasetSourceConfig(type="file", path="test.txt", file_format="txt")
    )

    with pytest.raises(
        ValueError, match="File format 'txt' is not supported by Image loader"
    ):
        MockDatasetLoader(config)
