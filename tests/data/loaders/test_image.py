from unittest.mock import MagicMock, patch

import pytest

from genai_bench.data.config import DatasetConfig, DatasetSourceConfig
from genai_bench.data.loaders.image import ImageDatasetLoader


@pytest.fixture
def mock_dataset():
    # Create mock PIL Images instead of strings

    from PIL import Image as PILImage

    # Create fake image data
    mock_image_1 = PILImage.new("RGB", (100, 100), color="red")
    mock_image_2 = PILImage.new("RGB", (100, 100), color="blue")

    mock_dataset = MagicMock()
    mock_dataset.__len__.return_value = 2
    mock_dataset.features = ["image_column", "prompt_column"]
    mock_dataset.__iter__.return_value = iter(
        [
            {"image_column": mock_image_1, "prompt_column": "prompt_1"},
            {"image_column": mock_image_2, "prompt_column": "prompt_2"},
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

    # Check we got 2 results with correct prompts and PIL Images
    assert len(results) == 2
    assert results[0][0] == "prompt_1"
    assert results[1][0] == "prompt_2"
    # Verify images are PIL Images
    from PIL import Image as PILImage

    assert isinstance(results[0][1], PILImage.Image)
    assert isinstance(results[1][1], PILImage.Image)


@patch("genai_bench.data.sources.DatasetSourceFactory.create")
def test_load_requests_missing_prompt(mock_factory, mock_dataset, dataset_config):
    """Test if load_requests handles missing prompt column"""
    from PIL import Image as PILImage

    mock_image_1 = PILImage.new("RGB", (100, 100), color="red")
    mock_image_2 = PILImage.new("RGB", (100, 100), color="blue")

    mock_dataset.__iter__.return_value = [
        {"image_column": mock_image_1},
        {"image_column": mock_image_2},
    ]
    mock_source = MagicMock()
    mock_source.load.return_value = mock_dataset
    mock_factory.return_value = mock_source
    dataset_config.prompt_column = None

    results = ImageDatasetLoader(dataset_config).load_request()

    # Check results have empty prompts and PIL Images
    assert len(results) == 2
    assert results[0][0] == ""
    assert results[1][0] == ""
    assert isinstance(results[0][1], PILImage.Image)
    assert isinstance(results[1][1], PILImage.Image)


@patch("genai_bench.data.sources.DatasetSourceFactory.create")
def test_load_requests_empty_dataset(mock_factory, mock_empty_dataset, dataset_config):
    """Test if load_requests handles empty dataset gracefully"""
    mock_source = MagicMock()
    mock_source.load.return_value = mock_empty_dataset
    mock_factory.return_value = mock_source

    results = ImageDatasetLoader(dataset_config).load_request()
    assert results == []


@patch("genai_bench.data.sources.DatasetSourceFactory.create")
def test_load_requests_override_prompt(mock_factory, mock_dataset, dataset_config):
    """Test if load_requests uses prompt_lambda for fixed prompt"""
    mock_source = MagicMock()
    mock_source.load.return_value = mock_dataset
    mock_factory.return_value = mock_source
    dataset_config.prompt_lambda = 'lambda x: "Fixed prompt for all"'

    results = ImageDatasetLoader(dataset_config).load_request()

    # Check fixed prompt is used for all images
    assert len(results) == 2
    assert results[0][0] == "Fixed prompt for all"
    assert results[1][0] == "Fixed prompt for all"
    from PIL import Image as PILImage

    assert isinstance(results[0][1], PILImage.Image)
    assert isinstance(results[1][1], PILImage.Image)


@patch("genai_bench.data.sources.DatasetSourceFactory.create")
def test_load_requests_missing_prompt_column_in_data(mock_factory, dataset_config):
    """Test if load_requests fails fast when prompt column doesn't exist in data"""
    from PIL import Image as PILImage

    mock_image_1 = PILImage.new("RGB", (100, 100), color="red")
    mock_image_2 = PILImage.new("RGB", (100, 100), color="blue")

    # Dataset items don't have the prompt column
    mock_dataset = MagicMock()
    mock_dataset.__iter__.return_value = iter(
        [
            {"image_column": mock_image_1, "other_column": "other_1"},
            {"image_column": mock_image_2, "other_column": "other_2"},
        ]
    )
    mock_source = MagicMock()
    mock_source.load.return_value = mock_dataset
    mock_factory.return_value = mock_source

    # Config specifies a prompt column that doesn't exist
    dataset_config.prompt_column = "nonexistent_column"

    # Should raise ValueError for missing column (fail fast)
    with pytest.raises(ValueError, match="Cannot extract image data from dataset"):
        ImageDatasetLoader(dataset_config).load_request()


@patch("genai_bench.data.sources.DatasetSourceFactory.create")
@patch("requests.get")
@patch("PIL.Image.open")
def test_load_requests_url_images(
    mock_pil_open, mock_requests_get, mock_factory, dataset_config
):
    """Test if load_requests handles URL images"""
    # Mock dataset with URL strings
    mock_dataset = MagicMock()
    mock_dataset.__iter__.return_value = iter(
        [
            {
                "image_column": "https://example.com/image1.jpg",
                "prompt_column": "prompt_1",
            },
            {
                "image_column": "https://example.com/image2.jpg",
                "prompt_column": "prompt_2",
            },
        ]
    )
    mock_source = MagicMock()
    mock_source.load.return_value = mock_dataset
    mock_factory.return_value = mock_source

    results = ImageDatasetLoader(dataset_config).load_request()

    # URLs are now passed through directly, not fetched
    assert len(results) == 2
    assert results[0][0] == "prompt_1"
    assert results[0][1] == "https://example.com/image1.jpg"
    assert results[1][0] == "prompt_2"
    assert results[1][1] == "https://example.com/image2.jpg"

    # No requests should be made during loading
    assert mock_requests_get.call_count == 0
    assert mock_pil_open.call_count == 0
