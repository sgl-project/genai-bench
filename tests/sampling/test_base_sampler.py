from unittest.mock import MagicMock, Mock, patch

import pytest

from genai_bench.sampling.base_sampler import Sampler
from genai_bench.sampling.dataset_loader import (
    DatasetConfig,
    DatasetFormat,
    DatasetPath,
)
from genai_bench.sampling.image_sampler import ImageSampler
from genai_bench.sampling.text_sampler import TextSampler


def test_base_sampler_creation():
    with pytest.raises(TypeError):
        Sampler(tokenizer=Mock(), model="test-model")


@pytest.fixture
def mock_vision_dataset():
    return [
        {"text": "A cat", "image": "cat.jpg"},
        {"text": "A dog", "image": "dog.jpg"},
    ]


@patch("genai_bench.sampling.text_sampler.TextDatasetLoader")
@patch("genai_bench.sampling.image_sampler.ImageDatasetLoader")
def test_sampler_factory(mock_image_loader, mock_text_loader, mock_vision_dataset):
    dataset_config = MagicMock(spec=DatasetConfig)
    dataset_config.dataset_path = MagicMock(spec=DatasetPath)
    dataset_config.dataset_path.path = "sonnet.txt"
    dataset_config.dataset_path.type = DatasetFormat.TEXT

    sampler = Sampler.create(
        "text-to-text", tokenizer=Mock(), model="gpt-3", dataset_config=dataset_config
    )
    assert isinstance(sampler, TextSampler)

    sampler = Sampler.create(
        "text-to-embeddings",
        tokenizer=Mock(),
        model="gpt-3",
        dataset_config=dataset_config,
    )
    assert isinstance(sampler, TextSampler)

    mock_image_loader.load_request.return_value = mock_vision_dataset
    sampler = Sampler.create(
        "image-to-text",
        tokenizer=Mock(),
        model="gpt-3",
        dataset_config=dataset_config,
    )
    assert isinstance(sampler, ImageSampler)

    sampler = Sampler.create(
        "image-to-embeddings",
        tokenizer=Mock(),
        model="gpt-3",
        dataset_config=dataset_config,
    )
    assert isinstance(sampler, ImageSampler)

    with pytest.raises(ValueError):
        Sampler.create("text-to-image", tokenizer=Mock(), model="dummy-model")
