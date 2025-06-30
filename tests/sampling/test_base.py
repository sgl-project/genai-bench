from unittest.mock import Mock

import pytest

from genai_bench.sampling.base import Sampler
from genai_bench.sampling.image import ImageSampler
from genai_bench.sampling.text import TextSampler


def test_base_sampler_creation():
    with pytest.raises(TypeError):
        Sampler(tokenizer=Mock(), model="test-model")


@pytest.fixture
def mock_vision_dataset():
    return [
        {"text": "A cat", "image": "cat.jpg"},
        {"text": "A dog", "image": "dog.jpg"},
    ]


def test_sampler_factory(mock_vision_dataset):
    text_data = ["Sample text 1", "Sample text 2"]

    # Create a tokenizer mock that returns a list of token ids so len() works
    tokenizer = Mock()
    tokenizer.encode.return_value = [1, 2, 3]

    sampler = Sampler.create(
        task="text-to-text",
        tokenizer=tokenizer,
        model="gpt-3",
        data=text_data,
        use_scenario=True,
    )
    assert isinstance(sampler, TextSampler)

    sampler = Sampler.create(
        task="text-to-embeddings",
        tokenizer=tokenizer,
        model="gpt-3",
        data=text_data,
        use_scenario=True,
    )
    assert isinstance(sampler, TextSampler)

    sampler = Sampler.create(
        task="image-text-to-text",
        tokenizer=tokenizer,
        model="gpt-3",
        data=mock_vision_dataset,
        use_scenario=False,
    )
    assert isinstance(sampler, ImageSampler)

    sampler = Sampler.create(
        task="image-to-embeddings",
        tokenizer=tokenizer,
        model="gpt-3",
        data=mock_vision_dataset,
    )
    assert isinstance(sampler, ImageSampler)

    with pytest.raises(ValueError):
        Sampler.create(
            task="text-to-image", tokenizer=Mock(), model="dummy-model", data=[]
        )
