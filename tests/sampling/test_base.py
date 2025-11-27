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
    test_text_tokenizer = Mock()
    # Mock tokenizer's get_vocab to some tokens with special tokens
    test_text_tokenizer.get_vocab.return_value = {
        "token1": 0,
        "token2": 1,
        "token3": 2,
        "<special>": 3,
        "<pad>": 4,
        "token4": 5,
    }

    sampler = Sampler.create(
        task="text-to-text",
        tokenizer=test_text_tokenizer,
        model="gpt-3",
        data=text_data,
    )
    assert isinstance(sampler, TextSampler)

    sampler = Sampler.create(
        task="text-to-embeddings",
        tokenizer=test_text_tokenizer,
        model="gpt-3",
        data=text_data,
    )
    assert isinstance(sampler, TextSampler)

    sampler = Sampler.create(
        task="image-text-to-text",
        tokenizer=test_text_tokenizer,
        model="gpt-3",
        data=mock_vision_dataset,
    )
    assert isinstance(sampler, ImageSampler)

    sampler = Sampler.create(
        task="image-to-embeddings",
        tokenizer=Mock(),
        model="gpt-3",
        data=mock_vision_dataset,
    )
    assert isinstance(sampler, ImageSampler)

    with pytest.raises(ValueError):
        Sampler.create(
            task="text-to-image",
            tokenizer=test_text_tokenizer,
            model="dummy-model",
            data=[],
        )
