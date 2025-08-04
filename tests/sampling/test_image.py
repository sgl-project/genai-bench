from unittest.mock import MagicMock

import pytest
from PIL import Image

from genai_bench.protocol import UserImageChatRequest, UserImageEmbeddingRequest
from genai_bench.sampling.image import ImageSampler
from genai_bench.scenarios import ImageModality


@pytest.fixture
def mock_tokenizer():
    return MagicMock()


@pytest.fixture
def mock_vision_dataset():
    return [("A cat", Image.new("RGBA", (250, 250)))]


@pytest.fixture
def mock_image():
    mock_img = Image.new("RGB", (2048, 2048))
    return mock_img


def test_image_sampler(mock_tokenizer, mock_vision_dataset, mock_image):
    sampler = ImageSampler(
        tokenizer=mock_tokenizer,
        model="Phi-3-vision-128k-instruct",
        output_modality="text",
        data=mock_vision_dataset,
    )
    scenario = ImageModality(250, 250, 1, 100)

    user_request = sampler.sample(scenario=scenario)

    assert isinstance(user_request, UserImageChatRequest)
    assert user_request.model == "Phi-3-vision-128k-instruct"


def test_image_to_embeddings_sampler(mock_tokenizer, mock_vision_dataset, mock_image):
    sampler = ImageSampler(
        tokenizer=mock_tokenizer,
        model="Phi-3-vision-128k-instruct",
        output_modality="embeddings",
        data=mock_vision_dataset,
    )
    scenario = ImageModality(250, 250, 1, 100)

    user_request = sampler.sample(scenario=scenario)

    assert isinstance(user_request, UserImageEmbeddingRequest)
    assert user_request.model == "Phi-3-vision-128k-instruct"


def test_image_sampler_with_multiple_images(
    mock_tokenizer, mock_vision_dataset, mock_image
):
    sampler = ImageSampler(
        tokenizer=mock_tokenizer,
        model="Phi-3-vision-128k-instruct",
        output_modality="text",
        data=mock_vision_dataset * 2,
    )
    scenario = ImageModality(250, 250, 2, 100)

    user_request = sampler.sample(scenario=scenario)

    assert isinstance(user_request, UserImageChatRequest)
    assert user_request.num_images == 2


def test_image_sampler_with_invalid_scenario(mock_tokenizer, mock_vision_dataset):
    sampler = ImageSampler(
        tokenizer=mock_tokenizer,
        model="Phi-3-vision-128k-instruct",
        output_modality="text",
        data=mock_vision_dataset,
    )
    mock_scenario = MagicMock()
    mock_scenario.scenario_type = "InvalidType"

    with pytest.raises(
        ValueError,
        match="Expected MultiModality for image tasks, got <class 'str'>",
    ):
        sampler.sample(mock_scenario)

    # None scenario is now supported (dataset-only mode)
    request = sampler.sample(None)
    assert request is not None
