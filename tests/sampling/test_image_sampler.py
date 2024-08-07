from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from genai_bench.protocol import UserImageChatRequest, UserImageEmbeddingRequest
from genai_bench.sampling import ImageModality
from genai_bench.sampling.dataset_loader import (
    DatasetConfig,
    DatasetFormat,
    DatasetPath,
)
from genai_bench.sampling.image_sampler import ImageSampler


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
    with patch(
        "genai_bench.sampling.image_dataset_loader.ImageDatasetLoader.load_request",
        return_value=mock_vision_dataset,
    ):
        dataset_path = DatasetPath()
        dataset_path.path = "huggingface"
        dataset_path.type = DatasetFormat.HUGGINGFACE_HUB
        dataset_config = DatasetConfig(
            dataset_path=dataset_path,
            hf_prompt_column_name="prompt",
            hf_image_column_name="image",
            hf_subset=None,
            hf_split=None,
            hf_revision=None,
            dataset_prompt_column_index=0,
        )
        sampler = ImageSampler(
            tokenizer=mock_tokenizer,
            model="Phi-3-vision-128k-instruct",
            output_modality="text",
            dataset_config=dataset_config,
        )
        scenario = ImageModality(250, 250, 1, 100)

        user_request = sampler.sample(scenario=scenario)

        assert isinstance(user_request, UserImageChatRequest)
        assert user_request.model == "Phi-3-vision-128k-instruct"


def test_image_to_embeddings_sampler(mock_tokenizer, mock_vision_dataset, mock_image):
    with patch(
        "genai_bench.sampling.image_dataset_loader.ImageDatasetLoader.load_request",
        return_value=mock_vision_dataset,
    ):
        dataset_path = DatasetPath()
        dataset_path.path = "huggingface"
        dataset_path.type = DatasetFormat.HUGGINGFACE_HUB
        dataset_config = DatasetConfig(
            dataset_path=dataset_path,
            hf_prompt_column_name="prompt",
            hf_image_column_name="image",
            hf_subset=None,
            hf_split=None,
            hf_revision=None,
            dataset_prompt_column_index=0,
        )
        sampler = ImageSampler(
            tokenizer=mock_tokenizer,
            model="Phi-3-vision-128k-instruct",
            output_modality="embeddings",
            dataset_config=dataset_config,
        )
        scenario = ImageModality(250, 250, 1, 100)

        user_request = sampler.sample(scenario=scenario)

        assert isinstance(user_request, UserImageEmbeddingRequest)
        assert user_request.model == "Phi-3-vision-128k-instruct"


def test_image_sampler_with_multiple_images(
    mock_tokenizer, mock_vision_dataset, mock_image
):
    with patch(
        "genai_bench.sampling.image_dataset_loader.ImageDatasetLoader.load_request",
        return_value=mock_vision_dataset * 2,
    ):
        dataset_path = DatasetPath()
        dataset_path.path = "huggingface"
        dataset_path.type = DatasetFormat.HUGGINGFACE_HUB
        dataset_config = DatasetConfig(
            dataset_path=dataset_path,
            hf_prompt_column_name="prompt",
            hf_image_column_name="image",
            hf_subset=None,
            hf_split=None,
            hf_revision=None,
            dataset_prompt_column_index=0,
        )
        sampler = ImageSampler(
            tokenizer=mock_tokenizer,
            model="Phi-3-vision-128k-instruct",
            output_modality="text",
            dataset_config=dataset_config,
        )
        scenario = ImageModality(250, 250, 2, 100)

        user_request = sampler.sample(scenario=scenario)

        assert isinstance(user_request, UserImageChatRequest)
        assert user_request.num_images == 2


def test_image_sampler_with_invalid_scenario(mock_tokenizer, mock_vision_dataset):
    with patch(
        "genai_bench.sampling.image_dataset_loader.ImageDatasetLoader.load_request",
        return_value=mock_vision_dataset,
    ):
        dataset_path = DatasetPath()
        dataset_path.path = "huggingface"
        dataset_path.type = DatasetFormat.HUGGINGFACE_HUB
        dataset_config = DatasetConfig(
            dataset_path=dataset_path,
            hf_prompt_column_name="prompt",
            hf_image_column_name="image",
            hf_subset=None,
            hf_split=None,
            hf_revision=None,
            dataset_prompt_column_index=0,
        )
        sampler = ImageSampler(
            tokenizer=mock_tokenizer,
            model="Phi-3-vision-128k-instruct",
            output_modality="text",
            dataset_config=dataset_config,
        )
        mock_scenario = MagicMock()
        mock_scenario.scenario_type = "InvalidType"

        with pytest.raises(
            ValueError,
            match="Expected MultiModality for image tasks, got <class 'str'>",
        ):
            sampler.sample(mock_scenario)

        with pytest.raises(
            ValueError,
            match="A scenario is required for image sampling",
        ):
            sampler.sample(None)
