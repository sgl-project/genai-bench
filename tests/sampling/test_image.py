from unittest.mock import MagicMock

import pytest
from PIL import Image

from genai_bench.data.config import DatasetConfig, DatasetSourceConfig
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


def test_image_sampler_dict_rows_prompt_column(monkeypatch, mock_tokenizer):
    img = Image.new("RGB", (64, 64))
    data = [
        {"image_column": img, "prompt_column": "prompt_1"},
        {"image_column": img, "prompt_column": "prompt_2"},
    ]
    ds_cfg = DatasetConfig(
        source=DatasetSourceConfig(type="huggingface", path="test/dataset"),
        prompt_column="prompt_column",
        image_column="image_column",
    )
    # Force deterministic sampling
    monkeypatch.setattr("random.choices", lambda population, k: [population[0]])

    sampler = ImageSampler(
        tokenizer=mock_tokenizer,
        model="phi-vision",
        output_modality="text",
        data=data,
        dataset_config=ds_cfg,
    )
    req = sampler.sample(scenario=None)
    assert isinstance(req, UserImageChatRequest)
    assert req.prompt == "prompt_1"
    assert len(req.image_content) == 1
    assert req.image_content[0].startswith("data:image/jpeg;base64,")


def test_image_sampler_dict_rows_prompt_lambda(monkeypatch, mock_tokenizer):
    img = Image.new("RGB", (64, 64))
    data = [{"image_column": img, "anything": "ignored"}]
    ds_cfg = DatasetConfig(
        source=DatasetSourceConfig(type="huggingface", path="test/dataset"),
        prompt_lambda='lambda x: "Fixed prompt for all"',
        image_column="image_column",
    )
    monkeypatch.setattr("random.choices", lambda population, k: [population[0]])

    sampler = ImageSampler(
        tokenizer=mock_tokenizer,
        model="phi-vision",
        output_modality="text",
        data=data,
        dataset_config=ds_cfg,
    )
    req = sampler.sample(scenario=None)
    assert isinstance(req, UserImageChatRequest)
    assert req.prompt == "Fixed prompt for all"
    assert req.image_content[0].startswith("data:image/jpeg;base64,")


def test_image_sampler_dict_rows_url_images(monkeypatch, mock_tokenizer):
    data = [{"image_column": "https://example.com/a.jpg", "prompt_column": "p1"}]
    ds_cfg = DatasetConfig(
        source=DatasetSourceConfig(type="huggingface", path="test/dataset"),
        prompt_column="prompt_column",
        image_column="image_column",
    )
    monkeypatch.setattr("random.choices", lambda population, k: [population[0]])

    sampler = ImageSampler(
        tokenizer=mock_tokenizer,
        model="phi-vision",
        output_modality="text",
        data=data,
        dataset_config=ds_cfg,
    )
    req = sampler.sample(scenario=None)
    assert isinstance(req, UserImageChatRequest)
    assert req.prompt == "p1"
    assert req.image_content == ["https://example.com/a.jpg"]


def test_image_sampler_missing_prompt_column(monkeypatch, mock_tokenizer):
    img = Image.new("RGB", (64, 64))
    data = [{"image_column": img, "other": "x"}]
    ds_cfg = DatasetConfig(
        source=DatasetSourceConfig(type="huggingface", path="test/dataset"),
        prompt_column="nonexistent_column",
        image_column="image_column",
    )
    monkeypatch.setattr("random.choices", lambda population, k: [population[0]])

    sampler = ImageSampler(
        tokenizer=mock_tokenizer,
        model="phi-vision",
        output_modality="text",
        data=data,
        dataset_config=ds_cfg,
    )
    req = sampler.sample(scenario=None)
    assert isinstance(req, UserImageChatRequest)
    assert req.prompt == ""
    assert req.image_content[0].startswith("data:image/jpeg;base64,")
