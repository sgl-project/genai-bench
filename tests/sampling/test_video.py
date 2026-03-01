from unittest.mock import MagicMock

import pytest

from genai_bench.data.config import DatasetConfig, DatasetSourceConfig
from genai_bench.protocol import UserVideoChatRequest
from genai_bench.sampling.base import Sampler
from genai_bench.sampling.video import VideoSampler
from genai_bench.scenarios import VideoModality


@pytest.fixture
def mock_tokenizer():
    return MagicMock()


@pytest.fixture
def mock_video_dataset():
    # Simple (prompt, video) tuples with HTTP URLs so no real processing is needed
    return [
        ("Describe this video", "https://example.com/video1.mp4"),
        ("Summarize the action", "https://example.com/video2.mp4"),
    ]


def test_video_sampler_with_scenario(mock_tokenizer, mock_video_dataset):
    """Basic unit test: VideoSampler + VideoModality end-to-end for one request."""
    sampler = VideoSampler(
        tokenizer=mock_tokenizer,
        model="qwen3-omni-flash",
        output_modality="text",
        data=mock_video_dataset,
    )
    scenario = VideoModality(
        num_input_videos=1,
        max_output_token=128,
    )

    req = sampler.sample(scenario=scenario)

    assert isinstance(req, UserVideoChatRequest)
    assert req.model == "qwen3-omni-flash"
    assert isinstance(req.prompt, str)
    assert isinstance(req.video_content, list)
    assert len(req.video_content) == 1
    assert req.video_content[0].startswith("http") or req.video_content[0].startswith(
        "data:video/"
    )
    assert req.max_tokens == 128


def test_video_sampler_dataset_mode_tuple_data(
    monkeypatch, mock_tokenizer, mock_video_dataset
):
    """Dataset mode with tuple data should work when scenario is None."""
    # Make sampling deterministic
    monkeypatch.setattr("random.choices", lambda population, k: [population[0]])

    sampler = VideoSampler(
        tokenizer=mock_tokenizer,
        model="qwen3-omni-flash",
        output_modality="text",
        data=mock_video_dataset,
    )

    req = sampler.sample(scenario=None)

    assert isinstance(req, UserVideoChatRequest)
    assert req.prompt == "Describe this video"
    assert isinstance(req.video_content, list)
    assert req.video_content == ["https://example.com/video1.mp4"]
    # In dataset mode, max_tokens should be None
    assert req.max_tokens is None


def test_video_sampler_invalid_scenario(mock_tokenizer, mock_video_dataset):
    """VideoSampler should reject non-multimodal scenarios."""
    sampler = VideoSampler(
        tokenizer=mock_tokenizer,
        model="qwen3-omni-flash",
        output_modality="text",
        data=mock_video_dataset,
    )
    bad_scenario = MagicMock()
    bad_scenario.scenario_type = "InvalidType"

    with pytest.raises(
        ValueError,
        match="Expected MultiModality for video tasks, got <class 'str'>",
    ):
        sampler.sample(bad_scenario)


@pytest.mark.skip(
    reason="Video modality not registered in Sampler.modality_registry yet"
)
def test_video_sampler_factory_and_dataset_integration(monkeypatch, mock_tokenizer):
    """Integration-style test: Sampler.create + DatasetConfig + VideoSampler."""
    data = [
        {
            "video_column": "https://example.com/video1.mp4",
            "prompt_column": "prompt_1",
        },
        {
            "video_column": "https://example.com/video2.mp4",
            "prompt_column": "prompt_2",
        },
    ]
    ds_cfg = DatasetConfig(
        source=DatasetSourceConfig(type="huggingface", path="dummy/path"),
        prompt_column="prompt_column",
        image_column=None,
        video_column="video_column",
    )
    # Deterministic sampling
    monkeypatch.setattr("random.choices", lambda population, k: [population[0]])

    sampler = Sampler.create(
        task="video-text-to-text",
        tokenizer=mock_tokenizer,
        model="qwen3-omni-flash",
        data=data,
        dataset_config=ds_cfg,
    )

    assert isinstance(sampler, VideoSampler)

    req = sampler.sample(scenario=None)

    assert isinstance(req, UserVideoChatRequest)
    assert req.model == "qwen3-omni-flash"
    assert req.prompt == "prompt_1"
    assert isinstance(req.video_content, list)
    assert req.video_content == ["https://example.com/video1.mp4"]
    assert req.max_tokens is None


def test_video_sampler_dataset_mode_hf_base64_column(monkeypatch, mock_tokenizer):
    """Dataset mode: HF-style dict rows with pure base64 video column."""
    data = [
        {
            "video_bytes": "AAAABASE64VIDEO",  # pure base64 without data: prefix
            "prompt": "Describe this video",
        }
    ]
    ds_cfg = DatasetConfig(
        source=DatasetSourceConfig(type="huggingface", path="dummy/path"),
        prompt_column="prompt",
        image_column=None,
        video_column="video_bytes",
    )

    # Deterministic sampling
    monkeypatch.setattr("random.choices", lambda population, k: [population[0]])

    sampler = VideoSampler(
        tokenizer=mock_tokenizer,
        model="qwen3-omni-flash",
        output_modality="text",
        data=data,
        dataset_config=ds_cfg,
    )

    req = sampler.sample(scenario=None)

    assert isinstance(req, UserVideoChatRequest)
    assert req.prompt == "Describe this video"
    assert req.video_content == [
        "data:video/mp4;base64,AAAABASE64VIDEO",
    ]


def test_video_sampler_dataset_mode_hf_data_url_preserved(monkeypatch, mock_tokenizer):
    """Dataset mode: values already data URLs should be preserved as-is."""
    data = [
        {
            "video_bytes": "data:video/mp4;base64,AAAABASE64VIDEO",
            "prompt": "Describe this video",
        }
    ]
    ds_cfg = DatasetConfig(
        source=DatasetSourceConfig(type="huggingface", path="dummy/path"),
        prompt_column="prompt",
        image_column=None,
        video_column="video_bytes",
    )

    monkeypatch.setattr("random.choices", lambda population, k: [population[0]])

    sampler = VideoSampler(
        tokenizer=mock_tokenizer,
        model="qwen3-omni-flash",
        output_modality="text",
        data=data,
        dataset_config=ds_cfg,
    )

    req = sampler.sample(scenario=None)

    assert isinstance(req, UserVideoChatRequest)
    assert req.prompt == "Describe this video"
    assert req.video_content == [
        "data:video/mp4;base64,AAAABASE64VIDEO",
    ]


def test_video_sampler_dataset_mode_invalid_video_type_raises(
    monkeypatch, mock_tokenizer
):
    """Dataset mode: non-string video value should trigger a clear error."""
    data = [
        {
            "video_bytes": 12345,
            "prompt": "Describe this video",
        }
    ]
    ds_cfg = DatasetConfig(
        source=DatasetSourceConfig(type="huggingface", path="dummy/path"),
        prompt_column="prompt",
        image_column=None,
        video_column="video_bytes",
    )

    monkeypatch.setattr("random.choices", lambda population, k: [population[0]])

    sampler = VideoSampler(
        tokenizer=mock_tokenizer,
        model="qwen3-omni-flash",
        output_modality="text",
        data=data,
        dataset_config=ds_cfg,
    )

    with pytest.raises(
        ValueError,
        match="No valid video URL found in dataset to sample from.",
    ):
        sampler.sample(scenario=None)
