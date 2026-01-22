"""Video sampler for video-text-to-text tasks."""

import random
from typing import Any, List, Optional, Tuple

from genai_bench.data.config import DatasetConfig
from genai_bench.logging import init_logger
from genai_bench.protocol import UserRequest, UserVideoChatRequest
from genai_bench.sampling.base import Sampler
from genai_bench.scenarios.base import MultiModality, Scenario
from genai_bench.utils import safe_eval_prompt

logger = init_logger(__name__)


class VideoSampler(Sampler):
    """
    A sampler for video-based tasks, supporting video-text-to-text one task types:
    - video-text-to-text
    """

    input_modality = "video"
    supported_tasks = {"video-text-to-text"}

    def __init__(
        self,
        tokenizer,
        model: str,
        output_modality: str,
        data: Any,
        dataset_config: Optional[DatasetConfig] = None,
        additional_request_params: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(
            tokenizer,
            model,
            output_modality,
            additional_request_params,
            dataset_config=dataset_config,
        )
        self.data = data

    def sample(self, scenario: Optional[Scenario]) -> UserRequest:
        """Samples a request based on the scenario or dataset configuration.

        Args:
            scenario (Scenario, optional): The scenario to use for sampling.
                If None, uses dataset configuration directly.

        Returns:
            UserRequest: A request object for the video-text-to-text task.
        """
        # Dataset mode when scenario is dataset or None
        if self._is_dataset_mode(scenario):
            num_videos, num_output_tokens = 1, None
        else:
            self._validate_scenario(scenario)
            # VideoModality.sample() returns (num_videos, max_tokens)
            num_videos, num_output_tokens = scenario.sample()

        prompt, video_content = self._sample_video_and_text(num_videos)

        return self._generate_video_chat_request(
            prompt, video_content, num_output_tokens
        )

    def _generate_video_chat_request(
        self,
        prompt: str,
        video_content: List[str],
        num_output_tokens: Optional[int],
    ) -> UserVideoChatRequest:
        """Generate a `UserVideoChatRequest` for video-text-to-text tasks.

        Args:
            prompt: The textual prompt accompanying the videos.
            video_content: List of video URLs.
            num_output_tokens: Number of output tokens expected.

        Returns:
            UserVideoChatRequest: A request object for video-text-to-text tasks.
        """
        return UserVideoChatRequest(
            model=self.model,
            prompt=prompt,
            video_content=video_content,
            max_tokens=num_output_tokens,
            num_prefill_tokens=None,
            additional_request_params=self.additional_request_params,
        )

    def _validate_scenario(self, scenario: Scenario) -> None:
        """
        Validates that a scenario has the correct type.

        Raises:
            ValueError: If the scenario is invalid.
        """
        if not isinstance(scenario.scenario_type, MultiModality):
            raise ValueError(
                f"Expected MultiModality for video tasks, got "
                f"{type(scenario.scenario_type)}"
            )

    def _sample_video_and_text(
        self,
        num_videos: int = 1,
    ) -> Tuple[str, List[str]]:
        """Sample one or more video URLs and accompanying text.

        Supports two input shapes:
        - Sequence of (prompt, video_url) tuples
        - Sequence of dict rows (e.g., HF Dataset rows) using dataset_config

        Args:
            num_videos: Number of videos to sample.

        Returns:
            Tuple of (combined_prompt_text, list_of_video_urls).

        Raises:
            ValueError: If no valid video item can be sampled from the dataset.
        """
        videos: List[str] = []
        texts: List[str] = []

        chosen = random.choices(self.data, k=num_videos)
        for item in chosen:
            prompt: str = ""
            video_url: Optional[str] = None

            # Backward-compatible format: (prompt, video_url)
            if isinstance(item, tuple) and len(item) == 2:
                prompt, video_url = item
            # Dict row format (HuggingFace Dataset)
            elif isinstance(item, dict) and self.dataset_config is not None:
                cfg = self.dataset_config
                if cfg.video_column:
                    video_url = item.get(cfg.video_column)
                if cfg.prompt_lambda:
                    prompt = safe_eval_prompt(cfg.prompt_lambda, item)
                elif cfg.prompt_column:
                    prompt = str(item.get(cfg.prompt_column, ""))
            else:
                continue

            if video_url is None:
                continue

            # Validate it's a string and normalize to URL / base64 data URL
            if not isinstance(video_url, str):
                logger.warning(
                    f"Expected video URL string, got {type(video_url)}. Skipping."
                )
                continue

            video_url = video_url.strip()
            if not video_url:
                continue

            if video_url.startswith(("http://", "https://", "data:")):
                normalized_video_url = video_url
            else:
                # Treat as raw base64 bytes without schema
                normalized_video_url = f"data:video/mp4;base64,{video_url}"

            videos.append(normalized_video_url)
            texts.append(prompt or "")
        if not videos:
            raise ValueError("No valid video URL found in dataset to sample from.")

        return " ".join(texts), videos


