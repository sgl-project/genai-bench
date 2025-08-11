import base64
import random
from typing import Any, List, Optional, Tuple

import PIL
from PIL.Image import Image
from six import BytesIO

from genai_bench.data.config import DatasetConfig
from genai_bench.logging import init_logger
from genai_bench.protocol import (
    UserImageChatRequest,
    UserImageEmbeddingRequest,
    UserRequest,
)
from genai_bench.sampling.base import Sampler
from genai_bench.scenarios.base import MultiModality, Scenario
from genai_bench.utils import safe_eval_prompt

logger = init_logger(__name__)


class ImageSampler(Sampler):
    """
    A sampler for image-based tasks, supporting multiple output modalities:
    - `image-text-to-text`: Generates `UserImageChatRequest` for vision-based chat
      tasks.
    - `image-to-embeddings`: Generates `UserImageEmbeddingRequest` for image
      embedding tasks.
    """

    input_modality = "image"
    supported_tasks = {"image-text-to-text", "image-to-embeddings"}

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
        """
        Samples a request based on the scenario or dataset configuration.

        Args:
            scenario (Scenario, optional): The scenario to use for sampling.
                If None, uses dataset configuration directly.

        Returns:
            UserRequest: A request object for the task.
        """
        # Dataset mode when scenario is dataset or None
        if self._is_dataset_mode(scenario):
            image_dimension, num_images, num_output_tokens = None, 1, None
        else:
            self._validate_scenario(scenario)
            image_dimension, num_images, num_output_tokens = scenario.sample()
        prompt, image_content = self._sample_image_and_text(image_dimension, num_images)

        # TODO: create Delegated Request Creator to replace if-else
        if self.output_modality == "text":
            return self._generate_image_chat_request(
                prompt, image_content, num_images, num_output_tokens
            )
        elif self.output_modality == "embeddings":
            return self._generate_image_embedding_request(image_content, num_images)
        else:
            raise ValueError(f"Unsupported output modality: {self.output_modality}")

    def _generate_image_chat_request(
        self,
        prompt: str,
        image_content: List[str],
        num_images: int,
        num_output_tokens: int | None,
    ) -> UserImageChatRequest:
        """
        Generates a `UserImageChatRequest` for image-text-to-text tasks.

        Args:
            prompt (str): The textual prompt accompanying the images.
            image_content (List[str]): List of image URLs (base64 data or http URLs).
            num_images (int): Number of images in the request.
            num_output_tokens (int): Number of output tokens expected.

        Returns:
            UserImageChatRequest: A request object for image-text-to-text tasks.
        """
        return UserImageChatRequest(
            model=self.model,
            prompt=prompt,
            image_content=image_content,
            num_images=num_images,
            max_tokens=num_output_tokens,
            num_prefill_tokens=None,
            additional_request_params=self.additional_request_params,
        )

    def _generate_image_embedding_request(
        self, image_content: List[str], num_images: int
    ) -> UserImageEmbeddingRequest:
        """
        Generates a `UserImageEmbeddingRequest` for image-to-embedding tasks.

        Args:
            image_content (List[str]): List of image URLs (base64 data or http URLs).
            num_images (int): Number of images in the request.

        Returns:
            UserImageEmbeddingRequest: A request object for
                image-to-embedding tasks.
        """
        return UserImageEmbeddingRequest(
            model=self.model,
            documents=[],  # empty documents placeholder
            image_content=image_content,
            num_images=num_images,
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
                f"Expected MultiModality for image tasks, got "
                f"{type(scenario.scenario_type)}"
            )

    def _sample_image_and_text(
        self, image_dimension: Optional[Tuple[int, int]] = None, num_images: int = 1
    ) -> Tuple[str, List[str]]:
        """
        Lazily sample and process images and accompanying texts from the dataset.

        Supports two input shapes:
        - Sequence of (prompt, image) tuples (backward compatible)
        - Sequence of dict rows (e.g., HF Dataset rows) using dataset_config
        """
        images: List[str] = []
        texts: List[str] = []

        chosen = random.choices(self.data, k=num_images)
        for item in chosen:
            prompt: str = ""
            raw_image: Any = None
            # Backward-compatible format
            if isinstance(item, tuple) and len(item) == 2:
                prompt, raw_image = item
            # Dict row format
            elif isinstance(item, dict) and self.dataset_config is not None:
                cfg = self.dataset_config
                if cfg.image_column:
                    raw_image = item.get(cfg.image_column)
                if cfg.prompt_lambda:
                    prompt = safe_eval_prompt(cfg.prompt_lambda, item)
                elif cfg.prompt_column:
                    prompt = str(item.get(cfg.prompt_column, ""))
            else:
                continue

            if raw_image is None:
                continue
            processed_image = ImageSampler.process_image(
                raw_image, resize=image_dimension
            )
            images.append(processed_image)
            texts.append(prompt or "")

        return " ".join(texts), images

    @staticmethod
    def process_image(image: Any, resize: Optional[Tuple[int, int]] = None) -> str:
        """
        Process a single image input and return a data URL or HTTP(S) URL.

        Supports three input types:
        1. Dictionary with raw image bytes
        2. PIL.Image.Image input
        3. String input (URL or file path)
        """
        if isinstance(image, dict) and "bytes" in image:
            image = PIL.Image.open(BytesIO(image["bytes"]))

        if isinstance(image, Image):
            if image.mode != "RGB":
                image = image.convert("RGB")
            if resize:
                image = image.resize(resize, PIL.Image.Resampling.LANCZOS)
            with BytesIO() as image_data:
                image.save(image_data, format="JPEG")
                image_base64 = base64.b64encode(image_data.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{image_base64}"

        if isinstance(image, str) and image.startswith(("http://", "https://")):
            return image

        raise ValueError(
            f"Invalid image input {image}. Must be a PIL.Image.Image"
            " or str or dictionary with raw image bytes."
        )
