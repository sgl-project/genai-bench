import base64
import random
from typing import List, Optional, Tuple

import PIL
from PIL.Image import Image
from six import BytesIO

from genai_bench.logging import init_logger
from genai_bench.protocol import (
    UserImageChatRequest,
    UserImageEmbeddingRequest,
    UserRequest,
)
from genai_bench.sampling.base import Sampler
from genai_bench.scenarios.base import MultiModality, Scenario

logger = init_logger(__name__)


class ImageSampler(Sampler):
    """
    A sampler for image-based tasks, supporting multiple output modalities:
    - `image-to-text`: Generates `UserImageChatRequest` for vision-based chat
      tasks.
    - `image-to-embeddings`: Generates `UserImageEmbeddingRequest` for image
      embedding tasks.
    """

    input_modality = "image"
    supported_tasks = {"image-to-text", "image-to-embeddings"}

    def __init__(
        self,
        tokenizer,
        model: str,
        output_modality: str,
        data: List[Tuple[str, Image]],
        additional_request_params: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(tokenizer, model, output_modality, additional_request_params)
        self.data = data

    def sample(self, scenario: Scenario) -> UserRequest:
        """
        Samples a request based on the scenario.

        Args:
            scenario (Scenario): The scenario to use for sampling.

        Returns:
            UserRequest: A request object for the task.
        """
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
        num_output_tokens: int,
    ) -> UserImageChatRequest:
        """
        Generates a `UserImageChatRequest` for image-to-text tasks.

        Args:
            prompt (str): The textual prompt accompanying the images.
            image_content (List[str]): List of Base64-encoded images.
            num_images (int): Number of images in the request.
            num_output_tokens (int): Number of output tokens expected.

        Returns:
            UserImageChatRequest: A request object for image-to-text tasks.
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
            image_content (List[str]): List of Base64-encoded images.
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

    def _validate_scenario(self, scenario: Optional[Scenario]) -> None:
        """
        Validates that a scenario is provided and has the correct type.

        Raises:
            ValueError: If the scenario is invalid or missing.
        """
        if scenario is None:
            raise ValueError("A scenario is required for image sampling.")
        if not isinstance(scenario.scenario_type, MultiModality):
            raise ValueError(
                f"Expected MultiModality for image tasks, got "
                f"{type(scenario.scenario_type)}"
            )

    def encode_image_base64(cls, image: Image) -> str:
        """Convert image to base64 encoding format."""
        buffered = BytesIO()
        # TODO: why do we need this
        if image.mode == "RGBA":
            image = image.convert("RGB")  # convert to RGB before saving to JPEG
        # Save the image in the buffer in JPEG format
        image.save(buffered, format="JPEG")
        encoded_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return encoded_string

    def _sample_image_and_text(
        self, image_dimension: Tuple[int, int], num_images: int
    ) -> Tuple[str, List[str]]:
        """
        Loads and processes images and accompanying texts from the dataset.

        Args:
            image_dimension (Tuple[int, int]): Dimensions to resize the images.
            num_images (int): Number of images to load.

        Returns:
            Tuple[str, List[str]]: A tuple containing:
                - Texts concatenated as a single prompt.
                - A list of Base64-encoded images.
        """
        selected_data = random.choices(self.data, k=num_images)
        images, texts = [], []

        for data in selected_data:
            image = data[1]
            image = image.resize(image_dimension, PIL.Image.Resampling.LANCZOS)
            images.append(self.encode_image_base64(image))
            texts.append(data[0])
        return " ".join(texts), images
