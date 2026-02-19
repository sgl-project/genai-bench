import random
from typing import Any, Dict, List, Optional

from genai_bench.data.config import DatasetConfig
from genai_bench.logging import init_logger
from genai_bench.protocol import (
    UserChatRequest,
    UserEmbeddingRequest,
    UserImageGenerationRequest,
    UserRequest,
    UserReRankRequest,
)
from genai_bench.sampling.base import Sampler
from genai_bench.scenarios.base import (
    EmbeddingDistribution,
    MultiModality,
    Scenario,
    TextDistribution,
)

logger = init_logger(__name__)


class TextSampler(Sampler):
    """
    Unified sampler for text-based tasks, supporting multiple task types:
    - `text-to-text`: Standard chat or generation tasks.
    - `text-to-embeddings`: Embedding generation from text.
    """

    input_modality = "text"
    supported_tasks = {
        "text-to-text",
        "text-to-embeddings",
        "text-to-rerank",
        "text-to-image",
    }

    def __init__(
        self,
        tokenizer,
        model: str,
        output_modality: str,
        data: List[str],
        additional_request_params: Optional[Dict[str, Any]] = None,
        dataset_config: Optional[DatasetConfig] = None,
        **kwargs,
    ):
        super().__init__(
            tokenizer, model, output_modality, additional_request_params, dataset_config
        )

        self.data = data
        self.batch_size = 1  # Default batch size

    def sample(self, scenario: Optional[Scenario]) -> UserRequest:
        """
        Samples a request based on the scenario.

        Args:
            scenario (Scenario): The scenario to use for sampling.

        Returns:
            UserRequest: A request object for the task.
        """
        # TODO: create Delegated Request Creator to replace if-else
        if self.output_modality == "text":
            return self._sample_chat_request(scenario)
        elif self.output_modality == "embeddings":
            return self._sample_embedding_request(scenario)
        elif self.output_modality == "rerank":
            return self._sample_rerank_request(scenario)
        elif self.output_modality == "image":
            return self._sample_image_generation_request(scenario)
        else:
            raise ValueError(f"Unsupported output modality: {self.output_modality}")

    def _sample_chat_request(self, scenario: Optional[Scenario]) -> UserChatRequest:
        """Samples a chat request based on the scenario."""
        if self._is_dataset_mode(scenario):
            # Use dataset-mode sampling
            num_input_tokens, num_output_tokens = None, None
            self.additional_request_params["ignore_eos"] = False
        else:
            # Use scenario-based sampling
            self._validate_scenario(scenario)
            num_input_tokens, num_output_tokens = scenario.sample()
            self.additional_request_params["ignore_eos"] = True

        prompt = self._sample_text(num_input_tokens)
        num_prefill_tokens = self._get_prefill_token_count(prompt)

        return UserChatRequest(
            model=self.model,
            prompt=prompt,
            num_prefill_tokens=num_prefill_tokens,
            max_tokens=num_output_tokens,
            additional_request_params=self.additional_request_params,
        )

    def _sample_embedding_request(
        self, scenario: Optional[Scenario]
    ) -> UserEmbeddingRequest:
        """Samples an embedding request based on the scenario and batch size"""
        if self._is_dataset_mode(scenario):
            tokens_per_document = None
        else:
            self._validate_scenario(scenario)
            tokens_per_document = scenario.sample()

        # Sample different documents for each batch item
        documents = [
            self._sample_text(tokens_per_document) for _ in range(self.batch_size)
        ]
        num_prefill_tokens = sum(self.get_token_length(doc) for doc in documents)
        if tokens_per_document is not None:
            num_expected_tokens = tokens_per_document * self.batch_size
            self._check_discrepancy(num_expected_tokens, num_prefill_tokens, 0.2)

        return UserEmbeddingRequest(
            model=self.model,
            documents=documents,
            num_prefill_tokens=num_prefill_tokens,
            additional_request_params=self.additional_request_params,
        )

    def _sample_rerank_request(self, scenario: Optional[Scenario]) -> UserReRankRequest:
        """Samples a rerank request based on the scenario and batch size"""
        if self._is_dataset_mode(scenario):
            tokens_per_document, tokens_per_query = None, None
        else:
            self._validate_scenario(scenario)
            tokens_per_document, tokens_per_query = scenario.sample()

        query = self._sample_text(tokens_per_query)
        # Sample different documents for each batch item
        documents = [
            self._sample_text(tokens_per_document) for _ in range(self.batch_size)
        ]
        num_prefill_tokens = sum(
            self.get_token_length(doc) for doc in documents
        ) + self.get_token_length(query)

        return UserReRankRequest(
            model=self.model,
            documents=documents,
            query=query,
            num_prefill_tokens=num_prefill_tokens,
            additional_request_params=self.additional_request_params,
        )

    def _sample_image_generation_request(
        self, scenario: Optional[Scenario]
    ) -> UserImageGenerationRequest:
        """
        Samples an image generation request based on the scenario.

        Args:
            scenario (Scenario, optional): The scenario defining image size.
                Uses I(width,height) format same as vision tasks.
                e.g., I(1024,1024) or I(512,512)

        Returns:
            UserImageGenerationRequest: A request for image generation.
        """
        size = "auto"  # Default size
        num_images = 1  # Default number of images

        if scenario is not None and not self._is_dataset_mode(scenario):
            self._validate_scenario(scenario)
            try:
                # I() scenario returns: ((width, height), num_images, max_output_tokens)
                sample_result = scenario.sample()
                width, height = sample_result[0]
                size = f"{width}x{height}"
                num_images = sample_result[1]
            except (ValueError, TypeError, IndexError) as e:
                # Fallback to default if scenario parsing fails
                logger.warning(
                    f"Failed to sample from scenario {scenario.to_string()}: {e}. "
                    f"Falling back to default size '{size}' and "
                    f"num_images '{num_images}'."
                )

        prompt = self._sample_text(None)

        return UserImageGenerationRequest(
            model=self.model,
            prompt=prompt,
            size=size,
            quality=self.additional_request_params.get("quality", "auto"),
            num_images=num_images,
            additional_request_params=self.additional_request_params,
        )

    def _validate_scenario(self, scenario: Optional[Scenario]) -> None:
        """
        Validates that a scenario is provided and is of the correct type
        based on the output modality.

        Raises:
            ValueError: If no scenario is provided or if the scenario type
                is invalid.
        """
        if scenario is None:
            raise ValueError("A scenario is required when using the default dataset.")

        if self.output_modality == "text" and not isinstance(
            scenario.scenario_type, TextDistribution
        ):
            raise ValueError(
                f"Expected TextDistribution for text output, got "
                f"{type(scenario.scenario_type)}"
            )
        elif self.output_modality == "embeddings" and not isinstance(
            scenario.scenario_type, EmbeddingDistribution
        ):
            raise ValueError(
                f"Expected EmbeddingDistribution for embeddings output, got "
                f"{type(scenario.scenario_type)}"
            )
        elif self.output_modality == "image" and not isinstance(
            scenario.scenario_type, MultiModality
        ):
            raise ValueError(
                f"Expected MultiModality (I) for image output, got "
                f"{type(scenario.scenario_type)}"
            )

    def _sample_text(self, num_input_tokens: Optional[int]) -> str:
        """
        Samples text from a list of lines based on the specified number of
        input tokens. If num_input_tokens is None, samples a random line
        from `self.data`.

        Args:
            num_input_tokens (int): The target number of input tokens.

        Returns:
            str: A text prompt containing the desired number of tokens.
        """
        if not num_input_tokens:
            return random.choice(self.data)

        data_copy = self.data.copy()
        prompt = ""
        left_tokens_to_sample = num_input_tokens

        while left_tokens_to_sample > 0:
            random.shuffle(data_copy)
            for line in data_copy:
                line_tokens = self.tokenizer.encode(line, add_special_tokens=False)
                num_line_tokens = len(line_tokens)
                if num_line_tokens > left_tokens_to_sample:
                    # Truncate at token level, decode only needed tokens
                    truncated_text = str(
                        self.tokenizer.decode(
                            line_tokens[:left_tokens_to_sample],
                            skip_special_tokens=True,
                        )
                    )
                    prompt += (" " if prompt else "") + truncated_text
                    return prompt
                prompt += line
                left_tokens_to_sample -= num_line_tokens
        return prompt

    def _check_discrepancy(
        self, num_input_tokens: int, num_prefill_tokens: int, threshold: float = 0.1
    ) -> None:
        """
        Checks for and logs large discrepancies in token counts.

        Args:
            num_input_tokens (int): Expected number of input tokens.
            num_prefill_tokens (int): Actual number of input tokens.
            threshold (float, optional): Threshold for discrepancies.

        Raises:
            Warning: If the discrepancy exceeds threshold * num_input_tokens.
        """
        discrepancy = abs(num_input_tokens - num_prefill_tokens)
        if discrepancy > threshold * num_input_tokens:
            logger.warning(
                f"🚨 Sampling discrepancy detected: "
                f"num_input_tokens={num_input_tokens}, "
                f"num_prefill_tokens={num_prefill_tokens}, "
                f"discrepancy={discrepancy}"
            )

    def _get_prefill_token_count(self, prompt: str) -> int:
        """
        Counts prefill tokens by applying the chat template, matching how
        the model server counts prompt tokens.

        Falls back to raw token count if the tokenizer does not support
        chat templates.
        """
        messages = [{"role": "user", "content": prompt}]
        system_message = self.additional_request_params.get("system_message")
        if system_message:
            messages.insert(0, {"role": "system", "content": system_message})
        try:
            return len(
                self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True
                )
            )
        except Exception:
            return self.get_token_length(prompt, add_special_tokens=True)
