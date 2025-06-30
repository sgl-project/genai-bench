import random
from typing import Any, Dict, List, Optional

from genai_bench.logging import init_logger
from genai_bench.protocol import (
    UserChatRequest,
    UserEmbeddingRequest,
    UserRequest,
    UserReRankRequest,
)
from genai_bench.sampling.base import Sampler
from genai_bench.scenarios.base import EmbeddingDistribution, Scenario, TextDistribution
from genai_bench.utils import calculate_char_token_ratio

logger = init_logger(__name__)


class TextSampler(Sampler):
    """
    Unified sampler for text-based tasks, supporting multiple task types:
    - `text-to-text`: Standard chat or generation tasks.
    - `text-to-embeddings`: Embedding generation from text.
    """

    input_modality = "text"
    supported_tasks = {"text-to-text", "text-to-embeddings", "text-to-rerank"}

    def __init__(
        self,
        tokenizer,
        model: str,
        output_modality: str,
        data: List[str],
        use_scenario: bool = True,
        prefix_length: int = 0,
        additional_request_params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(tokenizer, model, output_modality, additional_request_params)

        self.data = data
        self.use_scenario = use_scenario
        self.char_token_ratio = calculate_char_token_ratio(tokenizer, data)

        # Set ignore_eos based on scenario usage
        if use_scenario:
            self.additional_request_params["ignore_eos"] = True
        else:
            self.additional_request_params["ignore_eos"] = False

        self.batch_size = 1  # Default batch size
        self.prefix_length = prefix_length
        self.prefix = ""

    def sample(self, scenario: Scenario) -> UserRequest:
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
        else:
            raise ValueError(f"Unsupported output modality: {self.output_modality}")

    def _sample_chat_request(self, scenario: Scenario) -> UserChatRequest:
        """Samples a chat request based on the scenario."""
        if not self.use_scenario:
            # Sample directly from a CSV dataset
            prompt = self._sample_prompt()
            max_tokens = None
            num_prefill_tokens = self.get_token_length(prompt)
        else:
            # Use scenario-based sampling
            self._validate_scenario(scenario)
            num_input_tokens, num_output_tokens = scenario.sample()

            prompt = self._sample_text(num_input_tokens)
            max_tokens = num_output_tokens
            num_prefill_tokens = self.get_token_length(prompt)
            self._check_discrepancy(num_input_tokens, num_prefill_tokens)

        return UserChatRequest(
            model=self.model,
            prompt=prompt,
            num_prefill_tokens=num_prefill_tokens,
            max_tokens=max_tokens,
            additional_request_params=self.additional_request_params,
        )

    def _sample_embedding_request(self, scenario: Scenario) -> UserEmbeddingRequest:
        """Samples an embedding request based on the scenario and batch size"""
        self._validate_scenario(scenario)
        tokens_per_document = scenario.sample()
        document = self._sample_text(tokens_per_document)

        # Use batch_size to create multiple copies of the document
        documents = [document] * self.batch_size
        num_prefill_tokens = sum(self.get_token_length(doc) for doc in documents)
        num_expected_tokens = tokens_per_document * self.batch_size

        self._check_discrepancy(num_expected_tokens, num_prefill_tokens, 20)

        return UserEmbeddingRequest(
            model=self.model,
            documents=documents,
            num_prefill_tokens=num_prefill_tokens,
            additional_request_params=self.additional_request_params,
        )

    def _sample_rerank_request(self, scenario: Scenario) -> UserReRankRequest:
        """Samples a rerank request based on the scenario and batch size"""
        self._validate_scenario(scenario)
        tokens_per_document, tokens_per_query = scenario.sample()
        document = self._sample_text(tokens_per_document)
        query = self._sample_text(tokens_per_query)

        # Use batch_size to create multiple copies of the document
        documents = [document] * self.batch_size
        num_prefill_tokens = self.batch_size * (
            self.get_token_length(document) + self.get_token_length(document)
        )

        return UserReRankRequest(
            model=self.model,
            documents=documents,
            query=query,
            num_prefill_tokens=num_prefill_tokens,
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

    def _generate_prefix(self) -> str:
        """
        Generates prefix of length self.prefix_length to be
        prepended to all input prompts.
        """

        data_copy = self.data.copy()

        prefix = ""
        prefix_tokens = 0
        # Generate the prefix
        while prefix_tokens < self.prefix_length:
            random.shuffle(data_copy)
            for line in data_copy:
                tokens = self.get_token_length(line)
                if prefix_tokens + tokens > self.prefix_length:
                    # Truncate the line if it exceeds the remaining prefix length
                    remaining_prefix_len = self.prefix_length - prefix_tokens
                    prefix += line[: int(remaining_prefix_len * self.char_token_ratio)]
                    prefix_tokens += remaining_prefix_len
                    break
                prefix += line
                prefix_tokens += tokens

        return prefix

    def _sample_text(self, num_input_tokens: int) -> str:
        """
        Samples text from a list of lines based on the specified number of
        input tokens.

        Args:
            num_input_tokens (int): The target number of input tokens.

        Raises:
            ValueError: if the prompt length is shorter than the prefix
                length.

        Returns:
            str: A text prompt containing the desired number of tokens.
        """
        data_copy = self.data.copy()

        if num_input_tokens <= self.prefix_length:
            raise ValueError("Prefix length must be shorter than total input length")

        # Generate the prefix if it hasn't been created yet
        if self.get_token_length(self.prefix) != self.prefix_length:
            self.prefix = self._generate_prefix()

        # Prepend the prefix to all prompts with a 4 randomly picked digits
        prompt = f"{self.prefix}{random.randint(1000,9999)}"
        left_tokens_to_sample = num_input_tokens - self.get_token_length(prompt)

        if left_tokens_to_sample < 0:
            return prompt[: self.get_token_length(prompt) + left_tokens_to_sample]
        while left_tokens_to_sample > 0:
            random.shuffle(data_copy)
            for line in data_copy:
                tokens = self.get_token_length(line)
                if tokens > left_tokens_to_sample:
                    # This will cut off a line in the middle of a word, but
                    # that's ok since a llm should be able to handle that.
                    prompt += line[: int(left_tokens_to_sample * self.char_token_ratio)]
                    return prompt
                prompt += line
                left_tokens_to_sample -= tokens
        return prompt

    def _sample_prompt(self) -> str:
        """
        Samples a single prompt from the loaded data.

        Returns:
            str: A single prompt.
        """
        return random.choice(self.data)

    def _check_discrepancy(
        self, num_input_tokens: int, num_prefill_tokens: int, threshold: float = 10
    ) -> None:
        """
        Checks for and logs large discrepancies in token counts.

        Args:
            num_input_tokens (int): Expected number of input tokens.
            num_prefill_tokens (int): Actual number of input tokens.
            threshold (float, optional): Threshold for discrepancies.

        Raises:
            Warning: If the discrepancy exceeds 10% or is greater than 10
                tokens.
        """
        discrepancy = abs(num_input_tokens - num_prefill_tokens)
        if discrepancy > threshold * num_input_tokens and discrepancy > 10:
            logger.warning(
                f"🚨 Sampling discrepancy detected: "
                f"num_input_tokens={num_input_tokens}, "
                f"num_prefill_tokens={num_prefill_tokens}, "
                f"discrepancy={discrepancy}"
            )
