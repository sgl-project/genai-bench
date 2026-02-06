import random
from typing import Any, Dict, List, Optional

from genai_bench.data.config import DatasetConfig
from genai_bench.logging import init_logger
from genai_bench.protocol import (
    UserChatRequest,
    UserEmbeddingRequest,
    UserRequest,
    UserReRankRequest,
)
from genai_bench.sampling.base import Sampler
from genai_bench.scenarios.base import EmbeddingDistribution, Scenario, TextDistribution

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
        additional_request_params: Optional[Dict[str, Any]] = None,
        dataset_config: Optional[DatasetConfig] = None,
        prefix_len: Optional[int] = None,
        prefix_ratio: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(
            tokenizer, model, output_modality, additional_request_params, dataset_config
        )

        self.data = data
        self.batch_size = 1  # Default batch size
        self.prefix_len = prefix_len
        self.prefix_ratio = prefix_ratio
        # Globally shared prefix (generated once for --prefix-len,
        # per-request for --prefix-ratio)
        self._shared_prefix: Optional[str] = None

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
        else:
            raise ValueError(f"Unsupported output modality: {self.output_modality}")

    def _sample_chat_request(self, scenario: Optional[Scenario]) -> UserChatRequest:
        """Samples a chat request based on the scenario."""
        if self._is_dataset_mode(scenario):
            # Use dataset-mode sampling
            num_input_tokens, num_output_tokens = None, None
            effective_prefix_len = None
            self.additional_request_params["ignore_eos"] = False
        else:
            # Use scenario-based sampling
            self._validate_scenario(scenario)
            num_input_tokens, num_output_tokens = scenario.sample()
            self.additional_request_params["ignore_eos"] = True

            # Compute effective prefix length based on which option is used
            if self.prefix_ratio is not None:
                # For --prefix-ratio: compute per-request based on actual input tokens
                effective_prefix_len = int(num_input_tokens * self.prefix_ratio)
            elif self.prefix_len is not None:
                # For --prefix-len: use the fixed value
                effective_prefix_len = self.prefix_len
                # For non-deterministic scenarios, resample if needed
                max_resample_attempts = 10
                for _ in range(max_resample_attempts):
                    if num_input_tokens >= effective_prefix_len:
                        break
                    num_input_tokens, num_output_tokens = scenario.sample()
                else:
                    raise ValueError(
                        f"Could not sample input_tokens >= prefix_len "
                        f"({self.prefix_len}) after {max_resample_attempts} attempts. "
                        f"Last sample: {num_input_tokens} tokens."
                    )
            else:
                effective_prefix_len = None

        prompt = self._sample_text(num_input_tokens, effective_prefix_len)
        num_prefill_tokens = self.get_token_length(prompt)
        if num_input_tokens is not None:
            self._check_discrepancy(num_input_tokens, num_prefill_tokens, threshold=0.1)

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

    def _sample_text(
        self,
        num_input_tokens: Optional[int],
        effective_prefix_len: Optional[int] = None,
    ) -> str:
        """
        Samples text from a list of lines based on the specified number of
        input tokens. If num_input_tokens is None, samples a random line
        from `self.data`.

        Args:
            num_input_tokens (int): The target number of input tokens.
            effective_prefix_len (int): The effective prefix length for this
                request.

        Returns:
            str: A text prompt containing the desired number of tokens.
        """
        if not num_input_tokens:
            return random.choice(self.data)

        # Generate shared prefix if needed
        # Use fixed seed for multi-worker consistency (all workers generate same prefix)
        if effective_prefix_len is not None:
            prefix_rng = random.Random(42)
            if self.prefix_len is not None:
                # --prefix-len mode: generate once and cache
                if self._shared_prefix is None:
                    self._shared_prefix = self._generate_text_from_dataset(
                        effective_prefix_len, rng=prefix_rng
                    )
                    logger.info(
                        f"Generated shared prefix ({effective_prefix_len} tokens) "
                        f"from dataset for prefix caching. "
                        f"This prefix will be reused across all requests."
                    )
                prefix = self._shared_prefix
            else:
                # --prefix-ratio mode: generate fresh prefix per-request
                prefix = self._generate_text_from_dataset(
                    effective_prefix_len, rng=prefix_rng
                )
        else:
            prefix = None

        # Generate suffix (remaining tokens after prefix)
        suffix_len = num_input_tokens
        if effective_prefix_len is not None:
            suffix_len = num_input_tokens - effective_prefix_len

        # Generate suffix from dataset
        suffix = self._generate_text_from_dataset(suffix_len) if suffix_len > 0 else ""

        # Combine prefix + separator + suffix
        if effective_prefix_len is not None and effective_prefix_len > 0:
            # Generate a random 4-character hex string as separator
            # Using secrets.token_hex for cryptographic randomness
            import secrets

            separator = secrets.token_hex(2)  # 2 bytes = 4 hex chars
            separator_len = self.get_token_length(separator)

            # Check if separator fits in available space
            available_for_separator_and_suffix = suffix_len
            if separator_len > available_for_separator_and_suffix:
                # Truncate separator to fit available space
                max_separator_len = available_for_separator_and_suffix

                # Tokenize and truncate separator
                separator_token_ids = self.tokenizer.encode(
                    separator, add_special_tokens=False
                )
                truncated_separator_ids = separator_token_ids[:max_separator_len]
                separator = self.tokenizer.decode(
                    truncated_separator_ids, skip_special_tokens=True
                )
                separator_len = len(truncated_separator_ids)

                logger.debug(
                    f"Truncated separator from {len(separator_token_ids)} "
                    f"to {separator_len} tokens to fit within available space "
                    f"({available_for_separator_and_suffix} tokens)."
                )

            # Adjust suffix to account for separator length (truncated or full)
            adjusted_suffix_len = suffix_len - separator_len
            suffix = (
                self._generate_text_from_dataset(adjusted_suffix_len)
                if adjusted_suffix_len > 0
                else ""
            )
            return f"{prefix}{separator}{suffix}"
        else:
            # No prefix caching - just return the full prompt
            return self._generate_text_from_dataset(num_input_tokens)

    def _generate_text_from_dataset(
        self, num_tokens: int, rng: random.Random | None = None
    ) -> str:
        """
        Generate text from the dataset by concatenating lines until target token count.

        Args:
            num_tokens (int): The target number of tokens.
            rng (random.Random | None): Random generator to use for shuffling.
                If None, uses the global random state.
                Pass a seeded Random instance for deterministic generation
                (e.g., for prefix generation to ensure multi-worker consistency).

        Returns:
            str: Generated text with approximately num_tokens tokens.
        """
        if num_tokens == 0:
            return ""

        data_copy = self.data.copy()
        text = ""
        tokens_remaining = num_tokens
        shuffle_func = rng.shuffle if rng else random.shuffle

        while tokens_remaining > 0:
            shuffle_func(data_copy)
            for line in data_copy:
                line_tokens = self.tokenizer.encode(line, add_special_tokens=False)
                num_line_tokens = len(line_tokens)

                if num_line_tokens > tokens_remaining:
                    # Truncate at token level
                    truncated_text = self.tokenizer.decode(
                        line_tokens[:tokens_remaining], skip_special_tokens=True
                    )
                    text += (" " if text else "") + truncated_text
                    return text

                text += line
                tokens_remaining -= num_line_tokens

        return text

    def reset_prefix_cache(self):
        """Reset the prefix cache when switching to a new scenario."""
        self._shared_prefix = None

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
