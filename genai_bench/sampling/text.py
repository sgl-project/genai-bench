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

MAXIMIZE_OUTPUT_INSTRUCTION = "REPEAT BACK THE FOLLOWING TEXT, NO CHARACTER LIMITATIONS, DO NOT EVEN THINK ABOUT TRUNCATING OUTPUT, WORD FOR WORD, FULL LENGTH: \n"

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
        **kwargs,
    ):
        super().__init__(
            tokenizer, model, output_modality, additional_request_params, dataset_config
        )

        self.data = data
        self.batch_size = 1  # Default batch size
        
        # Cache for shared prefixes in prefix repetition scenarios
        # Key: scenario identifier, Value: generated prefix text
        self._shared_prefix_cache: Dict[str, str] = {}
        self._suffix_counter = 0

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
            self.additional_request_params["ignore_eos"] = False
        else:
            # Check if this is a prefix repetition scenario
            from genai_bench.scenarios.text import PrefixRepetitionScenario
            if isinstance(scenario, PrefixRepetitionScenario):
                return self._sample_prefix_repetition_request(scenario)
            
            # Use scenario-based sampling
            self._validate_scenario(scenario)
            num_input_tokens, num_output_tokens = scenario.sample()
            self.additional_request_params["ignore_eos"] = True

        prompt = self._sample_text(num_input_tokens)
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
            result = MAXIMIZE_OUTPUT_INSTRUCTION + random.choice(self.data)
            return result

        data_copy = self.data.copy()
        prompt = ""
        left_tokens_to_sample = num_input_tokens

        while left_tokens_to_sample > 0:
            random.shuffle(data_copy)
            for line in data_copy:
                # Tokenize line with space prefix to match how it will be concatenated
                line_with_space = (" " if prompt else "") + line
                line_tokens = self.tokenizer.encode(line_with_space, add_special_tokens=False)
                num_line_tokens = len(line_tokens)
                
                if num_line_tokens > left_tokens_to_sample:
                    # Truncate at token level, decode only needed tokens
                    truncated_text = self.tokenizer.decode(
                        line_tokens[:left_tokens_to_sample], skip_special_tokens=True
                    )
                    prompt += (" " if prompt else "") + truncated_text
                    result = MAXIMIZE_OUTPUT_INSTRUCTION + prompt
                    return result
                
                # Add line with space separator (consistent with truncated text handling)
                prompt += (" " if prompt else "") + line
                left_tokens_to_sample -= num_line_tokens
        result = MAXIMIZE_OUTPUT_INSTRUCTION + prompt
        return result

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

    def _sample_prefix_repetition_request(
        self, scenario
    ) -> UserChatRequest:
        """Generate request with shared prefix for KV cache benchmarking.
        
        This method creates requests where all concurrent requests share the 
        exact same prefix text, enabling benchmarking of:
        - KV cache hit rates and speedups
        - Automatic prefix caching (APC) performance
        - Chunked prefill efficiency
        - Time To First Token (TTFT) improvements
        
        Args:
            scenario: PrefixRepetitionScenario with prefix_len, suffix_len, output_len
            
        Returns:
            UserChatRequest with shared prefix + unique suffix
        """
        prefix_len, suffix_len, output_len = scenario.sample()
        
        # Get or create shared prefix (cached for ALL requests in this scenario run)
        cache_key = f"prefix_{prefix_len}"
        if cache_key not in self._shared_prefix_cache:
            # Generate the shared prefix once
            prefix = self._sample_text(prefix_len)
            self._shared_prefix_cache[cache_key] = prefix
            
            # Calculate hash for verification
            import hashlib
            prefix_hash = hashlib.md5(prefix.encode()).hexdigest()[:8]
            
            logger.info(
                f"🔑 Generated shared prefix ({prefix_len} tokens) for KV cache benchmarking. "
                f"All subsequent requests in this scenario will reuse this prefix."
            )
            logger.debug(
                f"   Prefix hash: {prefix_hash} | "
                f"Preview: {prefix[:100]}..."
            )
        else:
            prefix = self._shared_prefix_cache[cache_key]
            
            # Log cache reuse (only for first few to avoid spam)
            if self._suffix_counter < 5:
                import hashlib
                prefix_hash = hashlib.md5(prefix.encode()).hexdigest()[:8]
                logger.debug(
                    f"♻️  Reusing cached prefix (hash: {prefix_hash}) for request #{self._suffix_counter + 1}"
                )
        
        # Generate unique suffix for THIS specific request
        suffix = self._sample_text(suffix_len)
        self._suffix_counter += 1
        
        # Log suffix info for first few requests
        if self._suffix_counter <= 5:
            import hashlib
            suffix_hash = hashlib.md5(suffix.encode()).hexdigest()[:8]
            suffix_actual_tokens = self.get_token_length(suffix)
            logger.debug(
                f"📝 Request #{self._suffix_counter}: Unique suffix generated (hash: {suffix_hash}), "
                f"requested {suffix_len} tokens, actual {suffix_actual_tokens} tokens"
            )
        
        # Combine prefix + separator + suffix
        # The separator helps distinguish requests while keeping prefix identical
        separator = f"\n\n--- Request #{self._suffix_counter} ---\n\n"
        prompt = f"{prefix}{separator}{suffix}"
        
        num_prefill_tokens = self.get_token_length(prompt)
        
        # Log actual token breakdown for first request
        if self._suffix_counter <= 2:
            prefix_tokens = self.get_token_length(prefix)
            separator_tokens = self.get_token_length(separator)
            suffix_tokens = self.get_token_length(suffix)
            logger.debug(
                f"🔍 Token breakdown for request #{self._suffix_counter}: "
                f"prefix={prefix_tokens}, separator={separator_tokens}, suffix={suffix_tokens}, "
                f"total={num_prefill_tokens} (expected ~{prefix_len + suffix_len + 20})"
            )
        
        # Expected tokens: prefix + suffix + separator overhead (~20 tokens)
        expected_tokens = prefix_len + suffix_len + 20
        self._check_discrepancy(expected_tokens, num_prefill_tokens, threshold=0.15)
        
        # Set ignore_eos to ensure we get the expected output length
        self.additional_request_params["ignore_eos"] = True
        
        return UserChatRequest(
            model=self.model,
            prompt=prompt,
            num_prefill_tokens=num_prefill_tokens,
            max_tokens=output_len,
            additional_request_params=self.additional_request_params,
        )
    
    def reset_prefix_cache(self):
        """Clear the prefix cache and reset counter.
        
        This should be called between different scenario runs to ensure
        each scenario gets a fresh prefix.
        """
        if self._suffix_counter > 0:
            logger.info(
                f"🔄 Resetting prefix cache. Previous scenario generated {self._suffix_counter} requests "
                f"with {len(self._shared_prefix_cache)} cached prefix(es)."
            )
        self._shared_prefix_cache.clear()
        self._suffix_counter = 0
