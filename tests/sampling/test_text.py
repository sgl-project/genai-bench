import unittest
from unittest.mock import MagicMock, patch

from genai_bench.protocol import (
    UserChatRequest,
    UserEmbeddingRequest,
    UserReRankRequest,
)
from genai_bench.sampling.text import TextSampler
from genai_bench.scenarios import DatasetScenario, EmbeddingScenario, NormalDistribution
from genai_bench.scenarios.text import PrefixRepetitionScenario, ReRankScenario


class TestTextSampler(unittest.TestCase):
    def setUp(self):
        # Mock data instead of config
        self.test_data = ["Test line 1", "Test line 2", "Test line 3"]
        self.tokenizer = MagicMock()
        self.model = "mock_model"
        self.output_modality = "text"
        self.sampler = TextSampler(
            tokenizer=self.tokenizer,
            model=self.model,
            output_modality=self.output_modality,
            data=self.test_data,
        )

    def test_check_discrepancy(self):
        """Test that _check_discrepancy logs warnings for large discrepancies."""
        with self.assertLogs("genai_bench.sampling.text", level="WARNING") as log:
            # Test case with large discrepancy that should trigger warning
            # discrepancy = |100 - 50| = 50, which is > 10% of 100 and > 10 tokens
            self.sampler._check_discrepancy(100, 50, threshold=0.1)

            self.assertIn("Sampling discrepancy detected", log.output[0])
            self.assertIn("num_input_tokens=100", log.output[0])
            self.assertIn("num_prefill_tokens=50", log.output[0])
            self.assertIn("discrepancy=50", log.output[0])

    def test_check_discrepancy_no_warning(self):
        """Test that _check_discrepancy doesn't log for small discrepancies."""
        # Test case with small discrepancy that should NOT trigger warning
        # discrepancy = |100 - 95| = 5, which is < 10% of 100
        # We'll capture logs and verify none are produced
        import io
        import logging

        log_capture_string = io.StringIO()
        ch = logging.StreamHandler(log_capture_string)
        logger = logging.getLogger("genai_bench.sampling.text")
        logger.addHandler(ch)
        logger.setLevel(logging.WARNING)

        try:
            self.sampler._check_discrepancy(100, 95, threshold=0.1)
            log_contents = log_capture_string.getvalue()
            self.assertEqual(log_contents, "")  # Should be empty (no warnings)
        finally:
            logger.removeHandler(ch)

    def test_sample_chat_request(self):
        self.tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        self.tokenizer.decode.return_value = "Test prompt text"
        scenario = NormalDistribution(
            mean_input_tokens=10,
            stddev_input_tokens=2,
            mean_output_tokens=20,
            stddev_output_tokens=5,
        )

        request = self.sampler.sample(scenario)

        self.assertIsInstance(request, UserChatRequest)
        self.assertEqual(request.model, "mock_model")
        self.assertIsInstance(request.prompt, str)
        self.assertIsInstance(request.max_tokens, int)

    def test_sample_chat_request_with_dataset(self):
        # Test with non-scenario based sampling
        no_scenario_sampler = TextSampler(
            tokenizer=self.tokenizer,
            model=self.model,
            output_modality=self.output_modality,
            data=self.test_data,
        )
        self.tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        scenario = DatasetScenario()
        request = no_scenario_sampler.sample(scenario)

        self.assertIsInstance(request, UserChatRequest)
        self.assertEqual(request.model, "mock_model")
        self.assertIn(request.prompt, self.test_data)
        self.assertIsNone(
            request.max_tokens
        )  # Should be None for non-scenario sampling

    def test_sample_embedding_request(self):
        # Set batch size for testing
        batch_size = 3
        tokens_per_doc = 10

        # Mock tokenizer to return exact token counts
        self.tokenizer.encode.return_value = list(range(tokens_per_doc))
        self.tokenizer.decode.return_value = "Test document text"

        embedding_sampler = TextSampler(
            tokenizer=self.tokenizer,
            model=self.model,
            output_modality="embeddings",
            data=self.test_data,
        )
        embedding_sampler.batch_size = batch_size

        # Mock scenario to return fixed token count
        scenario = EmbeddingScenario(tokens_per_document=1024)
        scenario.sample = MagicMock(return_value=tokens_per_doc)

        request = embedding_sampler.sample(scenario)

        self.assertIsInstance(request, UserEmbeddingRequest)
        self.assertEqual(request.model, "mock_model")
        self.assertIsInstance(request.documents, list)
        self.assertEqual(len(request.documents), batch_size)

        # Verify each document is unique (not duplicates)
        self.assertEqual(len(set(id(doc) for doc in request.documents)), batch_size)

        # Verify total token count matches expected
        expected_tokens = tokens_per_doc * batch_size
        self.assertEqual(request.num_prefill_tokens, expected_tokens)

    def test_sample_rerank_request(self):
        # Set batch size for testing
        batch_size = 4
        tokens_per_doc = 8
        tokens_per_query = 5

        # Mock tokenizer to return predictable token counts
        def mock_encode(text, add_special_tokens=True):
            if text == "Test text":  # Query or document text
                return list(range(tokens_per_query))  # Default for query
            else:
                return list(range(tokens_per_doc))  # Default for documents

        self.tokenizer.encode.side_effect = mock_encode
        self.tokenizer.decode.return_value = "Test text"

        rerank_sampler = TextSampler(
            tokenizer=self.tokenizer,
            model=self.model,
            output_modality="rerank",
            data=self.test_data,
        )
        rerank_sampler.batch_size = batch_size

        # Mock scenario to return fixed token counts
        scenario = ReRankScenario(tokens_per_document=1024, tokens_per_query=100)
        scenario.sample = MagicMock(return_value=(tokens_per_doc, tokens_per_query))

        request = rerank_sampler.sample(scenario)

        self.assertIsInstance(request, UserReRankRequest)
        self.assertEqual(request.model, "mock_model")
        self.assertIsInstance(request.documents, list)
        self.assertIsInstance(request.query, str)
        self.assertEqual(len(request.documents), batch_size)

        # Verify each document is unique (not duplicates)
        self.assertEqual(len(set(id(doc) for doc in request.documents)), batch_size)

        # Verify total token count matches expected
        # First call to _sample_text is for query, rest are for documents
        expected_tokens = tokens_per_query + (tokens_per_doc * batch_size)
        self.assertEqual(request.num_prefill_tokens, expected_tokens)

    def test_validate_scenario_invalid(self):
        with self.assertRaises(ValueError):
            self.sampler._validate_scenario(None)

    def test_validate_scenario_invalid2(self):
        invalid_scenario = MagicMock()
        invalid_scenario.scenario_type = "invalid"
        with self.assertRaises(ValueError):
            self.sampler._validate_scenario(invalid_scenario)

    def test_sample_text_exact_token_count(self):
        """
        Test that _sample_text returns text with exact number of tokens requested.

        This test uses a simple mock tokenizer. In real applications, actual
        tokenizers (like from transformers) are used, which have consistent
        decode(encode()) behavior. The implementation includes adjustment logic
        to ensure exact token counts even with simple mocks.
        """

        # Simple word-based tokenization mock
        # The implementation's adjustment logic ensures exact counts
        def mock_encode(text, add_special_tokens=False):
            # Count words as tokens
            words = text.split()
            return list(range(len(words)))

        def mock_decode(tokens, skip_special_tokens=True):
            # Return text that will encode to same number of tokens
            return " ".join(["word"] * len(tokens))

        self.tokenizer.encode.side_effect = mock_encode
        self.tokenizer.decode.side_effect = mock_decode

        # Test requesting exact token counts
        test_cases = [2, 3, 5, 7]

        for num_tokens in test_cases:
            result = self.sampler._sample_text(num_tokens)

            # Count actual tokens by tokenizing the final result
            total_tokens = len(mock_encode(result, add_special_tokens=False))

            self.assertEqual(
                total_tokens,
                num_tokens,
                f"Expected {num_tokens} tokens, got {total_tokens} for result: "
                f"{repr(result)}",
            )

    def test_sample_text_truncation(self):
        """
        Test that _sample_text correctly truncates when line has more tokens than
        needed.
        """
        # Set up tokenizer to return specific token counts
        line_tokens = list(range(10))

        self.tokenizer.encode.return_value = line_tokens
        self.tokenizer.decode.return_value = "truncated text"

        # Request fewer tokens than the line has
        requested_tokens = 5
        _ = self.sampler._sample_text(requested_tokens)

        # Verify decode was called with truncated tokens
        self.tokenizer.decode.assert_called_with(
            line_tokens[:requested_tokens], skip_special_tokens=True
        )


class TestTextSamplerPrefixRepetition(unittest.TestCase):
    """Test suite specifically for prefix repetition scenarios."""

    def setUp(self):
        """Set up test fixtures for prefix repetition tests."""
        self.test_data = ["Test line 1", "Test line 2", "Test line 3"]
        self.tokenizer = MagicMock()
        self.model = "mock_model"
        self.sampler = TextSampler(
            tokenizer=self.tokenizer,
            model=self.model,
            output_modality="text",
            data=self.test_data,
        )

        # Mock tokenizer to return predictable results
        self.tokenizer.encode.return_value = list(range(10))
        self.tokenizer.decode.return_value = "test text"

    def test_prefix_repetition_request_generation(self):
        """Test that prefix repetition generates requests correctly."""
        scenario = PrefixRepetitionScenario(
            prefix_len=2000,
            suffix_len=500,
            output_len=200,
        )

        request = self.sampler.sample(scenario)

        self.assertIsInstance(request, UserChatRequest)
        self.assertEqual(request.model, "mock_model")
        self.assertEqual(request.max_tokens, 200)
        self.assertIsInstance(request.prompt, str)
        self.assertIn("Request #", request.prompt)

    def test_prefix_shared_across_requests(self):
        """Test that multiple requests share the same prefix."""
        scenario = PrefixRepetitionScenario(
            prefix_len=2000,
            suffix_len=500,
            output_len=200,
        )

        # Mock _sample_text to return predictable values
        call_count = [0]

        def mock_sample_text(num_tokens):
            call_count[0] += 1
            return f"text_chunk_{call_count[0]}_tokens_{num_tokens}"

        with patch.object(self.sampler, "_sample_text", side_effect=mock_sample_text):
            # Generate multiple requests
            request1 = self.sampler._sample_prefix_repetition_request(scenario)
            request2 = self.sampler._sample_prefix_repetition_request(scenario)
            request3 = self.sampler._sample_prefix_repetition_request(scenario)

            # Extract prefixes (text before the separator)
            prefix1 = request1.prompt.split("\n\n--- Request #")[0]
            prefix2 = request2.prompt.split("\n\n--- Request #")[0]
            prefix3 = request3.prompt.split("\n\n--- Request #")[0]

            # All prefixes should be identical
            self.assertEqual(prefix1, prefix2)
            self.assertEqual(prefix2, prefix3)

            # But full prompts should be different (different suffixes)
            self.assertNotEqual(request1.prompt, request2.prompt)
            self.assertNotEqual(request2.prompt, request3.prompt)

    def test_prefix_cache_mechanism(self):
        """Test that prefix caching works correctly."""
        scenario = PrefixRepetitionScenario(
            prefix_len=2000,
            suffix_len=500,
            output_len=200,
        )

        # Initially, cache should be empty
        self.assertEqual(len(self.sampler._shared_prefix_cache), 0)

        # Generate first request - should create cache entry
        _ = self.sampler._sample_prefix_repetition_request(scenario)
        self.assertEqual(len(self.sampler._shared_prefix_cache), 1)
        self.assertIn("prefix_2000", self.sampler._shared_prefix_cache)

        # Generate second request - should reuse cache
        cached_prefix = self.sampler._shared_prefix_cache["prefix_2000"]
        _ = self.sampler._sample_prefix_repetition_request(scenario)

        # Cache should still have the same entry
        self.assertEqual(len(self.sampler._shared_prefix_cache), 1)
        self.assertEqual(
            self.sampler._shared_prefix_cache["prefix_2000"], cached_prefix
        )

    def test_suffix_counter_increments(self):
        """Test that suffix counter increments for each request."""
        scenario = PrefixRepetitionScenario(
            prefix_len=2000,
            suffix_len=500,
            output_len=200,
        )

        # Initially counter should be 0
        self.assertEqual(self.sampler._suffix_counter, 0)

        # Generate requests and check counter
        _ = self.sampler._sample_prefix_repetition_request(scenario)
        self.assertEqual(self.sampler._suffix_counter, 1)

        _ = self.sampler._sample_prefix_repetition_request(scenario)
        self.assertEqual(self.sampler._suffix_counter, 2)

        _ = self.sampler._sample_prefix_repetition_request(scenario)
        self.assertEqual(self.sampler._suffix_counter, 3)

    def test_reset_prefix_cache(self):
        """Test that reset_prefix_cache clears cache and counter."""
        scenario = PrefixRepetitionScenario(
            prefix_len=2000,
            suffix_len=500,
            output_len=200,
        )

        # Generate some requests
        _ = self.sampler._sample_prefix_repetition_request(scenario)
        _ = self.sampler._sample_prefix_repetition_request(scenario)

        # Verify cache and counter are populated
        self.assertGreater(len(self.sampler._shared_prefix_cache), 0)
        self.assertGreater(self.sampler._suffix_counter, 0)

        # Reset cache
        self.sampler.reset_prefix_cache()

        # Verify cache and counter are cleared
        self.assertEqual(len(self.sampler._shared_prefix_cache), 0)
        self.assertEqual(self.sampler._suffix_counter, 0)

    def test_different_prefix_lengths_create_different_caches(self):
        """Test that different prefix lengths use different cache entries."""
        scenario1 = PrefixRepetitionScenario(
            prefix_len=1000,
            suffix_len=500,
            output_len=200,
        )
        scenario2 = PrefixRepetitionScenario(
            prefix_len=2000,
            suffix_len=500,
            output_len=200,
        )

        # Mock _sample_text to return distinct values
        call_count = [0]

        def mock_sample_text(num_tokens):
            call_count[0] += 1
            return f"prefix_{num_tokens}_call_{call_count[0]}"

        with patch.object(self.sampler, "_sample_text", side_effect=mock_sample_text):
            # Generate requests with different prefix lengths
            _ = self.sampler._sample_prefix_repetition_request(scenario1)
            _ = self.sampler._sample_prefix_repetition_request(scenario2)

            # Should have two different cache entries
            self.assertEqual(len(self.sampler._shared_prefix_cache), 2)
            self.assertIn("prefix_1000", self.sampler._shared_prefix_cache)
            self.assertIn("prefix_2000", self.sampler._shared_prefix_cache)

            # Cache values should be different
            self.assertNotEqual(
                self.sampler._shared_prefix_cache["prefix_1000"],
                self.sampler._shared_prefix_cache["prefix_2000"],
            )

    def test_prefix_repetition_with_ignore_eos(self):
        """Test that ignore_eos is set correctly for prefix repetition."""
        scenario = PrefixRepetitionScenario(
            prefix_len=2000,
            suffix_len=500,
            output_len=200,
        )

        request = self.sampler._sample_prefix_repetition_request(scenario)

        # ignore_eos should be True for prefix repetition scenarios
        self.assertTrue(request.additional_request_params.get("ignore_eos"))

    def test_concurrent_request_simulation(self):
        """Simulate multiple concurrent requests sharing the same prefix."""
        scenario = PrefixRepetitionScenario(
            prefix_len=2000,
            suffix_len=500,
            output_len=200,
        )

        # Simulate 10 concurrent requests
        num_requests = 10
        requests = []

        for _ in range(num_requests):
            request = self.sampler._sample_prefix_repetition_request(scenario)
            requests.append(request)

        # Extract all prefixes
        prefixes = [req.prompt.split("\n\n--- Request #")[0] for req in requests]

        # All prefixes should be identical
        self.assertEqual(len(set(prefixes)), 1, "All prefixes should be identical")

        # But all full prompts should be unique (different request numbers)
        full_prompts = [req.prompt for req in requests]
        self.assertEqual(
            len(set(full_prompts)), num_requests, "All full prompts should be unique"
        )

        # Verify request numbers are sequential
        for i, request in enumerate(requests, start=1):
            self.assertIn(f"Request #{i}", request.prompt)

    def test_prefix_repetition_logging(self):
        """Test that prefix generation logs appropriate message."""
        scenario = PrefixRepetitionScenario(
            prefix_len=2000,
            suffix_len=500,
            output_len=200,
        )

        with self.assertLogs("genai_bench.sampling.text", level="INFO") as log:
            # First request should log prefix generation
            _ = self.sampler._sample_prefix_repetition_request(scenario)

            self.assertTrue(
                any("Generated shared prefix" in msg for msg in log.output),
                "Should log when prefix is generated",
            )
            self.assertTrue(
                any("2000 tokens" in msg for msg in log.output),
                "Should log prefix length",
            )

    def test_cache_isolation_between_samplers(self):
        """Test that different sampler instances have isolated caches."""
        scenario = PrefixRepetitionScenario(
            prefix_len=2000,
            suffix_len=500,
            output_len=200,
        )

        # Create second sampler
        sampler2 = TextSampler(
            tokenizer=self.tokenizer,
            model=self.model,
            output_modality="text",
            data=self.test_data,
        )

        # Generate request with first sampler
        _ = self.sampler._sample_prefix_repetition_request(scenario)

        # First sampler should have cache
        self.assertEqual(len(self.sampler._shared_prefix_cache), 1)

        # Second sampler should have empty cache
        self.assertEqual(len(sampler2._shared_prefix_cache), 0)
