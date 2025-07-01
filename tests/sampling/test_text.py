import unittest
from unittest.mock import MagicMock

from genai_bench.protocol import (
    UserChatRequest,
    UserEmbeddingRequest,
    UserReRankRequest,
)
from genai_bench.sampling.text import TextSampler
from genai_bench.scenarios import EmbeddingScenario, NormalDistribution
from genai_bench.scenarios.text import ReRankScenario


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
            use_scenario=True,
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
            use_scenario=False,
        )
        self.tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        scenario = NormalDistribution(
            mean_input_tokens=10,
            stddev_input_tokens=2,
            mean_output_tokens=20,
            stddev_output_tokens=5,
        )

        request = no_scenario_sampler.sample(scenario)

        self.assertIsInstance(request, UserChatRequest)
        self.assertEqual(request.model, "mock_model")
        self.assertIn(request.prompt, self.test_data)
        self.assertIsNone(
            request.max_tokens
        )  # Should be None for non-scenario sampling

    def test_sample_embedding_request(self):
        self.tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        embedding_sampler = TextSampler(
            tokenizer=self.tokenizer,
            model=self.model,
            output_modality="embeddings",
            data=self.test_data,
        )
        scenario = EmbeddingScenario(tokens_per_document=1024)

        request = embedding_sampler.sample(scenario)

        self.assertIsInstance(request, UserEmbeddingRequest)
        self.assertEqual(request.model, "mock_model")
        self.assertIsInstance(request.documents, list)
        self.assertTrue(len(request.documents) > 0)

    def test_sample_rerank_request(self):
        self.tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        rerank_sampler = TextSampler(
            tokenizer=self.tokenizer,
            model=self.model,
            output_modality="rerank",
            data=self.test_data,
        )
        scenario = ReRankScenario(tokens_per_document=1024, tokens_per_query=100)

        request = rerank_sampler.sample(scenario)

        self.assertIsInstance(request, UserReRankRequest)
        self.assertEqual(request.model, "mock_model")
        self.assertIsInstance(request.documents, list)
        self.assertIsInstance(request.query, str)

    def test_validate_scenario_invalid(self):
        with self.assertRaises(ValueError):
            self.sampler._validate_scenario(None)

    def test_validate_scenario_invalid2(self):
        invalid_scenario = MagicMock()
        invalid_scenario.scenario_type = "invalid"
        with self.assertRaises(ValueError):
            self.sampler._validate_scenario(invalid_scenario)

    def test_sample_chat_prefix_request(self):
        self.tokenizer.encode.side_effect = [
            [1] * 0,
            [1] * 20,
            [1] * 14,
            [1] * 14,
            [1] * 14,
            [1] * 14,
            [1] * 14,
        ]
        scenario = NormalDistribution(
            mean_input_tokens=20,
            stddev_input_tokens=0,
            mean_output_tokens=20,
            stddev_output_tokens=0,
        )
        prefix_sampler = TextSampler(
            tokenizer=self.tokenizer,
            model=self.model,
            output_modality=self.output_modality,
            data=self.test_data,
            use_scenario=True,
            prefix_length=10,  # Set a prefix length for testing
        )
        # Mock the char_token_ratio since the tokenizer is mocked
        prefix_sampler.char_token_ratio = 1.0
        result = prefix_sampler.sample(scenario)
        self.assertIsInstance(result, UserChatRequest)
        self.assertEqual(result.model, self.model)
        self.assertTrue(isinstance(result.prompt, str))
        self.assertGreater(len(result.prompt), 0)
        # The prompt should start with the generated prefix and a 4-digit number
        self.assertTrue(result.prompt.startswith(prefix_sampler.prefix))
        self.assertEqual(len(result.prompt), 20)

    def test_sample_chat_prefix_ratio_request(self):
        """Test prefix generation using ratio instead of fixed length."""
        self.tokenizer.encode.side_effect = [
            [1] * 0,
            [1] * 20,
            [1] * 14,
            [1] * 14,
            [1] * 14,
            [1] * 14,
            [1] * 14,
        ]
        scenario = NormalDistribution(
            mean_input_tokens=20,
            stddev_input_tokens=0,
            mean_output_tokens=20,
            stddev_output_tokens=0,
        )
        prefix_sampler = TextSampler(
            tokenizer=self.tokenizer,
            model=self.model,
            output_modality=self.output_modality,
            data=self.test_data,
            use_scenario=True,
            prefix_length_ratio=0.5,  # 50% of 20 tokens = 10 tokens
        )
        # Mock the char_token_ratio since the tokenizer is mocked
        prefix_sampler.char_token_ratio = 1.0
        result = prefix_sampler.sample(scenario)
        self.assertIsInstance(result, UserChatRequest)
        self.assertEqual(result.model, self.model)
        self.assertTrue(isinstance(result.prompt, str))
        self.assertGreater(len(result.prompt), 0)
        self.assertTrue(result.prompt.startswith(prefix_sampler.prefix))
        self.assertEqual(len(result.prompt), 20)
        # Verify the calculated prefix length is correct
        self.assertEqual(prefix_sampler._current_prefix_length, 10)

    def test_short_prompt_request(self):
        self.tokenizer.encode.return_value = [1] * 10
        self.sampler.data = ["2"]

        # Scenario asks for only 1 input token
        scenario = NormalDistribution(1, 0, 1, 0)

        result = self.sampler.sample(scenario)
        self.assertIsInstance(result, UserChatRequest)
        # The prompt will be the 4-digit number, truncated to 1 char
        self.assertEqual(result.prompt, str(result.prompt)[0])
        self.assertGreater(len(result.prompt), 0)

    def test_short_prompt_prefix_request(self):
        # Prefix length is 10, but scenario asks for only 1 input token
        self.tokenizer.encode.return_value = [1] * 10
        test_data = ["2"]
        scenario = NormalDistribution(1, 0, 1, 0)
        prefix_sampler = TextSampler(
            tokenizer=self.tokenizer,
            model=self.model,
            output_modality=self.output_modality,
            data=test_data,
            use_scenario=True,
            prefix_length=10,
        )
        with self.assertRaises(ValueError):
            prefix_sampler.sample(scenario)

    def test_empty_dataset(self):
        """Test sampling from an empty dataset."""
        empty_sampler = TextSampler(
            tokenizer=self.tokenizer,
            model=self.model,
            output_modality=self.output_modality,
            data=[],
            use_scenario=True,
        )
        scenario = NormalDistribution(10, 0, 10, 0)

        with self.assertRaises(ValueError) as context:
            empty_sampler.sample(scenario)

        self.assertEqual(
            str(context.exception), "Cannot sample text from an empty dataset"
        )
