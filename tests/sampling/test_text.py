import unittest
from unittest.mock import MagicMock

from genai_bench.protocol import (
    UserChatRequest,
    UserEmbeddingRequest,
    UserReRankRequest,
)
from genai_bench.sampling.text import TextSampler
from genai_bench.scenarios import DatasetScenario, EmbeddingScenario, NormalDistribution
from genai_bench.scenarios.text import ReRankScenario
from genai_bench.data.config import DatasetConfig, DatasetSourceConfig


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

    def test_synthetic_cached_prefix_and_exact_length(self):
        """
        Ensure synthetic prompt generation matches tore-speed semantics:
        - Exact input token length equals requested.
        - Cached prefix occupies the first synthetic_cached_input_length tokens.
        - Unique marker and tail are present after the prefix.
        """

        # Fake tokenizer that understands the synthetic builder pieces
        class FakeTokenizer:
            pad_token_id = 0
            eos_token_id = 100

            def __init__(self):
                self.base_phrase = "hi,"
                self.base_tokens = [1, 2]
                self.marker_tokens = [8, 9, 10]
                self.tail_text = " Write a very long essay about San Francisco"
                self.tail_tokens = [3, 4, 5]

            def encode(self, text, add_special_tokens=False):
                # Exact matches used by the builder
                if text == self.base_phrase:
                    return list(self.base_tokens)
                if text == self.tail_text:
                    return list(self.tail_tokens)
                # Marker text during build contains '-' and digits
                if any(ch.isdigit() or ch == '-' for ch in text):
                    return list(self.marker_tokens)

                # When re-encoding the final prompt for counting, parse known segments
                tokens = []
                i = 0
                while i < len(text):
                    if text.startswith(self.base_phrase, i):
                        tokens.extend(self.base_tokens)
                        i += len(self.base_phrase)
                    elif text.startswith("MARK", i):
                        tokens.extend(self.marker_tokens)
                        i += len("MARK")
                    elif text.startswith(self.tail_text, i):
                        tokens.extend(self.tail_tokens)
                        i += len(self.tail_text)
                    else:
                        # Skip any other chars (spaces/pads) as 0-cost
                        i += 1
                return tokens

            def decode(self, tokens, skip_special_tokens=True):
                # Reconstruct the text from token ids by chunking known patterns
                out = []
                i = 0
                while i < len(tokens):
                    if tokens[i : i + 2] == self.base_tokens:
                        out.append(self.base_phrase)
                        i += 2
                    elif tokens[i : i + 3] == self.marker_tokens:
                        out.append("MARK")
                        i += 3
                    elif tokens[i : i + 3] == self.tail_tokens:
                        out.append(self.tail_text)
                        i += 3
                    elif tokens[i] == self.pad_token_id:
                        i += 1
                    else:
                        # Fallback for any stray token
                        i += 1
                return "".join(out)

        tokenizer = FakeTokenizer()

        # Synthetic config
        cached_len = 3000
        target_in = 10000
        target_out = 825

        ds_cfg = DatasetConfig(
            source=DatasetSourceConfig(type="file", path=None, file_format="txt"),
            prompt_column=None,
            image_column=None,
            synthetic=True,
            synthetic_input_length=target_in,
            synthetic_output_length=target_out,
            synthetic_cached_input_length=cached_len,
        )

        sampler = TextSampler(
            tokenizer=tokenizer,
            model="mock_model",
            output_modality="text",
            data=["irrelevant"],
            dataset_config=ds_cfg,
        )

        # Deterministic scenario D(10000,825)
        class Deterministic:
            scenario_type = NormalDistribution.scenario_type  # irrelevant for validation in this path
            def sample(self):
                return target_in, target_out

        req = sampler.sample(Deterministic())

        # Exact token length via tokenizer re-encode of final prompt
        self.assertEqual(req.num_prefill_tokens, target_in)

        # Cached prefix check: leading repeats of base phrase
        base_repeat = cached_len // len(tokenizer.base_tokens)
        expected_prefix = tokenizer.base_phrase * base_repeat
        self.assertTrue(
            req.prompt.startswith(expected_prefix),
            "Prompt does not start with expected cached prefix",
        )

        # Ensure marker and tail exist after the prefix
        remaining = req.prompt[len(expected_prefix) :]
        self.assertIn("MARK", remaining)
        self.assertIn(tokenizer.tail_text, remaining)

    def test_sample_text_exact_token_count(self):
        """
        Test that _sample_text returns text with exact number of tokens
        requested.
        """

        # Set up consistent tokenization behavior
        # Each line in test_data has a predictable token count
        def mock_encode(text, add_special_tokens=False):
            # Map our test lines to token counts
            token_map = {
                "Test line 1": [0, 1, 2],  # 3 tokens
                "Test line 2": [0, 1],  # 2 tokens
                "Test line 3": [0, 1, 2, 3],  # 4 tokens
            }
            # For decoded text (when truncated)
            if text in token_map:
                return token_map[text]
            else:
                # For decoded truncated text, return tokens based on length
                words = text.split()
                return list(range(len(words)))

        self.tokenizer.encode.side_effect = mock_encode
        # Decode returns a string with same number of words as tokens
        self.tokenizer.decode.side_effect = (
            lambda tokens, skip_special_tokens=True: " ".join(["word"] * len(tokens))
        )

        # Test requesting exact token counts
        test_cases = [2, 3, 5, 7]

        for num_tokens in test_cases:
            result = self.sampler._sample_text(num_tokens)

            # Count actual tokens in result
            # Need to handle mixed content (original lines + decoded text)
            total_tokens = 0
            # Split by our test lines to count tokens properly
            remaining = result
            for line in self.test_data:
                if line in remaining:
                    total_tokens += len(mock_encode(line))
                    remaining = remaining.replace(line, "", 1)

            # Any remaining text is decoded text
            if remaining:
                total_tokens += len(remaining.split())

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
