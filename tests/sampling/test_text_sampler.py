import unittest
from unittest.mock import MagicMock, patch

from genai_bench.protocol import (
    UserChatRequest,
    UserEmbeddingRequest,
    UserReRankRequest,
)
from genai_bench.sampling import EmbeddingScenario, NormalDistribution
from genai_bench.sampling.text_sampler import (
    DatasetConfig,
    DatasetFormat,
    DatasetPath,
    TextSampler,
)
from genai_bench.sampling.text_scenario import ReRankScenario


class TestTextSampler(unittest.TestCase):
    @patch("genai_bench.sampling.text_sampler.TextDatasetLoader")
    @patch("genai_bench.sampling.text_sampler.init_logger")
    def setUp(self, mock_logger, mock_text_loader):
        # Mocking the dataset loader and logger
        self.mock_text_loader = mock_text_loader.return_value
        self.mock_text_loader.load_request.return_value = ["Test line 1"]

        self.dataset_config = DatasetConfig(
            dataset_path=DatasetPath(path="sonnet.txt", type=DatasetFormat.TEXT),
            hf_prompt_column_name="prompt",
            hf_image_column_name="image",
            hf_split=None,
            hf_subset=None,
            hf_revision=None,
            dataset_prompt_column_index=0,
        )
        self.embedding_config = DatasetConfig(
            dataset_path=DatasetPath(
                path="datasetId", type=DatasetFormat.HUGGINGFACE_HUB
            ),
            hf_prompt_column_name="prompt",
            hf_image_column_name="image",
            hf_split=None,
            hf_subset=None,
            hf_revision=None,
            dataset_prompt_column_index=0,
        )
        self.tokenizer = MagicMock()
        self.model = "mock_model"
        self.output_modality = "text"
        self.sampler = TextSampler(
            tokenizer=self.tokenizer,
            model=self.model,
            output_modality=self.output_modality,
            dataset_config=self.dataset_config,
        )
        self.embedding_sampler = TextSampler(
            tokenizer=self.tokenizer,
            model=self.model,
            output_modality="embeddings",
            dataset_config=self.embedding_config,
        )
        dataset_config = DatasetConfig(
            dataset_path=DatasetPath(
                path="hugginfaceId", type=DatasetFormat.HUGGINGFACE_HUB
            ),
            hf_prompt_column_name="prompt",
            hf_image_column_name="image",
            hf_split=None,
            hf_subset=None,
            hf_revision=None,
            dataset_prompt_column_index=0,
        )
        self.hugginface_sampler = TextSampler(
            tokenizer=self.tokenizer,
            model=self.model,
            output_modality="text",
            dataset_config=dataset_config,
        )
        self.invalid_modality_sampler = TextSampler(
            tokenizer=self.tokenizer,
            model=self.model,
            output_modality="invalid",
            dataset_config=dataset_config,
        )
        self.rerank_sampler = TextSampler(
            tokenizer=self.tokenizer,
            model=self.model,
            output_modality="rerank",
            dataset_config=self.dataset_config,
        )

    def test_sample_chat_request(self):
        # Test for sampling a chat request
        scenario = NormalDistribution(5, 0, 5, 0)
        with patch.object(self.sampler, "get_token_length", side_effect=[10, 5]):
            result = self.sampler.sample(scenario)

            self.assertIsInstance(result, UserChatRequest)
            self.assertEqual(result.model, self.model)
            self.assertTrue(isinstance(result.prompt, str))
            self.assertEqual(result.prompt, "Test ")
            self.assertGreater(len(result.prompt), 0)

    def test_sample_chat_request_with_dataset(self):
        # Test for sampling a chat request
        scenario = NormalDistribution(5, 0, 5, 0)

        with patch.object(
            self.hugginface_sampler, "get_token_length", side_effect=[10, 5]
        ):
            result = self.hugginface_sampler.sample(scenario)

            self.assertIsInstance(result, UserChatRequest)
            self.assertEqual(result.model, self.model)
            self.assertTrue(isinstance(result.prompt, str))
            self.assertEqual(result.prompt, "Test line 1")
            self.assertGreater(len(result.prompt), 0)

    def test_sample_embedding_request(self):
        # Test for sampling an embedding request
        scenario = EmbeddingScenario(100)

        with patch.object(
            self.embedding_sampler, "get_token_length", side_effect=[50, 50, 500]
        ):
            result = self.embedding_sampler.sample(scenario)

            self.assertIsInstance(result, UserEmbeddingRequest)
            self.assertEqual(result.model, self.model)
            self.assertEqual(
                len(result.documents), 1
            )  # Single document due to default batch size
            self.assertTrue(isinstance(result.documents[0], str))

    def test_validate_scenario_invalid(self):
        # Test for scenario validation with an invalid scenario type
        scenario = EmbeddingScenario(100)

        with self.assertRaises(ValueError):
            self.sampler.sample(scenario)

    def test_validate_scenario_invalid2(self):
        # Test for scenario validation with an invalid scenario type
        scenario = NormalDistribution(100, 0, 100, 0)

        with self.assertRaises(ValueError):
            self.embedding_sampler.sample(scenario)

    def test_invalid_modality(self):
        # Test for scenario validation with an invalid scenario type
        scenario = NormalDistribution(100, 0, 100, 0)

        with self.assertRaises(ValueError):
            self.invalid_modality_sampler.sample(scenario)

    def test_check_discrepancy(self):
        scenario = NormalDistribution(10, 0, 10, 0)

        # Test for checking token count discrepancies
        with (
            patch.object(self.sampler, "get_token_length", side_effect=[5, 5, 500]),
            patch("genai_bench.sampling.text_sampler.logger") as mock_logger,
        ):
            self.sampler.sample(scenario)
            mock_logger.warning.assert_called_once()

    def test_sample_rerank_request(self):
        # Test for sampling an embedding request
        scenario = ReRankScenario(100, 100)

        with patch.object(self.rerank_sampler, "get_token_length", return_value=100):
            result = self.rerank_sampler.sample(scenario)

            self.assertIsInstance(result, UserReRankRequest)
            self.assertEqual(result.model, self.model)
            self.assertEqual(
                len(result.documents), 1
            )  # Single document due to default batch size
            self.assertTrue(isinstance(result.documents[0], str))
