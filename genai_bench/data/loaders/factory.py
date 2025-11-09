"""Factory for creating appropriate data loaders and loading data. test test"""

from typing import Any, List, Tuple, Union, cast

from genai_bench.data.config import DatasetConfig
from genai_bench.data.loaders.image import ImageDatasetLoader
from genai_bench.data.loaders.text import TextDatasetLoader
from genai_bench.logging import init_logger

logger = init_logger(__name__)


class DataLoaderFactory:
    """Factory for creating data loaders and loading data."""

    @staticmethod
    def load_data_for_task(
        task: str, dataset_config: DatasetConfig
    ) -> Union[List[str], List[Tuple[str, Any]]]:
        """Load data for a specific task.

        Args:
            task: Task name in format "input-to-output"
            dataset_config: Dataset configuration

        Returns:
            Loaded data for the task
        """
        input_modality, output_modality = task.split("-to-")

        if input_modality == "text":
            return DataLoaderFactory._load_text_data(dataset_config, output_modality)
        elif "image" in input_modality:
            return DataLoaderFactory._load_image_data(dataset_config)
        else:
            raise ValueError(f"Unsupported input modality: {input_modality}")

    @staticmethod
    def _load_text_data(
        dataset_config: DatasetConfig, output_modality: str
    ) -> List[str]:
        """Load text data."""

        loader = TextDatasetLoader(dataset_config)
        data = loader.load_request()

        # TextDatasetLoader always returns List[str]
        text_data = cast(List[str], data)
        return text_data

    @staticmethod
    def _load_image_data(
        dataset_config: DatasetConfig,
    ) -> List[Tuple[str, Any]]:
        """Load image data."""
        loader = ImageDatasetLoader(dataset_config)
        data = loader.load_request()
        return data  # type: ignore
