"""Factory for creating appropriate data loaders and loading data."""

from pathlib import Path
from typing import List, Tuple, Union, cast

from PIL.Image import Image

from genai_bench.data.config import DatasetConfig, DatasetSourceConfig
from genai_bench.data.loaders.image import ImageDatasetLoader
from genai_bench.data.loaders.text import TextDatasetLoader
from genai_bench.logging import init_logger

logger = init_logger(__name__)


class DataLoaderFactory:
    """Factory for creating data loaders and loading data."""

    @staticmethod
    def load_data_for_task(
        task: str, dataset_config: DatasetConfig
    ) -> Tuple[Union[List[str], List[Tuple[str, Image]]], bool]:
        """Load data for a specific task.

        Args:
            task: Task name in format "input-to-output"
            dataset_config: Dataset configuration

        Returns:
            Tuple of (loaded_data, use_scenario)
        """
        input_modality, output_modality = task.split("-to-")

        if input_modality == "text":
            return DataLoaderFactory._load_text_data(dataset_config, output_modality)
        elif input_modality == "image":
            return DataLoaderFactory._load_image_data(dataset_config), False
        else:
            raise ValueError(f"Unsupported input modality: {input_modality}")

    @staticmethod
    def _load_text_data(
        dataset_config: DatasetConfig, output_modality: str
    ) -> Tuple[List[str], bool]:
        """Load text data with embeddings restrictions."""
        # Handle embeddings restriction for non-sonnet datasets
        is_sonnet = (
            "sonnet" in dataset_config.source.path
            if dataset_config.source.path
            else False
        )

        if not is_sonnet and output_modality == "embeddings":
            logger.warning(
                "Embeddings currently do not support user-specified datasets. "
                "Using the default `sonnet.txt` dataset."
            )
            # Create a new config with default sonnet path
            sonnet_path = str(Path(__file__).parent.parent / "sonnet.txt")
            dataset_config = DatasetConfig(
                source=DatasetSourceConfig(
                    type="file", path=sonnet_path, file_format="txt"
                )  # type: ignore[call-arg]
            )

        loader = TextDatasetLoader(dataset_config)
        data = loader.load_request()

        # TextDatasetLoader always returns List[str]
        text_data = cast(List[str], data)

        # Determine if we should use scenario-based sampling
        use_scenario = (
            dataset_config.source.type == "file"
            and dataset_config.source.file_format == "txt"
        )

        return text_data, use_scenario

    @staticmethod
    def _load_image_data(dataset_config: DatasetConfig) -> List[Tuple[str, Image]]:
        """Load image data."""
        loader = ImageDatasetLoader(dataset_config)
        data = loader.load_request()
        # ImageDatasetLoader always returns List[Tuple[str, Image]]
        return data  # type: ignore
