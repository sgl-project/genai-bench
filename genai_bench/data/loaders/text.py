from typing import Any, List, Set

from genai_bench.data.loaders.base import DatasetFormat, DatasetLoader


class TextDatasetLoader(DatasetLoader):
    """
    This datasetLoader is responsible for loading prompts from a data source.
    """

    supported_formats: Set[DatasetFormat] = {
        DatasetFormat.TEXT,
        DatasetFormat.CSV,
        DatasetFormat.JSON,
        DatasetFormat.HUGGINGFACE_HUB,
    }
    media_type = "Text"

    def _process_loaded_data(self, data: Any) -> List[str]:
        """Process data loaded from dataset source."""
        # Handle data from dataset sources
        if isinstance(data, list):
            return data
        # Handle HuggingFace datasets
        prompt_column = self.dataset_config.prompt_column
        if hasattr(data, "__getitem__") and prompt_column:
            return data[prompt_column]
        else:
            raise ValueError(f"Cannot extract prompts from data: {type(data)}")
