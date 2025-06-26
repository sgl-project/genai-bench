from typing import Any, List, Set, Tuple

from PIL.Image import Image

from genai_bench.data.loaders.base import DatasetFormat, DatasetLoader


class ImageDatasetLoader(DatasetLoader):
    """
    This datasetLoader is responsible for loading images and prompts from a data source.
    """

    supported_formats: Set[DatasetFormat] = {
        DatasetFormat.HUGGINGFACE_HUB,
    }
    media_type = "Image"

    def _process_loaded_data(self, data: Any) -> List[Tuple[str, Image]]:
        """Process data loaded from dataset source."""
        sampled_requests: List[Tuple[str, Image]] = []
        image_column = self.dataset_config.image_column
        prompt_column = self.dataset_config.prompt_column

        for item in data:
            image = item[image_column] if image_column else None
            prompt = item[prompt_column] if prompt_column else ""
            if image:
                sampled_requests.append((prompt, image))
        return sampled_requests
