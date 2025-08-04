from typing import Any, List, Set, Tuple

from PIL.Image import Image

from genai_bench.data.loaders.base import DatasetFormat, DatasetLoader
from genai_bench.logging import init_logger

logger = init_logger(__name__)


class ImageDatasetLoader(DatasetLoader):
    """
    This datasetLoader is responsible for loading images and prompts from a data source.
    """

    supported_formats: Set[DatasetFormat] = {
        DatasetFormat.HUGGINGFACE_HUB,
    }
    media_type = "Image"

    def _process_loaded_data(self, data: Any) -> List[Tuple[str, Image]]:
        """Process data loaded from dataset source with flexible configuration."""
        sampled_requests: List[Tuple[str, Image]] = []
        config = self.dataset_config

        try:
            for item in data:
                image = item.get(config.image_column) if config.image_column else None
                if not image:
                    continue

                if config.prompt_lambda:
                    prompt_func = eval(config.prompt_lambda)
                    prompt = prompt_func(item)
                elif config.prompt_column:
                    prompt = item[config.prompt_column]
                else:
                    prompt = ""

                sampled_requests.append((prompt, image))

        except (ValueError, KeyError) as e:
            raise ValueError(
                f"Cannot extract image data from dataset: {type(data)}, error: {str(e)}"
            ) from e
        return sampled_requests
