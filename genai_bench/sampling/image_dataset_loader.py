from typing import List, Tuple, Union

from PIL.Image import Image

from genai_bench.sampling.dataset_loader import DatasetFormat, DatasetLoader


class ImageDatasetLoader(DatasetLoader):
    """
    This datasetLoader is responsible for loading images and prompts from a data source.
    """

    supported_formats = {DatasetFormat.HUGGINGFACE_HUB}
    media_type = "Image"

    def _load_request_impl(self) -> Union[List[str], List[Tuple[str, Image]]]:
        hf_image_column_name = self.dataset_config.hf_image_column_name
        hf_prompt_column_name = self.dataset_config.hf_prompt_column_name
        sampled_requests: List[Tuple[str, Image]] = []
        dataset = self._load_huggingface_hub()
        for data in dataset:
            mm_content = data[hf_image_column_name]
            prompt = (
                data[hf_prompt_column_name] if hf_prompt_column_name is not None else ""
            )
            sampled_requests.append((prompt, mm_content))
        return sampled_requests
