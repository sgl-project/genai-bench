from typing import List, Set, Tuple, Union

from datasets import load_dataset
from PIL.Image import Image

from genai_bench.sampling.dataset_loader import DatasetFormat, DatasetLoader


class TextDatasetLoader(DatasetLoader):
    """
    This datasetLoader is responsible for loading prompts from a data source.
    """

    supported_formats: Set[DatasetFormat] = {
        DatasetFormat.TEXT,
        DatasetFormat.HUGGINGFACE_HUB,
        DatasetFormat.CSV,
    }
    media_type = "Text"

    def _load_txt_file(self, file_path: str) -> List[str]:
        dataset = load_dataset("text", data_files=file_path)

        assert len(dataset) > 0, f"No data found in {file_path}"

        # Huggingface dataset loader lods the entire text content into "train" dataset
        return dataset["train"]["text"]

    def _load_csv_file(self, file_path: str, column_index: int = 0) -> List[str]:
        # TODO: Allow option to set whether or not csv has column names
        dataset = load_dataset("csv", data_files=file_path, header=None)

        assert len(dataset) > 0, f"No data found in {file_path}"

        if column_index >= len(dataset["train"].features):
            raise ValueError(
                f"Column index '{column_index}' is out of bounds for this "
                f"CSV file, which has {dataset['train'].features.keys()} columns."
            )
        # huggingface keeps columns indexes as strings and rows as integer
        return dataset["train"][str(column_index)]

    def _load_request_impl(self) -> Union[List[str], List[Tuple[str, Image]]]:
        # TODO: Use polymorphism if this grows
        dataset_path = self.dataset_config.dataset_path
        dataset_type = self.dataset_config.dataset_path.type
        hf_prompt_column_name = self.dataset_config.hf_prompt_column_name
        column_index = self.dataset_config.dataset_prompt_column_index
        if dataset_type is DatasetFormat.TEXT:
            return self._load_txt_file(dataset_path.path)
        elif dataset_type is DatasetFormat.CSV:
            return self._load_csv_file(dataset_path.path, column_index)
        else:
            self.dataset_config.hf_image_column_name = (
                None  # text-dataset has no image.
            )
            dataset = self._load_huggingface_hub()
            return dataset[hf_prompt_column_name]
