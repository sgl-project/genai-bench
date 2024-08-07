import os
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional, Set, Tuple, Union

from datasets import load_dataset
from datasets.exceptions import DatasetNotFoundError
from huggingface_hub import dataset_info
from PIL.Image import Image
from pydantic import BaseModel

from genai_bench.logging import init_logger

logger = init_logger(__name__)


class DatasetFormat(Enum):
    """
    Format supported for datasets in genai-bench.
    """

    TEXT = ".txt"
    CSV = ".csv"
    HUGGINGFACE_HUB = "huggingface"


class DatasetPath(BaseModel):
    type: DatasetFormat = DatasetFormat.TEXT
    path: str = Path(__file__).with_name("sonnet.txt").as_posix()

    def from_value(value):
        dataset_path = DatasetPath()
        if value is not None:
            try:
                file_path = Path(value)
                if file_path.is_file():
                    dataset_path.type = DatasetFormat(file_path.suffix)
                else:
                    dataset_info(value, token=os.environ.get("HUGGINGFACE_API_KEY"))
                    dataset_path.type = DatasetFormat.HUGGINGFACE_HUB
                dataset_path.path = value
            except DatasetNotFoundError as e:
                raise ValueError(
                    f"Provided `--dataset-path` {value} is nether a local file nor "
                    f"an accessible dataset. If its gated repo, please set "
                    f"HUGGINGFACE_API_KEY environment variable."
                ) from e
        return dataset_path


class DatasetConfig(BaseModel):
    """
    Configurations for loading a dataset.
    """

    dataset_path: DatasetPath
    hf_prompt_column_name: Optional[str]
    hf_image_column_name: Optional[str]
    hf_subset: Optional[str]
    hf_split: Optional[str]
    hf_revision: Optional[str]
    dataset_prompt_column_index: int


class DatasetLoader(ABC):
    """Abstract base class for dataset loader. A dataset loader is responsible for
    loading the data from a source (CSV file, text file or dataset from huggingface hub)
    """

    supported_formats: Set[DatasetFormat] = set()
    media_type: Optional[str] = ""

    def __init__(self, dataset_config):
        self.dataset_config = dataset_config

    def _load_huggingface_hub(self) -> Any:
        huggingface_id = self.dataset_config.dataset_path.path
        hf_subset = self.dataset_config.hf_subset
        hf_split = self.dataset_config.hf_split
        hf_revision = self.dataset_config.hf_revision
        hf_prompt_column_name = self.dataset_config.hf_prompt_column_name
        hf_image_column_name = self.dataset_config.hf_image_column_name
        logger.info(f"Starting download of dataset: {huggingface_id}")
        dataset = load_dataset(
            huggingface_id, name=hf_subset, split=hf_split, revision=hf_revision
        )
        logger.info(f"Finished downing dataset: {huggingface_id}")
        assert len(dataset) > 0, f"No data found in dataset {huggingface_id}"
        if hf_prompt_column_name is not None:
            assert hf_prompt_column_name in dataset.features, (
                f"The column: {hf_prompt_column_name} is not one of "
                f"{dataset.features.keys()}"
            )
        if hf_image_column_name is not None:
            assert hf_image_column_name in dataset.features, (
                f"The column: {hf_image_column_name} is not one of "
                f"{dataset.features.keys()}"
            )
        return dataset

    def load_request(self) -> Union[List[str], List[Tuple[str, Image]]]:
        dataset_type = self.dataset_config.dataset_path.type
        if dataset_type not in self.supported_formats:
            raise ValueError(
                f"{dataset_type} is unsupported for {self.media_type} dataset-path "
                f"Please provide one of {self.supported_formats}"
            )
        return self._load_request_impl()

    @abstractmethod
    def _load_request_impl(self) -> Union[List[str], List[Tuple[str, Image]]]:
        """
        Load, process and return data from data source as per dataset_config.

        Returns: data that can be used by sampler to create request to llm.
        """
        pass
