from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, List, Set, Tuple, Union

from PIL.Image import Image

from genai_bench.data.config import DatasetConfig
from genai_bench.data.sources import DatasetSourceFactory
from genai_bench.logging import init_logger

logger = init_logger(__name__)


class DatasetFormat(Enum):
    """Format supported for datasets in genai-bench."""

    TEXT = "txt"
    CSV = "csv"
    JSON = "json"
    HUGGINGFACE_HUB = "huggingface"


class DatasetLoader(ABC):
    """Abstract base class for dataset loader. A dataset loader is responsible for
    loading the data from a source using the new dataset configuration system.
    """

    supported_formats: Set[DatasetFormat] = set()
    media_type: str = ""

    def __init__(self, dataset_config: DatasetConfig):
        """Initialize with new dataset configuration."""
        self.dataset_config = dataset_config
        self._validate_source_format()
        self.dataset_source = DatasetSourceFactory.create(dataset_config.source)

    def _validate_source_format(self):
        """Validate that the source format is supported by this loader."""
        source_type = self.dataset_config.source.type

        if source_type == "file":
            file_format = self.dataset_config.source.file_format
            if file_format:
                try:
                    format_enum = DatasetFormat(file_format.lower())
                except ValueError:
                    format_enum = None

                if format_enum and format_enum not in self.supported_formats:
                    raise ValueError(
                        f"File format '{file_format}' is not supported by "
                        f"{self.media_type} loader. Supported formats: "
                        f"{[f.value for f in self.supported_formats]}"
                    )
        elif source_type == "huggingface":
            if DatasetFormat.HUGGINGFACE_HUB not in self.supported_formats:
                raise ValueError(
                    f"HuggingFace datasets are not supported by "
                    f"{self.media_type} loader."
                )

    def load_request(self) -> Union[List[str], List[Tuple[str, Any]]]:
        """Load data from the dataset source."""
        data = self.dataset_source.load()
        return self._process_loaded_data(data)

    @abstractmethod
    def _process_loaded_data(
        self, data: Any
    ) -> Union[List[str], List[Tuple[str, Image]]]:
        """
        Process data loaded from dataset source. Must be implemented by subclasses.
        """
        pass
