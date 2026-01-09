"""Dataset source implementations for flexible dataset loading."""

import importlib
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Tuple, Union

import pandas as pd
from datasets import load_dataset
from datasets.exceptions import DatasetNotFoundError
from huggingface_hub import dataset_info

from genai_bench.data.config import DatasetSourceConfig
from genai_bench.logging import init_logger

logger = init_logger(__name__)


class DatasetSource(ABC):
    """Abstract base class for dataset sources."""

    def __init__(self, config: DatasetSourceConfig):
        self.config = config

    @abstractmethod
    def load(self) -> Any:
        """Load dataset from source.

        Returns:
            The loaded dataset in a format suitable for the sampler.
        """
        pass


class FileDatasetSource(DatasetSource):
    """Load datasets from local files (txt, csv, json)."""

    def load(self) -> Union[List[str], List[Tuple[str, Any]]]:
        """Load data from local file."""
        if not self.config.path:
            raise ValueError("File path is required for file dataset source")

        file_path = Path(self.config.path)
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        file_format = self.config.file_format or file_path.suffix.lstrip(".")

        if file_format == "txt":
            return self._load_text_file(file_path)
        elif file_format == "csv":
            return self._load_csv_file(file_path)
        elif file_format == "json":
            return self._load_json_file(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

    def _load_text_file(self, file_path: Path) -> List[str]:
        """Load text file line by line."""
        logger.info(f"Loading text file: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(lines)} lines from text file")
        return lines

    def _load_csv_file(self, file_path: Path) -> Any:
        """Load CSV file as a dictionary-like structure to support column names."""
        logger.info(f"Loading CSV file: {file_path}")

        # For CSV files, we'll return a pandas DataFrames
        df = pd.read_csv(file_path)
        logger.info(
            f"Loaded {len(df)} rows from CSV file with columns: {list(df.columns)}"
        )
        # Return a dict-like structure that supports column access
        return df.to_dict(orient="list")

    def _load_json_file(self, file_path: Path) -> List[Any]:
        """Load JSON file."""
        import json

        logger.info(f"Loading JSON file: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            logger.info(f"Loaded {len(data)} items from JSON file")
            return data
        else:
            raise ValueError(f"JSON file must contain a list, got {type(data)}")


class HuggingFaceDatasetSource(DatasetSource):
    """Load datasets from HuggingFace Hub with full flexibility."""

    def load(self) -> Any:
        """Load dataset from HuggingFace Hub."""
        if not self.config.path:
            raise ValueError("Dataset ID is required for HuggingFace dataset source")

        logger.info(f"Loading HuggingFace dataset: {self.config.path}")
        if self.config.huggingface_kwargs:
            logger.info(f"Using HuggingFace kwargs: {self.config.huggingface_kwargs}")

        kwargs = self.config.huggingface_kwargs or {}
        data_dir = Path(self.config.path).expanduser().resolve()
        is_local_data = data_dir.exists() and data_dir.is_dir()
        if not is_local_data:
            # Verify dataset exists
            try:
                dataset_info(self.config.path, token=os.environ.get("HF_TOKEN"))
            except DatasetNotFoundError as e:
                raise ValueError(
                    f"Dataset '{self.config.path}' not found on HuggingFace Hub. "
                    f"If it's a gated repo, please set HF_TOKEN environment "
                    f"variable."
                ) from e

        dataset = load_dataset(self.config.path, **kwargs)

        logger.info(f"Successfully loaded HuggingFace dataset: {self.config.path}")
        return dataset


class CustomDatasetSource(DatasetSource):
    """Load datasets using custom loader classes."""

    def load(self) -> Any:
        """Load dataset using custom loader."""
        if not self.config.loader_class:
            raise ValueError("Loader class is required for custom dataset source")

        # Import the custom loader class
        try:
            module_path, class_name = self.config.loader_class.rsplit(".", 1)
            module = importlib.import_module(module_path)
            loader_class = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(
                f"Failed to import custom loader class "
                f"'{self.config.loader_class}': {e}"
            ) from e

        # Instantiate and use the loader
        kwargs = self.config.loader_kwargs or {}
        logger.info(f"Loading dataset with custom loader: {self.config.loader_class}")

        loader = loader_class(**kwargs)
        if hasattr(loader, "load"):
            return loader.load()
        else:
            raise AttributeError(
                f"Custom loader class '{self.config.loader_class}' must have a 'load' "
                f"method"
            )


class DatasetSourceFactory:
    """Factory to create appropriate dataset sources."""

    _sources = {
        "file": FileDatasetSource,
        "huggingface": HuggingFaceDatasetSource,
        "custom": CustomDatasetSource,
    }

    @classmethod
    def create(cls, config: DatasetSourceConfig) -> DatasetSource:
        """Create a dataset source based on configuration.

        Args:
            config: Dataset source configuration

        Returns:
            DatasetSource instance

        Raises:
            ValueError: If source type is not supported
        """
        source_type = config.type.lower()

        if source_type not in cls._sources:
            raise ValueError(
                f"Unknown dataset source type: {source_type}. "
                f"Supported types: {list(cls._sources.keys())}"
            )

        source_class = cls._sources[source_type]
        return source_class(config)  # type: ignore

    @classmethod
    def register_source(cls, source_type: str, source_class: type[DatasetSource]):
        """Register a new dataset source type.

        Args:
            source_type: Name of the source type
            source_class: Class implementing DatasetSource
        """
        cls._sources[source_type] = source_class
