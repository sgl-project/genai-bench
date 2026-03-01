from typing import Any, Set

from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict
from datasets import IterableDataset as HFIterableDataset
from datasets import IterableDatasetDict as HFIterableDatasetDict

from genai_bench.data.loaders.base import DatasetFormat, DatasetLoader
from genai_bench.logging import init_logger

logger = init_logger(__name__)


class VideoDatasetLoader(DatasetLoader):
    """
    This datasetLoader is responsible for loading videos and prompts from a data source.
    """

    supported_formats: Set[DatasetFormat] = {
        DatasetFormat.HUGGINGFACE_HUB,
    }
    media_type = "Video"

    def _process_loaded_data(self, data: Any) -> Any:
        """Normalize HF datasets and avoid eager iteration.

        - If it's a `datasets.Dataset`, return it (supports len and __getitem__).
        - If it's a `datasets.DatasetDict`, select a split (prefer 'train' or the
          first available) and return that `Dataset`.
        - If it's a streaming dataset (`IterableDataset` or `IterableDatasetDict`),
          raise an error instructing to disable streaming.
        - Otherwise, pass-through.
        """
        if isinstance(data, HFDataset):
            return data
        if isinstance(data, HFDatasetDict):
            available_splits = list(data.keys())
            if not available_splits:
                raise ValueError(
                    "HuggingFace DatasetDict has no splits to select from."
                )
            chosen_split = "train" if "train" in data else available_splits[0]
            if len(available_splits) > 1 and chosen_split != "train":
                logger.warning(
                    "Multiple splits found %s; defaulting to '%s'. "
                    "Set config.huggingface_kwargs.split to control this.",
                    available_splits,
                    chosen_split,
                )
            return data[chosen_split]
        if isinstance(data, (HFIterableDataset, HFIterableDatasetDict)):
            raise ValueError(
                "Streaming datasets are not supported for video sampling. "
                "Load without streaming (streaming=False) and provide a concrete split."
            )
        return data
