import re
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
                    prompt = self._safe_eval_prompt(config.prompt_lambda, item)
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

    def _safe_eval_prompt(self, prompt_template: str, item: dict) -> str:
        """
        Safely evaluate lambda expressions using asteval.
        Supports: lambda x: x["conversations"][0]["content"]
        """

        # Handle lambda expressions
        if prompt_template.strip().startswith("lambda"):
            try:
                expr = prompt_template.split(":", 1)[1].strip()
                lambda_part, expr = prompt_template.split(":", 1)
                var_name = lambda_part.replace("lambda", "").strip()
                expr = expr.strip()
                # Replace x with context
                expr = re.sub(rf"\b{re.escape(var_name)}\b", "context", expr)
                # Safe evaluation by restricting allowed functions
                safe_dict = {"context": item, "str": str, "len": len}
                result = eval(expr, {"__builtins__": {}}, safe_dict)
                return str(result) if result is not None else ""
            except Exception:
                return ""

        # Simple field access
        if prompt_template in item:
            return str(item[prompt_template])

        return prompt_template
