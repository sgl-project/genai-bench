"""Dataset configuration models for flexible dataset loading."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator


class DatasetSourceConfig(BaseModel):
    """Configuration for a dataset source.

    Supports multiple dataset source types:
    - file: Local files (txt, csv, json)
    - huggingface: HuggingFace Hub datasets
    - custom: Custom dataset loaders
    """

    type: str = Field(
        ..., description="Dataset source type: 'file', 'huggingface', or 'custom'"
    )
    path: Optional[str] = Field(
        None, description="Path to dataset (file path or HuggingFace ID)"
    )

    # For file sources
    file_format: Optional[str] = Field(
        None, description="File format: 'csv', 'txt', 'json'"
    )

    # For HuggingFace sources - accepts ANY parameter that load_dataset supports
    huggingface_kwargs: Optional[Dict[str, Any]] = Field(
        None,
        description="Keyword arguments passed directly to HuggingFace load_dataset",
    )

    # For custom sources
    loader_class: Optional[str] = Field(
        None, description="Python import path for custom dataset loader"
    )
    loader_kwargs: Optional[Dict[str, Any]] = Field(
        None, description="Keyword arguments for custom loader"
    )

    @field_validator("type")
    def validate_type(cls, v):
        valid_types = {"file", "huggingface", "custom"}
        if v not in valid_types:
            raise ValueError(f"Dataset source type must be one of {valid_types}")
        return v


class DatasetConfig(BaseModel):
    """Complete dataset configuration."""

    source: DatasetSourceConfig
    prompt_column: Optional[str] = Field(
        None, description="Column name containing prompts"
    )
    image_column: Optional[str] = Field(
        None, description="Column name containing images"
    )
    prompt_lambda: Optional[str] = Field(
        None,
        description="Lambda expression string, "
        'e.g. \'lambda item: f"Question: {item["question"]}"\'',
    )
    unsafe_allow_large_images: bool = Field(
        False,
        description="Overrides pillows internal DDOS protection",
    )

    @classmethod
    def from_file(cls, config_path: str) -> "DatasetConfig":
        """Load configuration from a JSON file."""
        with open(config_path, "r") as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def from_cli_args(
        cls,
        dataset_path: Optional[str] = None,
        prompt_column: Optional[str] = None,
        image_column: Optional[str] = None,
        **kwargs,
    ) -> "DatasetConfig":
        """Create configuration from CLI arguments for backward compatibility."""
        if dataset_path is None:
            # Default to built-in sonnet.txt
            dataset_path = str(Path(__file__).parent / "sonnet.txt")
            source_type = "file"
            file_format = "txt"
        else:
            # Determine source type from path
            path = Path(dataset_path)
            if path.exists():
                source_type = "file"
                if path.suffix == ".csv":
                    file_format = "csv"
                elif path.suffix == ".txt":
                    file_format = "txt"
                else:
                    file_format = "json"
            else:
                # Assume it's a HuggingFace ID
                source_type = "huggingface"
                file_format = None

        source_config = DatasetSourceConfig(
            type=source_type,
            path=dataset_path,
            file_format=file_format,
            huggingface_kwargs=None,
            loader_class=None,
            loader_kwargs=None,
        )

        return cls(
            source=source_config,
            prompt_column=prompt_column,
            image_column=image_column,
            prompt_lambda=None,
            unsafe_allow_large_images=False,
        )
