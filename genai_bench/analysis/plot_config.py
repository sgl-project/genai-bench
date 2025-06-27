"""Plot configuration system for flexible, user-defined plots."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator, validator

from genai_bench.logging import init_logger
from genai_bench.metrics.metrics import AggregatedMetrics

logger = init_logger(__name__)


class PlotSpec(BaseModel):
    """Specification for a single plot."""

    title: str = Field(..., description="Title of the plot")
    x_field: str = Field(..., description="Field path for X-axis data")
    y_field: str = Field(..., description="Field path for Y-axis data")
    x_label: Optional[str] = Field(None, description="Custom X-axis label")
    y_label: Optional[str] = Field(None, description="Custom Y-axis label")
    plot_type: str = Field(
        default="line", description="Type of plot: line, scatter, bar"
    )
    position: Tuple[int, int] = Field(..., description="Position in grid (row, col)")

    @field_validator("plot_type")
    def validate_plot_type(cls, v):
        allowed_types = {"line", "scatter", "bar"}
        if v not in allowed_types:
            raise ValueError(f"plot_type must be one of {allowed_types}, got {v}")
        return v


class PlotLayout(BaseModel):
    """Layout configuration for the plot grid."""

    rows: int = Field(default=2, ge=1, le=5, description="Number of rows")
    cols: int = Field(default=4, ge=1, le=6, description="Number of columns")
    figsize: Optional[Tuple[int, int]] = Field(
        None, description="Figure size (width, height)"
    )


class PlotConfig(BaseModel):
    """Complete plot configuration."""

    layout: PlotLayout = Field(default_factory=PlotLayout)
    plots: List[PlotSpec] = Field(..., description="List of plot specifications")

    @validator("plots")
    def validate_plot_positions(cls, v, values):
        if "layout" not in values:
            return v

        layout = values["layout"]
        max_row, max_col = layout.rows - 1, layout.cols - 1

        positions = set()
        for plot in v:
            row, col = plot.position
            if row > max_row or col > max_col:
                raise ValueError(
                    f"Plot position ({row}, {col}) exceeds layout bounds "
                    f"({max_row}, {max_col})"
                )
            if plot.position in positions:
                raise ValueError(f"Duplicate plot position: {plot.position}")
            positions.add(plot.position)

        return v


class PlotConfigManager:
    """Manager for plot configurations."""

    # Built-in presets
    PRESETS = {
        "2x4_default": {
            "layout": {"rows": 2, "cols": 4, "figsize": [32, 12]},
            "plots": [
                {
                    "title": "Output Inference Speed per Request vs "
                    "Output Throughput of Server",
                    "x_field": "mean_output_throughput_tokens_per_s",
                    "y_field": "stats.output_inference_speed.mean",
                    "x_label": "Output Throughput of Server (tokens/s)",
                    "y_label": "Output Inference Speed per Request (tokens/s)",
                    "plot_type": "line",
                    "position": [0, 0],
                },
                {
                    "title": "TTFT vs Output Throughput of Server",
                    "x_field": "mean_output_throughput_tokens_per_s",
                    "y_field": "stats.ttft.mean",
                    "x_label": "Output Throughput of Server (tokens/s)",
                    "y_label": "TTFT",
                    "plot_type": "line",
                    "position": [0, 1],
                },
                {
                    "title": "Mean E2E Latency per Request vs RPS",
                    "x_field": "requests_per_second",
                    "y_field": "stats.e2e_latency.mean",
                    "x_label": "RPS (req/s)",
                    "y_label": "Mean E2E Latency per Request (s)",
                    "plot_type": "line",
                    "position": [0, 2],
                },
                {
                    "title": "Error Rates by HTTP Status vs Concurrency",
                    "x_field": "num_concurrency",
                    "y_field": "error_rate",
                    "x_label": "Concurrency",
                    "y_label": "Error Rate",
                    "plot_type": "bar",
                    "position": [0, 3],
                },
                {
                    "title": "Output Inference Speed per Request vs "
                    "Total Throughput (Input + Output) of Server",
                    "x_field": "mean_total_tokens_throughput_tokens_per_s",
                    "y_field": "stats.output_inference_speed.mean",
                    "x_label": "Total Throughput (Input + Output) of Server (tokens/s)",
                    "y_label": "Output Inference Speed per Request (tokens/s)",
                    "plot_type": "line",
                    "position": [1, 0],
                },
                {
                    "title": "TTFT vs Total Throughput (Input + Output) of Server",
                    "x_field": "mean_total_tokens_throughput_tokens_per_s",
                    "y_field": "stats.ttft.mean",
                    "x_label": "Total Throughput (Input + Output) of Server (tokens/s)",
                    "y_label": "TTFT",
                    "plot_type": "line",
                    "position": [1, 1],
                },
                {
                    "title": "P90 E2E Latency per Request vs RPS",
                    "x_field": "requests_per_second",
                    "y_field": "stats.e2e_latency.p90",
                    "x_label": "RPS (req/s)",
                    "y_label": "P90 E2E Latency per Request (s)",
                    "plot_type": "line",
                    "position": [1, 2],
                },
                {
                    "title": "P99 E2E Latency per Request vs RPS",
                    "x_field": "requests_per_second",
                    "y_field": "stats.e2e_latency.p99",
                    "x_label": "RPS (req/s)",
                    "y_label": "P99 E2E Latency per Request (s)",
                    "plot_type": "line",
                    "position": [1, 3],
                },
            ],
        },
        "simple_2x2": {
            "layout": {"rows": 2, "cols": 2, "figsize": [16, 12]},
            "plots": [
                {
                    "title": "Throughput vs Latency",
                    "x_field": "mean_output_throughput_tokens_per_s",
                    "y_field": "stats.e2e_latency.mean",
                    "plot_type": "line",
                    "position": [0, 0],
                },
                {
                    "title": "RPS vs Error Rate",
                    "x_field": "requests_per_second",
                    "y_field": "error_rate",
                    "plot_type": "scatter",
                    "position": [0, 1],
                },
                {
                    "title": "Concurrency vs TTFT",
                    "x_field": "num_concurrency",
                    "y_field": "stats.ttft.mean",
                    "plot_type": "line",
                    "position": [1, 0],
                },
                {
                    "title": "Throughput Distribution",
                    "x_field": "num_concurrency",
                    "y_field": "mean_total_tokens_throughput_tokens_per_s",
                    "plot_type": "line",
                    "position": [1, 1],
                },
            ],
        },
    }

    @classmethod
    def load_config(cls, config_source: Union[str, Dict, None] = None) -> PlotConfig:
        """Load plot configuration from various sources."""
        if config_source is None:
            # Use default 2x4 preset
            return cls.load_preset("2x4_default")

        if isinstance(config_source, str):
            if config_source in cls.PRESETS:
                # Load preset
                return cls.load_preset(config_source)
            else:
                # Load from file
                return cls.load_from_file(config_source)

        if isinstance(config_source, dict):
            # Load from dictionary
            return PlotConfig(**config_source)

        raise ValueError(f"Invalid config source type: {type(config_source)}")

    @classmethod
    def load_preset(cls, preset_name: str) -> PlotConfig:
        """Load a built-in preset configuration."""
        if preset_name not in cls.PRESETS:
            available = list(cls.PRESETS.keys())
            raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")

        return PlotConfig(**cls.PRESETS[preset_name])

    @classmethod
    def load_from_file(cls, file_path: str) -> PlotConfig:
        """Load configuration from JSON file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Plot config file not found: {file_path}")

        try:
            with open(path, "r") as f:
                config_data = json.load(f)
            return PlotConfig(**config_data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file {file_path}: {e}") from e
        except Exception as e:
            raise ValueError(f"Error loading config from {file_path}: {e}") from e

    @classmethod
    def save_config(cls, config: PlotConfig, file_path: str) -> None:
        """Save configuration to JSON file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(config.model_dump(), f, indent=2)

        logger.info(f"Plot configuration saved to {file_path}")

    @classmethod
    def get_available_fields(cls) -> Dict[str, str]:
        """Get all available fields from AggregatedMetrics."""
        fields = {}

        # Direct fields
        for field_name, field_info in AggregatedMetrics.model_fields.items():
            if field_name != "stats":
                fields[field_name] = field_info.description or field_name

        # Stats fields with statistical measures
        stats_measures = [
            "min",
            "max",
            "mean",
            "stddev",
            "sum",
            "p25",
            "p50",
            "p75",
            "p90",
            "p95",
            "p99",
        ]
        stats_fields = [
            "ttft",
            "tpot",
            "e2e_latency",
            "output_latency",
            "output_inference_speed",
            "num_input_tokens",
            "num_output_tokens",
            "total_tokens",
            "input_throughput",
            "output_throughput",
        ]

        for field in stats_fields:
            for measure in stats_measures:
                field_path = f"stats.{field}.{measure}"
                fields[field_path] = f"{field} {measure}"

        return fields

    @classmethod
    def get_fields_from_data(
        cls, metrics: AggregatedMetrics
    ) -> Dict[str, Tuple[Any, str]]:
        """
        Extract available fields with actual values from real experiment data.

        Returns:
            Dict mapping field_path -> (value, type_name)
        """
        available_fields = {}

        # Direct fields from AggregatedMetrics
        for field_name in AggregatedMetrics.model_fields:
            if field_name == "stats":
                continue

            try:
                value = getattr(metrics, field_name)
                if value is not None:
                    field_type = type(value).__name__
                    available_fields[field_name] = (value, field_type)
            except AttributeError:
                continue

        # Stats fields - check which ones have actual data
        if hasattr(metrics, "stats") and metrics.stats is not None:
            stats = metrics.stats

            # Get available stat fields from the stats object
            for stat_field_name in stats.model_fields:
                try:
                    stat_field = getattr(stats, stat_field_name)
                    if stat_field is not None:
                        # Check each statistical measure
                        for measure_name in stat_field.model_fields:
                            try:
                                measure_value = getattr(stat_field, measure_name)
                                if measure_value is not None:
                                    field_path = (
                                        f"stats.{stat_field_name}.{measure_name}"
                                    )
                                    field_type = type(measure_value).__name__
                                    available_fields[field_path] = (
                                        measure_value,
                                        field_type,
                                    )
                            except AttributeError:
                                continue
                except AttributeError:
                    continue

        return available_fields

    @classmethod
    def validate_field_path(
        cls, field_path: str, sample_metrics: AggregatedMetrics
    ) -> bool:
        """Validate that a field path exists in AggregatedMetrics."""
        try:
            value = cls.get_field_value(sample_metrics, field_path)
            return value is not None
        except Exception:
            return False

    @classmethod
    def get_field_value(cls, metrics: AggregatedMetrics, field_path: str) -> Any:
        """Get value from metrics using field path (e.g., 'stats.ttft.mean')."""
        parts = field_path.split(".")
        value = metrics

        for part in parts:
            if hasattr(value, part):
                value = getattr(value, part)
            else:
                raise AttributeError(f"Field path '{field_path}' not found")

        return value
