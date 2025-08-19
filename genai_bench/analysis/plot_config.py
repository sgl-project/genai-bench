"""Plot configuration system for flexible, user-defined plots."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator

from genai_bench.logging import init_logger
from genai_bench.metrics.metrics import AggregatedMetrics
from genai_bench.time_units import TimeUnitConverter

logger = init_logger(__name__)


class YFieldSpec(BaseModel):
    """Specification for a Y-axis field in multi-line plots."""

    field: str = Field(..., description="Field path for Y-axis data")
    label: Optional[str] = Field(None, description="Custom label for this line")
    color: Optional[str] = Field(None, description="Custom color for this line")
    linestyle: Optional[str] = Field(
        None, description="Line style: '-', '--', '-.', ':'"
    )


class PlotSpec(BaseModel):
    """Specification for a single plot."""

    title: str = Field(..., description="Title of the plot")
    x_field: str = Field(..., description="Field path for X-axis data")
    y_field: Optional[str] = Field(
        None, description="Field path for Y-axis data (single line)"
    )
    y_fields: Optional[List[YFieldSpec]] = Field(
        None, description="Multiple Y-fields for multi-line plots"
    )
    x_label: Optional[str] = Field(None, description="Custom X-axis label")
    y_label: Optional[str] = Field(None, description="Custom Y-axis label")
    plot_type: str = Field(
        default="line", description="Type of plot: line, scatter, bar"
    )
    position: Tuple[int, int] = Field(..., description="Position in grid (row, col)")
    y_scale: Optional[str] = Field(
        None, description="Y-axis scale type: 'linear' (default) or 'log'"
    )

    @field_validator("plot_type")
    def validate_plot_type(cls, v):
        allowed_types = {"line", "scatter", "bar"}
        if v not in allowed_types:
            raise ValueError(f"plot_type must be one of {allowed_types}, got {v}")
        return v

    @field_validator("y_scale")
    def validate_y_scale(cls, v):
        if v is not None:
            allowed_scales = {"linear", "log"}
            if v not in allowed_scales:
                raise ValueError(f"y_scale must be one of {allowed_scales}, got {v}")
        return v

    @field_validator("y_fields")
    def validate_y_fields(cls, v, info):
        """Ensure either y_field or y_fields is specified, but not both."""
        y_field = info.data.get("y_field") if hasattr(info, "data") else None

        if y_field is not None and v is not None:
            raise ValueError(
                "Cannot specify both y_field and y_fields. Use one or the other."
            )

        if y_field is None and (v is None or len(v) == 0):
            raise ValueError(
                "Must specify either y_field or y_fields with at least one field."
            )

        return v

    def get_y_field_specs(self) -> List[YFieldSpec]:
        """Get Y-field specifications, converting single y_field if needed."""
        if self.y_fields is not None:
            return self.y_fields
        elif self.y_field is not None:
            return [YFieldSpec(field=self.y_field)]  # type: ignore[call-arg]
        else:
            raise ValueError("No Y-field specifications found")

    def is_multi_line(self) -> bool:
        """Check if this plot has multiple Y-fields."""
        return self.y_fields is not None and len(self.y_fields) > 1


class PlotLayout(BaseModel):
    """Layout configuration for the plot grid."""

    rows: int = Field(default=2, ge=1, le=5, description="Number of rows")
    cols: int = Field(default=4, ge=1, le=6, description="Number of columns")
    figsize: Optional[Tuple[int, int]] = Field(
        None, description="Figure size (width, height)"
    )


class PlotConfig(BaseModel):
    """Complete plot configuration."""

    layout: PlotLayout = Field(default_factory=lambda: PlotLayout())  # type: ignore[call-arg]
    plots: List[PlotSpec] = Field(..., description="List of plot specifications")

    @field_validator("plots")
    def validate_plot_positions(cls, v, info):
        if not hasattr(info, "data") or "layout" not in info.data:
            return v

        layout = info.data["layout"]
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
        "multi_line_latency": {
            "layout": {"rows": 2, "cols": 2, "figsize": [16, 12]},
            "plots": [
                {
                    "title": "Latency Percentiles Comparison",
                    "x_field": "requests_per_second",
                    "y_fields": [
                        {
                            "field": "stats.e2e_latency.mean",
                            "label": "Mean Latency",
                            "color": "blue",
                            "linestyle": "-",
                        },
                        {
                            "field": "stats.e2e_latency.p90",
                            "label": "P90 Latency",
                            "color": "orange",
                            "linestyle": "--",
                        },
                        {
                            "field": "stats.e2e_latency.p99",
                            "label": "P99 Latency",
                            "color": "red",
                            "linestyle": "-.",
                        },
                    ],
                    "x_label": "Requests Per Second",
                    "y_label": "Latency (s)",
                    "plot_type": "line",
                    "position": [0, 0],
                },
                {
                    "title": "TTFT Performance Analysis",
                    "x_field": "mean_output_throughput_tokens_per_s",
                    "y_fields": [
                        {
                            "field": "stats.ttft.mean",
                            "label": "Mean TTFT",
                            "color": "green",
                        },
                        {
                            "field": "stats.ttft.p95",
                            "label": "P95 TTFT",
                            "color": "purple",
                        },
                    ],
                    "x_label": "Output Throughput (tokens/s)",
                    "y_label": "Time to First Token (s) (log scale)",
                    "plot_type": "line",
                    "position": [0, 1],
                    "y_scale": "log",
                },
                {
                    "title": "Throughput Components",
                    "x_field": "num_concurrency",
                    "y_fields": [
                        {
                            "field": "mean_output_throughput_tokens_per_s",
                            "label": "Output Throughput",
                        },
                        {
                            "field": "mean_total_tokens_throughput_tokens_per_s",
                            "label": "Total Throughput",
                        },
                    ],
                    "x_label": "Concurrency Level",
                    "y_label": "Throughput (tokens/s)",
                    "plot_type": "line",
                    "position": [1, 0],
                },
                {
                    "title": "Request Success Rate",
                    "x_field": "num_concurrency",
                    "y_field": "error_rate",
                    "x_label": "Concurrency Level",
                    "y_label": "Error Rate",
                    "plot_type": "bar",
                    "position": [1, 1],
                },
            ],
        },
        "single_scenario_analysis": {
            "layout": {"rows": 2, "cols": 2, "figsize": [16, 12]},
            "plots": [
                {
                    "title": "Latency Percentiles vs RPS",
                    "x_field": "requests_per_second",
                    "y_fields": [
                        {
                            "field": "stats.e2e_latency.mean",
                            "label": "Mean",
                            "color": "blue",
                            "linestyle": "-",
                        },
                        {
                            "field": "stats.e2e_latency.p90",
                            "label": "P90",
                            "color": "orange",
                            "linestyle": "--",
                        },
                        {
                            "field": "stats.e2e_latency.p99",
                            "label": "P99",
                            "color": "red",
                            "linestyle": "-.",
                        },
                    ],
                    "x_label": "Requests Per Second",
                    "y_label": "E2E Latency (s)",
                    "plot_type": "line",
                    "position": [0, 0],
                },
                {
                    "title": "TTFT Analysis",
                    "x_field": "mean_output_throughput_tokens_per_s",
                    "y_fields": [
                        {
                            "field": "stats.ttft.mean",
                            "label": "Mean TTFT",
                            "color": "green",
                        },
                        {
                            "field": "stats.ttft.p95",
                            "label": "P95 TTFT",
                            "color": "purple",
                        },
                    ],
                    "x_label": "Output Throughput (tokens/s)",
                    "y_label": "TTFT (s)",
                    "plot_type": "line",
                    "position": [0, 1],
                },
                {
                    "title": "Throughput vs Concurrency",
                    "x_field": "num_concurrency",
                    "y_field": "mean_output_throughput_tokens_per_s",
                    "x_label": "Concurrency",
                    "y_label": "Output Throughput (tokens/s)",
                    "plot_type": "line",
                    "position": [1, 0],
                },
                {
                    "title": "Error Rate vs Concurrency",
                    "x_field": "num_concurrency",
                    "y_field": "error_rate",
                    "x_label": "Concurrency",
                    "y_label": "Error Rate",
                    "plot_type": "bar",
                    "position": [1, 1],
                },
            ],
        },
    }

    @classmethod
    def apply_time_unit_conversion(cls, config_data: Dict[str, Any], time_unit: str = "s") -> Dict[str, Any]:
        """
        Apply time unit conversion to plot configuration labels.
        
        Args:
            config_data: Plot configuration data dictionary
            time_unit: Target time unit ('s' or 'ms')
            
        Returns:
            Configuration data with converted labels
        """
        if time_unit == "s":
            return config_data  # No conversion needed
            
        # Deep copy to avoid modifying original
        converted_config = config_data.copy()
        
        if "plots" in converted_config:
            converted_config["plots"] = []
            for plot in config_data["plots"]:
                plot_copy = plot.copy()
                
                # Convert title if it contains time notation
                if "title" in plot_copy:
                    plot_copy["title"] = TimeUnitConverter.get_unit_label(plot_copy["title"], time_unit)
                
                # Convert y_label if it contains time notation
                if "y_label" in plot_copy:
                    plot_copy["y_label"] = TimeUnitConverter.get_unit_label(plot_copy["y_label"], time_unit)
                
                # Convert x_label if it contains time notation
                if "x_label" in plot_copy:
                    plot_copy["x_label"] = TimeUnitConverter.get_unit_label(plot_copy["x_label"], time_unit)
                
                # Convert y_fields labels if present
                if "y_fields" in plot_copy:
                    for y_field in plot_copy["y_fields"]:
                        if "label" in y_field:
                            y_field["label"] = TimeUnitConverter.get_unit_label(y_field["label"], time_unit)
                
                converted_config["plots"].append(plot_copy)
        
        return converted_config

    @classmethod
    def load_config(cls, config_source: Union[str, Dict, None] = None, time_unit: str = "s") -> PlotConfig:
        """Load plot configuration from various sources."""
        if config_source is None:
            # Use default 2x4 preset
            return cls.load_preset("2x4_default", time_unit)

        if isinstance(config_source, str):
            if config_source in cls.PRESETS:
                # Load preset
                return cls.load_preset(config_source, time_unit)
            else:
                # Load from file
                return cls.load_from_file(config_source, time_unit)

        if isinstance(config_source, dict):
            # Load from dictionary
            converted_data = cls.apply_time_unit_conversion(config_source, time_unit)
            return PlotConfig(**converted_data)

        raise ValueError(f"Invalid config source type: {type(config_source)}")

    @classmethod
    def load_preset(cls, preset_name: str, time_unit: str = "s") -> PlotConfig:
        """Load a built-in preset configuration."""
        if preset_name not in cls.PRESETS:
            available = list(cls.PRESETS.keys())
            raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")

        preset_data = cls.PRESETS[preset_name]
        # Apply time unit conversion to preset data
        converted_data = cls.apply_time_unit_conversion(preset_data, time_unit)
        
        layout_data = converted_data["layout"]
        plots_data = converted_data["plots"]

        # Type ignore for mypy issues with preset data structure
        layout = PlotLayout(**layout_data)  # type: ignore[arg-type]
        plots = [PlotSpec(**plot_data) for plot_data in plots_data]  # type: ignore[arg-type]
        return PlotConfig(layout=layout, plots=plots)

    @classmethod
    def load_from_file(cls, file_path: str, time_unit: str = "s") -> PlotConfig:
        """Load configuration from JSON file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Plot config file not found: {file_path}")

        try:
            with open(path, "r") as f:
                config_data = json.load(f)
            converted_data = cls.apply_time_unit_conversion(config_data, time_unit)
            return PlotConfig(**converted_data)
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
