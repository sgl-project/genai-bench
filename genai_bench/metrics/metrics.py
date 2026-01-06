"""File for all the metrics definitions."""

from typing import Any, ClassVar, Dict, Optional

from pydantic import BaseModel, Field, model_validator


class RequestLevelMetrics(BaseModel):
    """
    A class to encapsulate metrics related to a single request.
    """

    ttft: Optional[float] = Field(None, description="Time to first token (TTFT)")
    tpot: Optional[float] = Field(None, description="Time per output token (TPOT)")
    e2e_latency: Optional[float] = Field(None, description="End-to-end latency")
    output_latency: Optional[float] = Field(None, description="Output latency")
    output_inference_speed: Optional[float] = Field(
        None, description="Output inference speed in tokens/s"
    )
    num_input_tokens: Optional[int] = Field(None, description="Number of input tokens")
    num_output_tokens: Optional[int] = Field(
        None, description="Number of output tokens"
    )
    total_tokens: Optional[int] = Field(None, description="Total tokens processed")
    input_throughput: Optional[float] = Field(
        None, description="Input throughput in tokens/s"
    )
    output_throughput: Optional[float] = Field(
        None, description="Output throughput in tokens/s"
    )
    error_code: Optional[int] = Field(None, description="Error code")
    error_message: Optional[str] = Field(None, description="Error message")

    # Class-level dictionaries to map output metrics to output fields
    OUTPUT_METRICS_FIELDS: ClassVar[set[str]] = {
        "tpot",
        "output_latency",
        "output_inference_speed",
        "num_output_tokens",
        "output_throughput",
    }

    @model_validator(mode="before")
    def validate_metrics(cls, values):
        """
        Ensure that metrics are validated only if error_code is None.
        """
        if not isinstance(values, dict):
            return values

        error_code = values.get("error_code")
        if error_code is None:
            # Validate all metric fields
            for field_name, field_value in values.items():
                if (
                    field_name not in {"error_code", "error_message"}
                    and field_value is None
                ):
                    raise ValueError(
                        f"{field_name} must not be None if error_code is None."
                    )
        return values


class StatField(BaseModel):
    """Statistics for a single metric field."""

    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    stddev: Optional[float] = None
    sum: Optional[float] = None
    p25: Optional[float] = None
    p50: Optional[float] = None
    p75: Optional[float] = None
    p90: Optional[float] = None
    p95: Optional[float] = None
    p99: Optional[float] = None

    def __getitem__(self, key: str) -> Optional[float]:
        """Allow dict-style access for percentiles and stats."""
        return getattr(self, key)

    def __setitem__(self, key: str, value: Optional[float]):
        """Allow dict-style setting for percentiles and stats."""
        setattr(self, key, value)


class MetricStats(BaseModel):
    """Statistics for RequestLevelMetrics fields."""

    ttft: StatField = Field(default_factory=StatField)
    tpot: StatField = Field(default_factory=StatField)
    e2e_latency: StatField = Field(default_factory=StatField)
    output_latency: StatField = Field(default_factory=StatField)
    output_inference_speed: StatField = Field(default_factory=StatField)
    num_input_tokens: StatField = Field(default_factory=StatField)
    num_output_tokens: StatField = Field(default_factory=StatField)
    total_tokens: StatField = Field(default_factory=StatField)
    input_throughput: StatField = Field(default_factory=StatField)
    output_throughput: StatField = Field(default_factory=StatField)

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        """Convert to dictionary format for serialization."""
        return {
            name: getattr(self, name).model_dump()
            for name in RequestLevelMetrics.model_fields
            if name not in {"error_code", "error_message"}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Dict[str, float]]) -> "MetricStats":
        """Create from dictionary format."""
        return cls(
            **{
                field_name: StatField(**field_stats)
                for field_name, field_stats in data.items()
            }
        )


class AggregatedMetrics(BaseModel):
    """
    A class to encapsulate metrics related to aggregated metrics for an entire
    run. A run is constrained by [scenario, num_concurrency].
    """

    # Run Metadata
    scenario: Optional[str] = Field(None, description="The sample scenario")
    num_concurrency: int = Field(1, description="Number of concurrency")
    batch_size: int = Field(1, description="Batch size for embedding tasks")
    iteration_type: str = Field(
        "num_concurrency",
        description="Type of iteration used (num_concurrency or batch_size)",
    )

    run_duration: float = Field(0.0, description="Run duration in seconds.")
    mean_output_throughput_tokens_per_s: float = Field(
        0.0,
        description="The mean of output tokens throughput across all requests "
        "in tokens/s",
    )
    mean_input_throughput_tokens_per_s: float = Field(
        0.0,
        description="The mean of output tokens throughput across all requests "
        "in tokens/s",
    )
    mean_total_tokens_throughput_tokens_per_s: float = Field(
        0.0,
        description="The mean of total tokens throughput across all requests "
        "in tokens/s",
    )
    requests_per_second: float = Field(
        0.0, description="The average number of completed requests per second"
    )
    error_codes_frequency: Dict[int, int] = Field(
        default_factory=dict, description="Frequency of error codes"
    )
    error_rate: float = Field(
        0.0, description="The rate of error requests across all requests"
    )
    num_error_requests: int = Field(0, description="Number of error requests")
    num_completed_requests: int = Field(0, description="Number of completed requests")
    num_requests: int = Field(0, description="Number of total requests")

    stats: MetricStats = Field(
        default_factory=MetricStats,
        description="Statistics for each field in request level metrics",
    )

    def model_dump(self, *args, **kwargs) -> Dict:
        data = super().model_dump()
        data["stats"] = self.stats.to_dict()
        return data

    @classmethod
    def model_validate(cls, obj: Any, *args, **kwargs) -> "AggregatedMetrics":
        if isinstance(obj.get("stats"), dict):
            obj["stats"] = MetricStats.from_dict(obj["stats"])
        return super().model_validate(obj)
