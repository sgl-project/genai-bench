import json
from typing import List, Optional

import numpy as np

from genai_bench.logging import init_logger
from genai_bench.metrics.metrics import AggregatedMetrics, RequestLevelMetrics
from genai_bench.protocol import LiveMetricsData
from genai_bench.time_units import TimeUnitConverter

logger = init_logger(__name__)


class AggregatedMetricsCollector:
    """
    A class to collect and aggregate metrics from individual requests.

    Attributes:
        aggregated_metrics (AggregatedMetrics): An instance to store aggregated
            metrics data.
        all_request_metrics (List[RequestLevelMetrics]): A list that contains
            all the single request level metrics.
        _live_metrics_data (Dict): A dictionary that stores the metrics data
            to update lively on the UI.
    """

    def __init__(self):
        self.aggregated_metrics: AggregatedMetrics = AggregatedMetrics()
        self.all_request_metrics: List[RequestLevelMetrics] = []
        self._live_metrics_data: LiveMetricsData = {
            "ttft": [],
            "input_throughput": [],
            "output_throughput": [],
            "output_latency": [],
            "stats": {},
        }

    def add_single_request_metrics(self, metrics: RequestLevelMetrics):
        """Adds metrics from a single request to the aggregated metrics."""
        self._null_unreliable_output_metrics(metrics)

        # Store individual request metrics
        self.all_request_metrics.append(metrics)

        # Track error codes frequency directly
        if metrics.error_code:
            if metrics.error_code not in self.aggregated_metrics.error_codes_frequency:
                self.aggregated_metrics.error_codes_frequency[metrics.error_code] = 0
            self.aggregated_metrics.error_codes_frequency[metrics.error_code] += 1
        else:
            self.aggregated_metrics.num_completed_requests += 1

            # Collect live metrics for updating in real-time
            if metrics.ttft is not None:
                self._live_metrics_data["ttft"].append(metrics.ttft)  # type: ignore[union-attr]
            if metrics.input_throughput is not None:
                self._live_metrics_data["input_throughput"].append(  # type: ignore[union-attr]
                    metrics.input_throughput
                )
            if metrics.output_throughput is not None:
                self._live_metrics_data["output_throughput"].append(  # type: ignore[union-attr]
                    metrics.output_throughput
                )
            if metrics.output_latency is not None:
                self._live_metrics_data["output_latency"].append(metrics.output_latency)  # type: ignore[union-attr]

            # Update live metrics aggregation
            self._update_live_metrics()

    # Above this, the per-token output metrics are treated as unmeasurable
    # rather than real. A short response that streams back in one (or very few)
    # chunks collapses output_latency toward zero, so the derived per-token
    # rates explode; genuine decode speeds for current models stay well under
    # this bound.
    MAX_PLAUSIBLE_OUTPUT_INFERENCE_SPEED = 1000  # tokens/s

    @classmethod
    def _null_unreliable_output_metrics(cls, metrics: RequestLevelMetrics) -> None:
        """Null per-token output metrics that could not be measured reliably.

        When a response is too short to observe inter-token timing (e.g. it
        arrives in a single streaming chunk), ``output_latency`` collapses
        toward zero and the derived per-token metrics — ``tpot``,
        ``output_inference_speed`` and ``output_throughput`` — blow up to
        implausible values. Such a request used to be dropped wholesale, which
        (a) discarded its still-valid ttft / e2e / token-count data and (b)
        could leave the run with zero usable ``tpot`` samples, crashing
        aggregation. Instead we null only the unreliable derived fields and keep
        the request; aggregation then simply has fewer (or zero) samples for
        those metrics, which it now tolerates.
        """
        speed = metrics.output_inference_speed
        if speed is not None and speed > cls.MAX_PLAUSIBLE_OUTPUT_INFERENCE_SPEED:
            logger.warning(
                f"Output inference speed {speed:.1f} tokens/s exceeds the "
                f"plausible maximum ({cls.MAX_PLAUSIBLE_OUTPUT_INFERENCE_SPEED} "
                f"tokens/s); the response was likely too short to measure "
                f"per-token timing. Nulling tpot, output_inference_speed and "
                f"output_throughput for this request (ttft, e2e latency and "
                f"token counts are kept)."
            )
            metrics.tpot = None
            metrics.output_inference_speed = None
            metrics.output_throughput = None

    def _update_live_metrics(self):
        """Calculates live metrics like avg, max, min, and percentiles."""
        for key in self._live_metrics_data:
            if key == "stats":
                continue
            values: List[float] = self._live_metrics_data[key]  # type: ignore
            if values:
                self._live_metrics_data["stats"][key] = {
                    "min": np.min(values).item(),
                    "max": np.max(values).item(),
                    "mean": np.mean(values).item(),
                }

                if key in ["ttft", "output_latency"]:  # Latency fields
                    percentiles = np.percentile(values, [50, 90, 99])
                    self._live_metrics_data["stats"][key].update(
                        {
                            "p50": percentiles[0].item(),
                            "p90": percentiles[1].item(),
                            "p99": percentiles[2].item(),
                        }
                    )

    def aggregate_metrics_data(
        self,
        start_time: float,
        end_time: float,
        dataset_character_to_token_ratio: float,
        warmup_ratio: Optional[float],
        cooldown_ratio: Optional[float],
    ):
        """
        Aggregates collected metrics data over all requests.

        Args:
            start_time (float): The start time of the aggregated metrics data.
            end_time (float): The end time of the aggregated metrics data.
            dataset_character_to_token_ratio (float): The ratio of characters
                to tokens. It is required to calculate character-level metric:
                mean_total_chars_per_hour.
            warmup_ratio (Optional[float]): The portion of initial requests
                to exclude from the aggregation as warmup.
            cooldown_ratio (Optional[float]): The portion of final requests
                to exclude from the aggregation as cooldown.
        """
        if not self.all_request_metrics:
            logger.warning(
                "‼️ No request metrics collected, one possible cause "
                "is that your run time is too short for the server "
                "to finish any requests. Skipping collecting "
                "aggregated metrics for this run."
            )
            return

        # Calculate statistical aggregates for each metric
        filtered_keys: List[str] = [
            key
            for key in RequestLevelMetrics.model_fields
            if key not in {"error_code", "error_message"}
        ]

        warmup_number = 0
        if warmup_ratio:
            warmup_number = int(len(self.all_request_metrics) * warmup_ratio)
            logger.info(
                f"Filtering out first {warmup_number}/{len(self.all_request_metrics)} "
                f"warmup requests."
            )

        cooldown_number = 0
        if cooldown_ratio:
            cooldown_number = int(len(self.all_request_metrics) * cooldown_ratio)
            logger.info(
                f"Filtering out last {cooldown_number}/{len(self.all_request_metrics)} "
                f"cooldown requests."
            )

        for key in filtered_keys:
            # Extract the list of values for this metric from all requests
            values: List[float] = []
            for i, metrics in enumerate(self.all_request_metrics):
                # Skip adding the value when the request metric has error
                if metrics.error_code:
                    continue
                value = getattr(metrics, key)
                if value is None:
                    logger.info(f"{i}th request has NoneType value in metric {key}.")
                    continue
                if warmup_number <= i < len(self.all_request_metrics) - cooldown_number:
                    values.append(value)

            # A metric can legitimately have zero samples — e.g. tpot,
            # output_inference_speed or output_throughput on a short-output run
            # where every response was too brief to measure per-token timing
            # (see _null_unreliable_output_metrics). Leave its aggregate stats
            # unset (None) and carry on rather than aborting the whole run and
            # discarding every other metric.
            if not values:
                logger.warning(
                    f"No values found for metric '{key}' across "
                    f"{len(self.all_request_metrics)} request(s); leaving its "
                    f"aggregate stats unset."
                )
                continue

            percentiles = np.percentile(values, [25, 50, 75, 90, 95, 99])
            stat_field = getattr(self.aggregated_metrics.stats, key)
            stat_field.min = np.min(values).item()
            stat_field.max = np.max(values).item()
            stat_field.mean = np.mean(values).item()
            stat_field.stddev = np.std(values).item()
            stat_field.sum = np.sum(values).item()
            stat_field.p25 = percentiles[0].item()
            stat_field.p50 = percentiles[1].item()
            stat_field.p75 = percentiles[2].item()
            stat_field.p90 = percentiles[3].item()
            stat_field.p95 = percentiles[4].item()
            stat_field.p99 = percentiles[5].item()

        # Calculate additional metadata based on aggregated data
        self.aggregated_metrics.run_duration = end_time - start_time

        # Handle None values safely
        num_output_tokens_sum = self.aggregated_metrics.stats.num_output_tokens.sum or 0
        num_input_tokens_sum = self.aggregated_metrics.stats.num_input_tokens.sum or 0
        total_tokens_sum = self.aggregated_metrics.stats.total_tokens.sum or 0

        self.aggregated_metrics.mean_output_throughput_tokens_per_s = (
            num_output_tokens_sum / self.aggregated_metrics.run_duration
        )

        self.aggregated_metrics.mean_input_throughput_tokens_per_s = (
            num_input_tokens_sum / self.aggregated_metrics.run_duration
        )

        self.aggregated_metrics.mean_total_tokens_throughput_tokens_per_s = (
            total_tokens_sum / self.aggregated_metrics.run_duration
        )

        self.aggregated_metrics.mean_total_chars_per_hour = (
            self.aggregated_metrics.mean_total_tokens_throughput_tokens_per_s
            * dataset_character_to_token_ratio
            * 3600
        )

        # Calculate error rate
        self.aggregated_metrics.num_error_requests = sum(
            self.aggregated_metrics.error_codes_frequency.values()
        )
        self.aggregated_metrics.error_rate = (
            self.aggregated_metrics.num_error_requests
            / self.aggregated_metrics.num_requests
            if self.aggregated_metrics.num_requests > 0
            else 0
        )

        if self.aggregated_metrics.error_rate >= 0.5:
            logger.warning(
                f"‼️ Error rate {self.aggregated_metrics.error_rate}"
                f" for current run is greater than 0.5! Please "
                f"check logs from genai-bench and server!"
            )

        # Calculate requests per minute
        self.aggregated_metrics.requests_per_second = (
            (
                self.aggregated_metrics.num_completed_requests
                / self.aggregated_metrics.run_duration
            )
            if self.aggregated_metrics.run_duration > 0
            else 0
        )
        self.aggregated_metrics.num_requests = (
            self.aggregated_metrics.num_completed_requests
            + self.aggregated_metrics.num_error_requests
        )

    def set_run_metadata(
        self, iteration: int, scenario_str: str, iteration_type: str = "num_concurrency"
    ):
        """Set metadata for the current run"""
        setattr(self.aggregated_metrics, iteration_type, iteration)
        self.aggregated_metrics.scenario = scenario_str
        self.aggregated_metrics.iteration_type = iteration_type

    def clear(self):
        """Clear the metrics to prepare for the next experiment run."""
        self.aggregated_metrics = AggregatedMetrics()
        self.all_request_metrics = []
        self._live_metrics_data = {
            "ttft": [],
            "input_throughput": [],
            "output_throughput": [],
            "output_latency": [],
            "stats": {},
        }

    def save(self, file_path: str, metrics_time_unit: str = "s"):
        if not self.all_request_metrics:
            return

        # Convert aggregated metrics to the specified time unit
        aggregated_dict = TimeUnitConverter.convert_metrics_dict(
            self.aggregated_metrics.model_dump(), metrics_time_unit
        )

        # Convert individual request metrics to the specified time unit
        individual_dicts = TimeUnitConverter.convert_metrics_list(
            [metrics.model_dump() for metrics in self.all_request_metrics],
            metrics_time_unit,
        )

        data_to_save = {
            "aggregated_metrics": aggregated_dict,
            "individual_request_metrics": individual_dicts,
            "_time_unit": metrics_time_unit,  # Store metadata for reference
        }
        with open(file_path, "w") as metrics_file:
            json.dump(data_to_save, metrics_file, indent=4)

    def get_live_metrics(self) -> LiveMetricsData:
        """Returns the latest live metrics for use in the UI."""
        return self._live_metrics_data

    def get_ui_scatter_plot_metrics(
        self, metrics_time_unit: str = "s"
    ) -> List[float] | None:
        """Returns the plot metrics for use in the UI."""
        mean_ttft = self.aggregated_metrics.stats.ttft.mean
        mean_output_latency = self.aggregated_metrics.stats.output_latency.mean
        if mean_ttft is None or mean_output_latency is None:
            return None

        # Convert time-based metrics to the specified time unit for UI display
        converted_ttft = TimeUnitConverter.convert_value(
            mean_ttft, "s", metrics_time_unit
        )
        converted_output_latency = TimeUnitConverter.convert_value(
            mean_output_latency, "s", metrics_time_unit
        )

        return [
            converted_ttft,
            converted_output_latency,
            self.aggregated_metrics.mean_input_throughput_tokens_per_s,
            self.aggregated_metrics.mean_output_throughput_tokens_per_s,
        ]
