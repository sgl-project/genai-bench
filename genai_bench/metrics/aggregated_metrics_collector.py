import json
from typing import List

import numpy as np

from genai_bench.logging import init_logger
from genai_bench.metrics.metrics import AggregatedMetrics, RequestLevelMetrics
from genai_bench.protocol import LiveMetricsData

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
        if self._should_filter_metrics(metrics):
            return

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

    @staticmethod
    def _should_filter_metrics(metrics: RequestLevelMetrics) -> bool:
        """Filters metrics based on out of range TPOT/inference speed."""
        inference_speed = metrics.output_inference_speed
        if inference_speed is not None and inference_speed > 1000:
            logger.warning(
                f"Metric has abnormal inference speed: {inference_speed} tokens/s."
                " Filtering it out."
            )
            return True
        return False

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
    ):
        """
        Aggregates collected metrics data over all requests.

        Args:
            start_time (float): The start time of the aggregated metrics data.
            end_time (float): The end time of the aggregated metrics data.
            dataset_character_to_token_ratio (float): The ratio of characters
                to tokens. It is required to calculate character-level metric:
                mean_total_chars_per_hour.
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
                values.append(value)

            # Validate that all values are valid for processing
            if not values:
                raise ValueError(
                    f"No values found for metric '{key}'. This should never happen!"
                )

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

    def save(self, file_path: str):
        if not self.all_request_metrics:
            return

        data_to_save = {
            "aggregated_metrics": self.aggregated_metrics.model_dump(),
            "individual_request_metrics": [
                metrics.model_dump() for metrics in self.all_request_metrics
            ],
        }
        with open(file_path, "w") as metrics_file:
            json.dump(data_to_save, metrics_file, indent=4)

    def get_live_metrics(self) -> LiveMetricsData:
        """Returns the latest live metrics for use in the UI."""
        return self._live_metrics_data

    def get_ui_scatter_plot_metrics(self) -> List[float] | None:
        """Returns the plot metrics for use in the UI."""
        mean_ttft = self.aggregated_metrics.stats.ttft.mean
        mean_output_latency = self.aggregated_metrics.stats.output_latency.mean
        if mean_ttft is None or mean_output_latency is None:
            return None

        return [
            mean_ttft,
            mean_output_latency,
            self.aggregated_metrics.mean_input_throughput_tokens_per_s,
            self.aggregated_metrics.mean_output_throughput_tokens_per_s,
        ]
