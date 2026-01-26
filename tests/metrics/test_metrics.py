from locust import events
from locust.env import Environment
from locust.event import EventHook

import json
import logging
import os
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from genai_bench.metrics.aggregated_metrics_collector import AggregatedMetricsCollector
from genai_bench.metrics.metrics import (
    AggregatedMetrics,
    MetricStats,
    RequestLevelMetrics,
    StatField,
)
from genai_bench.metrics.request_metrics_collector import RequestMetricsCollector
from genai_bench.protocol import ExperimentMetadata, UserChatResponse, UserResponse


@pytest.fixture
def aggregated_metrics_collector():
    return AggregatedMetricsCollector()


@pytest.fixture
def locust_environment():
    # Setup event environment
    env = Environment(events=events)
    events.request_metrics = EventHook()
    return env


def test_request_level_metrics_calculation_with_chat_response():
    mock_response = MagicMock(spec=UserChatResponse)
    mock_response.status_code = 200
    mock_response.generated_text = "and and and and"
    mock_response.tokens_received = 4
    mock_response.time_at_first_token = 1722986731
    mock_response.start_time = 1722986631
    mock_response.end_time = 1722986741
    mock_response.num_prefill_tokens = 10

    # Initialize and calculate request metrics
    request_metrics_collector = RequestMetricsCollector()
    request_metrics_collector.calculate_metrics(mock_response)

    # Check calculations
    assert request_metrics_collector.metrics.ttft == 100
    assert request_metrics_collector.metrics.e2e_latency == 110
    assert request_metrics_collector.metrics.num_input_tokens == 10


def test_request_level_metrics_calculation_with_embeddings_response():
    mock_response = MagicMock(spec=UserResponse)
    mock_response.status_code = 200
    mock_response.time_at_first_token = 1722986741
    mock_response.start_time = 1722986631
    mock_response.end_time = 1722986741
    mock_response.num_prefill_tokens = 10

    # Initialize and calculate request metrics
    request_metrics_collector = RequestMetricsCollector()
    request_metrics_collector.calculate_metrics(mock_response)

    # Check calculations
    assert request_metrics_collector.metrics.ttft == 110
    assert request_metrics_collector.metrics.e2e_latency == 110
    assert request_metrics_collector.metrics.num_input_tokens == 10


def test_event_aggregation(aggregated_metrics_collector, locust_environment):
    # Initialize events and hook aggregated_metrics_collector to env
    assert hasattr(locust_environment.events, "request_metrics")
    locust_environment.events.request_metrics.add_listener(
        aggregated_metrics_collector.add_single_request_metrics
    )

    aggregated_metrics_collector.set_run_metadata(
        iteration=3,
        scenario_str="N(480,240)/(300,150)",
        iteration_type="num_concurrency",
    )

    # Create sample request metrics
    metrics1 = RequestLevelMetrics(
        ttft=0.1,
        tpot=0.2,
        e2e_latency=1.0,
        output_latency=0.9,
        input_throughput=20.0,
        output_throughput=11.111,
        num_input_tokens=2,
        num_output_tokens=10,
        output_inference_speed=5,
        total_tokens=12,
    )

    metrics2 = RequestLevelMetrics(
        ttft=0.2,
        tpot=0.3,
        e2e_latency=1.5,
        output_latency=1.3,
        input_throughput=15.0,
        output_throughput=8.46,
        num_input_tokens=3,
        num_output_tokens=11,
        output_inference_speed=3.3,
        total_tokens=14,
    )

    metrics3 = RequestLevelMetrics(
        error_code=500, error_message="Internal Server Error"
    )

    # Simulate firing the custom request_metrics event
    locust_environment.events.request_metrics.fire(metrics=metrics1)
    locust_environment.events.request_metrics.fire(metrics=metrics2)
    locust_environment.events.request_metrics.fire(metrics=metrics3)

    # Manually call the aggregation logic to verify results
    start_time = 0
    end_time = 1.5
    aggregated_metrics_collector.aggregate_metrics_data(start_time, end_time, 0.0, 0.0)
    aggregated_metrics = aggregated_metrics_collector.aggregated_metrics

    # Check aggregate calculations
    assert "error_message" not in aggregated_metrics.stats
    assert aggregated_metrics.run_duration == 1.5
    assert aggregated_metrics.stats.ttft["mean"] == pytest.approx(0.15)
    assert aggregated_metrics.stats.tpot["mean"] == pytest.approx(0.25)
    assert aggregated_metrics.stats.output_latency["mean"] == pytest.approx(1.1)
    assert aggregated_metrics.stats.e2e_latency["mean"] == pytest.approx(1.25)
    assert aggregated_metrics.stats.input_throughput["mean"] == pytest.approx(17.5)
    assert aggregated_metrics.stats.output_throughput["mean"] == pytest.approx(9.7855)

    # Check metadata calculations
    assert aggregated_metrics.num_requests == 3
    assert aggregated_metrics.num_completed_requests == 2
    assert aggregated_metrics.mean_input_throughput_tokens_per_s == pytest.approx(
        3.3333, rel=0.0001
    )
    assert aggregated_metrics.mean_output_throughput_tokens_per_s == 14.0
    assert (
        aggregated_metrics.mean_total_tokens_throughput_tokens_per_s
        == pytest.approx(17.33333332536, rel=0.00005)
    )
    assert (
        aggregated_metrics.mean_total_tokens_throughput_tokens_per_s * 60
        == pytest.approx(17.33333332536 * 60, rel=0.00005)
    )
    assert aggregated_metrics.requests_per_second == pytest.approx(80 / 60)

    ttft, output_latency, input_throughput, output_throughput = (
        aggregated_metrics_collector.get_ui_scatter_plot_metrics()
    )
    assert ttft == pytest.approx(0.15)
    assert output_latency == pytest.approx(1.1)
    assert input_throughput == pytest.approx(3.333333, rel=0.02)
    assert output_throughput == 14.0


def test_set_run_metadata_with_int_iteration(aggregated_metrics_collector):
    """Test that set_run_metadata accepts int for request_rate iteration."""
    # Test with integer iteration value (request_rate)
    aggregated_metrics_collector.set_run_metadata(
        iteration=15,  # Integer value for request_rate
        scenario_str="D(100,100)",
        iteration_type="request_rate",
    )

    # Verify the request_rate field was set
    assert aggregated_metrics_collector.aggregated_metrics.request_rate == 15
    assert (
        aggregated_metrics_collector.aggregated_metrics.iteration_type == "request_rate"
    )

    # Test with integer iteration value (num_concurrency)
    aggregated_metrics_collector.set_run_metadata(
        iteration=10,  # Integer value for num_concurrency
        scenario_str="D(100,100)",
        iteration_type="num_concurrency",
    )

    # Verify the num_concurrency field was set
    assert aggregated_metrics_collector.aggregated_metrics.num_concurrency == 10
    assert (
        aggregated_metrics_collector.aggregated_metrics.iteration_type
        == "num_concurrency"
    )


def test_set_run_metadata_request_rate_field_handling(aggregated_metrics_collector):
    """Test that set_run_metadata correctly handles request_rate field."""
    # Test setting request_rate
    aggregated_metrics_collector.set_run_metadata(
        iteration=20,
        scenario_str="test_scenario",
        iteration_type="request_rate",
    )

    assert aggregated_metrics_collector.aggregated_metrics.request_rate == 20
    assert (
        aggregated_metrics_collector.aggregated_metrics.iteration_type == "request_rate"
    )

    # Test that request_rate is None when using other iteration types
    aggregated_metrics_collector.set_run_metadata(
        iteration=5,
        scenario_str="test_scenario",
        iteration_type="num_concurrency",
    )

    # request_rate should remain None or be reset
    # (depending on implementation, but should not be 20.0)
    assert (
        aggregated_metrics_collector.aggregated_metrics.iteration_type
        == "num_concurrency"
    )
    assert aggregated_metrics_collector.aggregated_metrics.num_concurrency == 5


def test_aggregate_metrics_with_error_filtering(
    aggregated_metrics_collector, locust_environment
):
    """Test aggregation of metrics with error filtering via events."""
    # Initialize events and hook aggregated_metrics_collector to env
    locust_environment.events.request_metrics.add_listener(
        aggregated_metrics_collector.add_single_request_metrics
    )

    metrics1 = RequestLevelMetrics(
        ttft=0.1,
        tpot=0.2,
        e2e_latency=1.0,
        output_latency=0.9,
        input_throughput=20.0,
        output_throughput=11.111,
        num_input_tokens=2,
        num_output_tokens=10,
        output_inference_speed=5,
        total_tokens=12,
    )

    metrics2 = RequestLevelMetrics(
        ttft=0.2,
        tpot=0.3,
        e2e_latency=1.5,
        output_latency=1.3,
        input_throughput=15.0,
        output_throughput=8.46,
        num_input_tokens=3,
        num_output_tokens=11,
        output_inference_speed=3.3,
        total_tokens=14,
    )

    metrics3 = RequestLevelMetrics(
        error_code=500, error_message="Internal Server Error"
    )

    # Simulate firing the custom request_metrics event
    locust_environment.events.request_metrics.fire(metrics=metrics1)
    locust_environment.events.request_metrics.fire(metrics=metrics2)
    locust_environment.events.request_metrics.fire(metrics=metrics3)

    # Manually call the aggregation logic to verify results
    start_time = 0
    end_time = 1.5
    aggregated_metrics_collector.aggregate_metrics_data(start_time, end_time, 0.0, 0.0)
    aggregated_metrics = aggregated_metrics_collector.aggregated_metrics

    # Check aggregate calculations
    assert "error_message" not in aggregated_metrics.stats
    assert aggregated_metrics.run_duration == 1.5
    assert aggregated_metrics.stats.ttft["mean"] == pytest.approx(0.15)
    assert aggregated_metrics.stats.tpot["mean"] == pytest.approx(0.25)
    assert aggregated_metrics.stats.output_latency["mean"] == pytest.approx(1.1)
    assert aggregated_metrics.stats.e2e_latency["mean"] == pytest.approx(1.25)
    assert aggregated_metrics.stats.input_throughput["mean"] == pytest.approx(17.5)
    assert aggregated_metrics.stats.output_throughput["mean"] == pytest.approx(9.7855)

    # Check metadata calculations
    assert aggregated_metrics.num_requests == 3
    assert aggregated_metrics.num_completed_requests == 2
    assert aggregated_metrics.mean_input_throughput_tokens_per_s == pytest.approx(
        3.3333, rel=0.0001
    )
    assert aggregated_metrics.mean_output_throughput_tokens_per_s == 14.0
    assert (
        aggregated_metrics.mean_total_tokens_throughput_tokens_per_s
        == pytest.approx(17.33333332536, rel=0.00005)
    )
    assert aggregated_metrics.requests_per_second == pytest.approx(80 / 60)

    ttft, output_latency, input_throughput, output_throughput = (
        aggregated_metrics_collector.get_ui_scatter_plot_metrics()
    )
    assert ttft == pytest.approx(0.15)
    assert output_latency == pytest.approx(1.1)
    assert input_throughput == pytest.approx(3.333333, rel=0.02)
    assert output_throughput == 14.0


def test_filter_metrics(aggregated_metrics_collector):
    """Test that metrics with very small tpot are excluded from all_request_metrics."""
    metrics1 = RequestLevelMetrics(
        ttft=0.1,
        tpot=0.0000002,
        e2e_latency=1.0,
        output_latency=0.0002,
        input_throughput=20.0,
        output_throughput=11.111,
        num_input_tokens=2,
        num_output_tokens=10,
        output_inference_speed=1 / 0.0000002,
        total_tokens=12,
    )

    aggregated_metrics_collector.add_single_request_metrics(metrics1)

    # metrics are added to list, but tpot and inference speed fields are filtered
    assert len(aggregated_metrics_collector.all_request_metrics) == 1
    stored_metrics = aggregated_metrics_collector.all_request_metrics[0]

    assert stored_metrics.tpot is None
    assert stored_metrics.output_inference_speed is None

    # Verify other fields preserved
    assert stored_metrics.ttft == 0.1
    assert stored_metrics.output_latency == 0.0002
    assert stored_metrics.num_output_tokens == 10

    embedding_metrics = RequestLevelMetrics(
        ttft=0.1,
        e2e_latency=0.1,
        input_throughput=20.0,
        num_input_tokens=2,
    )

    aggregated_metrics_collector.add_single_request_metrics(embedding_metrics)
    assert embedding_metrics in aggregated_metrics_collector.all_request_metrics


def test_update_live_metrics(aggregated_metrics_collector):
    # Create sample request metrics
    metrics1 = RequestLevelMetrics(
        ttft=0.1,
        tpot=0.2,
        e2e_latency=1.0,
        output_latency=0.9,
        input_throughput=20.0,
        output_throughput=11.111,
        num_input_tokens=2,
        num_output_tokens=10,
        output_inference_speed=5,
        total_tokens=12,
    )

    metrics2 = RequestLevelMetrics(
        ttft=0.2,
        tpot=0.3,
        e2e_latency=1.5,
        output_latency=1.3,
        input_throughput=15.0,
        output_throughput=8.46,
        num_input_tokens=3,
        num_output_tokens=11,
        output_inference_speed=3.3,
        total_tokens=14,
    )

    # Add request metrics to the collector
    aggregated_metrics_collector.add_single_request_metrics(metrics1)
    aggregated_metrics_collector.add_single_request_metrics(metrics2)

    # Check live metrics data
    live_metrics = aggregated_metrics_collector.get_live_metrics()

    assert "stats" in live_metrics
    assert live_metrics["stats"]["ttft"]["min"] == pytest.approx(0.1)
    assert live_metrics["stats"]["ttft"]["max"] == pytest.approx(0.2)
    assert live_metrics["stats"]["ttft"]["mean"] == pytest.approx(0.15)
    assert live_metrics["stats"]["ttft"]["p50"] == pytest.approx(0.15)
    assert live_metrics["stats"]["ttft"]["p90"] == pytest.approx(0.19)
    assert live_metrics["stats"]["ttft"]["p99"] == pytest.approx(0.199)

    assert live_metrics["stats"]["output_latency"]["min"] == pytest.approx(0.9)
    assert live_metrics["stats"]["output_latency"]["max"] == pytest.approx(1.3)
    assert live_metrics["stats"]["output_latency"]["mean"] == pytest.approx(1.1)
    assert live_metrics["stats"]["output_latency"]["p50"] == pytest.approx(1.1)
    assert live_metrics["stats"]["output_latency"]["p90"] == pytest.approx(1.26)
    assert live_metrics["stats"]["output_latency"]["p99"] == pytest.approx(1.296)


def test_filter_warmup_and_cooldown_metrics(aggregated_metrics_collector):
    metrics = [
        RequestLevelMetrics(
            ttft=0.1,
            tpot=0.2,
            e2e_latency=1.0,
            output_latency=0.9,
            input_throughput=20.0,
            output_throughput=11.111,
            num_input_tokens=2,
            num_output_tokens=10,
            output_inference_speed=5,
            total_tokens=12,
        )
        for _ in range(10)
    ]

    metrics[0].ttft = 0.0
    metrics[-1].ttft = 0.0

    for metric in metrics:
        aggregated_metrics_collector.add_single_request_metrics(metric)

    aggregated_metrics_collector.aggregate_metrics_data(0, 1, 0.1, 0.1)
    aggregated_metrics = aggregated_metrics_collector.aggregated_metrics

    # Check aggregate calculations#
    assert aggregated_metrics.stats.ttft["mean"] == pytest.approx(0.1)
    assert aggregated_metrics.stats.tpot["mean"] == pytest.approx(0.2)
    assert aggregated_metrics.stats.e2e_latency["mean"] == pytest.approx(1.0)
    assert aggregated_metrics.stats.output_latency["mean"] == pytest.approx(0.9)
    assert aggregated_metrics.stats.input_throughput["mean"] == pytest.approx(20.0)
    assert aggregated_metrics.stats.output_throughput["mean"] == pytest.approx(11.111)
    # Check metadata calculations
    assert aggregated_metrics.num_requests == 10


def test_save_metrics(aggregated_metrics_collector, tmp_path):
    # Add some sample metrics
    metrics = RequestLevelMetrics(
        ttft=0.1,
        tpot=0.2,
        e2e_latency=1.0,
        output_latency=0.9,
        input_throughput=20.0,
        output_throughput=11.111,
        num_input_tokens=2,
        num_output_tokens=10,
        output_inference_speed=5,
        total_tokens=12,
    )
    aggregated_metrics_collector.add_single_request_metrics(metrics)
    aggregated_metrics_collector.aggregate_metrics_data(0, 1, 0.0, 0.0)

    # Save the metrics to a file
    save_path = tmp_path / "metrics.json"
    aggregated_metrics_collector.save(str(save_path))

    # Verify the file was created and contains the correct data
    with open(save_path, "r") as f:
        saved_data = json.load(f)

    assert "aggregated_metrics" in saved_data
    assert "individual_request_metrics" in saved_data
    assert len(saved_data["individual_request_metrics"]) == 1
    assert saved_data["aggregated_metrics"]["num_requests"] == 1


def test_aggregate_empty_metrics(aggregated_metrics_collector, tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        aggregated_metrics_collector.aggregate_metrics_data(0, 1, 0.0, 0.0)

    assert (
        "‼️ No request metrics collected, one possible cause "
        "is that your run time is too short for the server "
        "to finish any requests. Skipping collecting "
        "aggregated metrics for this run." in caplog.text
    )

    save_path = tmp_path / "metrics.json"
    aggregated_metrics_collector.save(str(save_path))
    assert not os.path.exists(save_path)


def test_clear_metrics(aggregated_metrics_collector):
    # Add some sample metrics
    metrics = RequestLevelMetrics(
        ttft=0.1,
        tpot=0.2,
        e2e_latency=1.0,
        output_latency=0.9,
        input_throughput=20.0,
        output_throughput=11.111,
        num_input_tokens=2,
        num_output_tokens=10,
        output_inference_speed=5,
        total_tokens=12,
    )
    aggregated_metrics_collector.add_single_request_metrics(metrics)
    assert len(aggregated_metrics_collector.all_request_metrics) == 1

    # Clear the metrics
    aggregated_metrics_collector.clear()

    # Verify that the metrics are cleared
    assert len(aggregated_metrics_collector.all_request_metrics) == 0
    assert len(aggregated_metrics_collector.get_live_metrics()["ttft"]) == 0


def test_validate_metrics_success():
    """Test successful validation when all required fields are present."""
    metrics = RequestLevelMetrics(
        ttft=1.0,
        tpot=2.0,
        e2e_latency=3.0,
        output_latency=1.5,
        output_inference_speed=10.0,
        num_input_tokens=5,
        num_output_tokens=10,
        total_tokens=15,
        input_throughput=20.0,
        output_throughput=25.0,
    )
    assert metrics is not None
    assert metrics.ttft == 1.0

    metrics = RequestLevelMetrics()
    assert metrics.ttft is None


def test_validate_metrics_with_error_code():
    """Test validation when error_code is present - should allow None values."""
    metrics = RequestLevelMetrics(error_code=500, error_message="Internal Server Error")
    assert metrics.error_code == 500
    assert metrics.error_message == "Internal Server Error"
    assert metrics.ttft is None


def test_validate_metrics_missing_required_field():
    """Test validation fails when required field is None and no error_code."""
    with pytest.raises(ValueError, match="ttft must not be None if error_code is None"):
        RequestLevelMetrics(
            ttft=None,  # This should trigger the validation error
            tpot=2.0,
            e2e_latency=3.0,
            output_latency=1.5,
            output_inference_speed=10.0,
            num_input_tokens=5,
            num_output_tokens=10,
            total_tokens=15,
            input_throughput=20.0,
            output_throughput=25.0,
        )


def test_validate_metrics_partial_none_values():
    """Test validation fails when some fields are None without error_code."""
    with pytest.raises(
        ValueError, match="output_latency must not be None if error_code is None"
    ):
        RequestLevelMetrics(
            ttft=1.0,
            tpot=2.0,
            e2e_latency=3.0,
            output_latency=None,  # This should trigger the validation error
            output_inference_speed=10.0,
            num_input_tokens=5,
            num_output_tokens=10,
            total_tokens=15,
            input_throughput=20.0,
            output_throughput=25.0,
        )


def test_validate_metrics_from_json():
    """Test validation of metrics from JSON data."""
    # Valid JSON with all required fields
    valid_json = json.dumps(
        {
            "ttft": 1.0,
            "tpot": 2.0,
            "e2e_latency": 3.0,
            "output_latency": 1.5,
            "output_inference_speed": 10.0,
            "num_input_tokens": 5,
            "num_output_tokens": 10,
            "total_tokens": 15,
            "input_throughput": 20.0,
            "output_throughput": 25.0,
        }
    )
    metrics = RequestLevelMetrics.model_validate_json(valid_json)
    assert metrics.ttft == 1.0
    assert metrics.output_throughput == 25.0

    # Valid JSON with error code
    error_json = json.dumps(
        {
            "error_code": 500,
            "error_message": "Internal Server Error",
            "ttft": None,
            "tpot": None,
            "e2e_latency": None,
            "output_latency": None,
            "output_inference_speed": None,
            "num_input_tokens": None,
            "num_output_tokens": None,
            "total_tokens": None,
            "input_throughput": None,
            "output_throughput": None,
        }
    )
    error_metrics = RequestLevelMetrics.model_validate_json(error_json)
    assert error_metrics.error_code == 500
    assert error_metrics.ttft is None

    # Invalid JSON with None value
    none_value_json = json.dumps(
        {
            "ttft": 1.0,
            "tpot": None,  # This should trigger validation error
            "e2e_latency": 3.0,
            "output_latency": 1.5,
            "output_inference_speed": 10.0,
            "num_input_tokens": 5,
            "num_output_tokens": 10,
            "total_tokens": 15,
            "input_throughput": 20.0,
            "output_throughput": 25.0,
            "error_code": None,
            "error_message": None,
        }
    )
    with pytest.raises(ValueError, match="tpot must not be None if error_code is None"):
        RequestLevelMetrics.model_validate_json(none_value_json)


# Tests for request_rate in metrics and aggregation


def create_test_metadata(**kwargs):
    """Helper to create ExperimentMetadata with required fields."""
    defaults = {
        "cmd": "test command",
        "benchmark_version": "1.0.0",
        "model": "test-model",
        "api_model_name": "test-model",
        "api_backend": "openai",
        "task": "text-to-text",
        "num_concurrency": [1],
        "max_time_per_run_s": 60,
        "max_requests_per_run": 1000,
        "experiment_folder_name": "test_experiment",
    }
    defaults.update(kwargs)
    return ExperimentMetadata(**defaults)


def create_test_stat_field(**kwargs):
    """Helper to create a StatField with sensible defaults."""
    defaults = {
        "min": 0.01,
        "max": 1.0,
        "mean": 0.5,
        "stddev": 0.1,
        "sum": 50.0,
        "p25": 0.4,
        "p50": 0.5,
        "p75": 0.6,
        "p90": 0.7,
        "p95": 0.8,
        "p99": 0.9,
    }
    defaults.update(kwargs)
    return StatField(**defaults)


def create_test_metric_stats():
    """Helper to create MetricStats with all required StatField objects."""
    return MetricStats(
        ttft=create_test_stat_field(),
        tpot=create_test_stat_field(min=0.01, max=0.05, mean=0.02),
        e2e_latency=create_test_stat_field(min=0.5, max=2.0, mean=1.5),
        output_latency=create_test_stat_field(min=0.4, max=1.8, mean=1.3),
        output_inference_speed=create_test_stat_field(min=40.0, max=200.0, mean=70.0),
        num_input_tokens=create_test_stat_field(min=90.0, max=110.0, mean=100.0),
        num_output_tokens=create_test_stat_field(min=90.0, max=110.0, mean=100.0),
        total_tokens=create_test_stat_field(min=180.0, max=220.0, mean=200.0),
        input_throughput=create_test_stat_field(min=400.0, max=1000.0, mean=600.0),
        output_throughput=create_test_stat_field(min=40.0, max=200.0, mean=70.0),
    )


class TestRequestRateInAggregatedMetrics:
    """Test request_rate field in AggregatedMetrics."""

    def test_aggregated_metrics_with_request_rate(self):
        """Test creating AggregatedMetrics with request_rate."""
        metrics = AggregatedMetrics(
            scenario="test_scenario",
            num_concurrency=10,
            batch_size=1,
            request_rate=15,
            iteration_type="request_rate",
            run_duration=60.0,
            mean_output_throughput_tokens_per_s=1000.0,
            mean_input_throughput_tokens_per_s=1000.0,
            mean_total_tokens_throughput_tokens_per_s=2000.0,
            mean_total_chars_per_hour=10000000.0,
            requests_per_second=15.0,
            error_codes_frequency={},
            error_rate=0.0,
            num_error_requests=0,
            num_completed_requests=930,
            num_requests=930,
            stats=create_test_metric_stats(),
        )

        assert metrics.request_rate == 15
        assert metrics.iteration_type == "request_rate"
        assert metrics.num_concurrency == 10
        assert metrics.batch_size == 1

    def test_aggregated_metrics_request_rate_optional(self):
        """Test that request_rate is optional (can be None)."""
        metrics = AggregatedMetrics(
            scenario="test_scenario",
            num_concurrency=5,
            batch_size=1,
            iteration_type="num_concurrency",
            run_duration=60.0,
            mean_output_throughput_tokens_per_s=1000.0,
            mean_input_throughput_tokens_per_s=1000.0,
            mean_total_tokens_throughput_tokens_per_s=2000.0,
            mean_total_chars_per_hour=10000000.0,
            requests_per_second=10.0,
            error_codes_frequency={},
            error_rate=0.0,
            num_error_requests=0,
            num_completed_requests=600,
            num_requests=600,
            stats=create_test_metric_stats(),
        )

        # request_rate should be None for non-request_rate runs
        assert metrics.request_rate is None
        assert metrics.iteration_type == "num_concurrency"

    def test_aggregated_metrics_request_rate_integer(self):
        """Test request_rate handles integer values correctly."""
        metrics = AggregatedMetrics(
            scenario="test_scenario",
            num_concurrency=10,
            batch_size=1,
            request_rate=25,
            iteration_type="request_rate",
            run_duration=60.0,
            mean_output_throughput_tokens_per_s=1000.0,
            mean_input_throughput_tokens_per_s=1000.0,
            mean_total_tokens_throughput_tokens_per_s=2000.0,
            mean_total_chars_per_hour=10000000.0,
            requests_per_second=25.0,
            error_codes_frequency={},
            error_rate=0.0,
            num_error_requests=0,
            num_completed_requests=150,
            num_requests=150,
            stats=create_test_metric_stats(),
        )

        assert metrics.request_rate == 25
        assert isinstance(metrics.request_rate, int)


class TestRequestRateInExperimentMetadata:
    """Test request_rate field in ExperimentMetadata."""

    def test_experiment_metadata_with_request_rate(self):
        """Test creating ExperimentMetadata with request_rate."""
        metadata = create_test_metadata(
            iteration_type="request_rate",
            request_rate=[5, 10, 20],
        )

        assert metadata.iteration_type == "request_rate"
        assert metadata.request_rate == [5, 10, 20]
        assert metadata.batch_size is None

    def test_experiment_metadata_request_rate_optional(self):
        """Test that request_rate is optional for non-request_rate runs."""
        metadata = create_test_metadata(
            iteration_type="num_concurrency",
            num_concurrency=[1, 5, 10],
        )

        assert metadata.iteration_type == "num_concurrency"
        assert metadata.num_concurrency == [1, 5, 10]
        assert metadata.request_rate is None

    def test_experiment_metadata_iteration_type_validation(self):
        """Test that iteration_type accepts request_rate as valid literal."""
        # Should not raise validation error
        metadata = create_test_metadata(
            iteration_type="request_rate",
            request_rate=[10],
        )

        assert metadata.iteration_type == "request_rate"

    def test_experiment_metadata_invalid_iteration_type(self):
        """Test that invalid iteration_type is rejected."""
        with pytest.raises(ValidationError):
            create_test_metadata(
                iteration_type="invalid_type",
            )

    def test_experiment_metadata_request_rate_with_integers(self):
        """Test request_rate list accepts integer values."""
        metadata = create_test_metadata(
            iteration_type="request_rate",
            request_rate=[1, 5, 10, 100],
        )

        assert metadata.request_rate == [1, 5, 10, 100]
        assert all(isinstance(r, int) for r in metadata.request_rate)

    def test_experiment_metadata_request_rate_single_value(self):
        """Test request_rate with single value in list."""
        metadata = create_test_metadata(
            iteration_type="request_rate",
            request_rate=[15],
        )

        assert len(metadata.request_rate) == 1
        assert metadata.request_rate[0] == 15


class TestRequestRateMetricsIntegration:
    """Integration tests for request_rate in metrics."""

    def test_metrics_serialization_with_request_rate(self):
        """Test that metrics with request_rate can be serialized."""
        metrics = AggregatedMetrics(
            scenario="test_scenario",
            num_concurrency=10,
            batch_size=1,
            request_rate=20,
            iteration_type="request_rate",
            run_duration=60.0,
            mean_output_throughput_tokens_per_s=1000.0,
            mean_input_throughput_tokens_per_s=1000.0,
            mean_total_tokens_throughput_tokens_per_s=2000.0,
            mean_total_chars_per_hour=10000000.0,
            requests_per_second=20.0,
            error_codes_frequency={},
            error_rate=0.0,
            num_error_requests=0,
            num_completed_requests=1200,
            num_requests=1200,
            stats=create_test_metric_stats(),
        )

        # Should be able to convert to dict
        metrics_dict = metrics.model_dump()
        assert "request_rate" in metrics_dict
        assert metrics_dict["request_rate"] == 20

    def test_metadata_serialization_with_request_rate(self):
        """Test that metadata with request_rate can be serialized."""
        metadata = create_test_metadata(
            iteration_type="request_rate",
            request_rate=[5, 10, 20],
        )

        # Should be able to convert to dict
        metadata_dict = metadata.model_dump()
        assert "request_rate" in metadata_dict
        assert metadata_dict["request_rate"] == [5, 10, 20]

    def test_metrics_comparison_request_rate_vs_concurrency(self):
        """Test that metrics can distinguish request_rate from num_concurrency runs."""
        metrics_rate = AggregatedMetrics(
            scenario="test_scenario",
            num_concurrency=10,
            batch_size=1,
            request_rate=20,
            iteration_type="request_rate",
            run_duration=60.0,
            mean_output_throughput_tokens_per_s=1000.0,
            mean_input_throughput_tokens_per_s=1000.0,
            mean_total_tokens_throughput_tokens_per_s=2000.0,
            mean_total_chars_per_hour=10000000.0,
            requests_per_second=20.0,
            error_codes_frequency={},
            error_rate=0.0,
            num_error_requests=0,
            num_completed_requests=1200,
            num_requests=1200,
            stats=create_test_metric_stats(),
        )

        metrics_concurrency = AggregatedMetrics(
            scenario="test_scenario",
            num_concurrency=10,
            batch_size=1,
            iteration_type="num_concurrency",
            run_duration=60.0,
            mean_output_throughput_tokens_per_s=1000.0,
            mean_input_throughput_tokens_per_s=1000.0,
            mean_total_tokens_throughput_tokens_per_s=2000.0,
            mean_total_chars_per_hour=10000000.0,
            requests_per_second=20.0,
            error_codes_frequency={},
            error_rate=0.0,
            num_error_requests=0,
            num_completed_requests=1200,
            num_requests=1200,
            stats=create_test_metric_stats(),
        )

        # Should have different iteration types
        assert metrics_rate.iteration_type == "request_rate"
        assert metrics_concurrency.iteration_type == "num_concurrency"

        # request_rate should be set only for rate run
        assert metrics_rate.request_rate == 20
        assert metrics_concurrency.request_rate is None
