from locust import events
from locust.env import Environment
from locust.event import EventHook

import json
import logging
import os
from unittest.mock import MagicMock

import pytest

from genai_bench.metrics.aggregated_metrics_collector import AggregatedMetricsCollector
from genai_bench.metrics.metrics import RequestLevelMetrics
from genai_bench.metrics.request_metrics_collector import RequestMetricsCollector
from genai_bench.protocol import UserChatResponse, UserResponse


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
    mock_response.reasoning_tokens = None

    # Initialize and calculate request metrics
    request_metrics_collector = RequestMetricsCollector()
    request_metrics_collector.calculate_metrics(mock_response)

    # Check calculations
    assert request_metrics_collector.metrics.ttft == 100
    assert request_metrics_collector.metrics.e2e_latency == 110
    assert request_metrics_collector.metrics.num_input_tokens == 10
    assert request_metrics_collector.metrics.num_reasoning_tokens is None


def test_request_level_metrics_calculation_with_reasoning_tokens():
    mock_response = MagicMock(spec=UserChatResponse)
    mock_response.status_code = 200
    mock_response.generated_text = "and and and and"
    mock_response.tokens_received = 4
    mock_response.time_at_first_token = 1722986731
    mock_response.start_time = 1722986631
    mock_response.end_time = 1722986741
    mock_response.num_prefill_tokens = 10
    mock_response.reasoning_tokens = 5

    # Initialize and calculate request metrics
    request_metrics_collector = RequestMetricsCollector()
    request_metrics_collector.calculate_metrics(mock_response)

    assert request_metrics_collector.metrics.num_reasoning_tokens == 5


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


def test_filter_metrics(aggregated_metrics_collector):
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
