import pytest
from genai_bench.metrics.aggregated_metrics_collector import AggregatedMetricsCollector
from genai_bench.metrics.metrics import RequestLevelMetrics


@pytest.fixture
def aggregated_metrics_collector():
    return AggregatedMetricsCollector()


def test_update_live_metrics_interval(aggregated_metrics_collector):
    """Test that live metrics are updated only after the specified interval."""
    from unittest.mock import patch

    # Set update interval to 1.0 second
    aggregated_metrics_collector.set_run_metadata(
        iteration=1, scenario_str="test", metrics_update_interval=1.0
    )

    metric1 = RequestLevelMetrics(
        ttft=0.1,
        tpot=0.1,
        e2e_latency=1.0,
        output_latency=0.9,
        input_throughput=10.0,
        output_throughput=10.0,
        num_input_tokens=10,
        num_output_tokens=10,
        total_tokens=20,
    )

    metric2 = RequestLevelMetrics(
        ttft=0.2,
        tpot=0.2,
        e2e_latency=2.0,
        output_latency=1.8,
        input_throughput=20.0,
        output_throughput=20.0,
        num_input_tokens=20,
        num_output_tokens=20,
        total_tokens=40,
    )

    # Mock time to control the flow
    with patch("time.time") as mock_time:
        start_time = 1000.0
        mock_time.return_value = start_time

        aggregated_metrics_collector.add_single_request_metrics(metric1)

        stats = aggregated_metrics_collector.get_live_metrics()["stats"]
        assert "ttft" in stats
        assert stats["ttft"]["mean"] == 0.1

        mock_time.return_value = start_time + 0.5
        aggregated_metrics_collector.add_single_request_metrics(metric2)

        stats = aggregated_metrics_collector.get_live_metrics()["stats"]

        assert len(aggregated_metrics_collector.get_live_metrics()["ttft"]) == 2
        assert stats["ttft"]["mean"] == 0.1

        mock_time.return_value = start_time + 1.1
        metric3 = RequestLevelMetrics(
            ttft=0.3,
            tpot=0.3,
            e2e_latency=3.0,
            output_latency=2.7,
            input_throughput=30.0,
            output_throughput=30.0,
            num_input_tokens=30,
            num_output_tokens=30,
            total_tokens=60,
        )
        aggregated_metrics_collector.add_single_request_metrics(metric3)

        stats = aggregated_metrics_collector.get_live_metrics()["stats"]
        assert len(aggregated_metrics_collector.get_live_metrics()["ttft"]) == 3
        assert stats["ttft"]["mean"] == pytest.approx(0.2)


def test_update_live_metrics_interval_invalid_value(aggregated_metrics_collector):
    """Test that setting an invalid update interval raises ValueError."""
    with pytest.raises(
        ValueError, match="metrics_update_interval must be non-negative"
    ):
        aggregated_metrics_collector.set_run_metadata(
            iteration=1, scenario_str="test", metrics_update_interval=-1.0
        )


def test_update_live_metrics_interval_invalid_type(aggregated_metrics_collector):
    """Test that setting an invalid type for update interval raises TypeError."""
    with pytest.raises(TypeError, match="metrics_update_interval must be a number"):
        aggregated_metrics_collector.set_run_metadata(
            iteration=1, scenario_str="test", metrics_update_interval="invalid"
        )


def test_clear_resets_last_update_time(aggregated_metrics_collector):
    """Test that clear() resets the last update time."""
    aggregated_metrics_collector._last_update_time = 12345.0
    aggregated_metrics_collector.clear()
    assert aggregated_metrics_collector._last_update_time == 0.0
