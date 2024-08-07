import os
from unittest.mock import MagicMock

import numpy as np
import pytest

from genai_bench.ui.dashboard import (
    MinimalDashboard,
    RichLiveDashboard,
    create_dashboard,
)


# Helper function to calculate stats for live metrics
def calculate_stats(values):
    """Helper function to calculate min, max, mean, and percentiles."""
    stats = {
        "min": np.min(values).item(),
        "max": np.max(values).item(),
        "mean": np.mean(values).item(),
    }
    if len(values) > 0:
        percentiles = np.percentile(values, [50, 90, 99])
        stats["p50"] = percentiles[0].item()
        stats["p90"] = percentiles[1].item()
        stats["p99"] = percentiles[2].item()
    return stats


# Fixture to create a mock Dashboard object
@pytest.fixture
def mock_dashboard():
    os.environ["ENABLE_UI"] = "true"
    dashboard = create_dashboard()
    assert isinstance(dashboard, RichLiveDashboard)
    dashboard.benchmark_progress_task_id = 0
    dashboard.start_time = 0
    dashboard.run_time = 1
    dashboard.max_requests_per_run = 5
    return dashboard


# Test for handle_single_request method with everything works
def test_handle_single_request_no_error(mock_dashboard: create_dashboard):
    # Prepare the live metrics and calculate stats
    live_metrics = {
        "ttft": [0.5],
        "input_throughput": [100],
        "output_throughput": [200],
        "output_latency": [1.5],
        "stats": {
            "ttft": calculate_stats([0.5]),
            "input_throughput": calculate_stats([100]),
            "output_latency": calculate_stats([1.5]),
            "output_throughput": calculate_stats([200]),
        },
    }
    mock_dashboard.calculate_time_based_progress = MagicMock(return_value=50)
    mock_dashboard.update_benchmark_progress_bars = MagicMock()
    mock_dashboard.update_metrics_panels = MagicMock()
    mock_dashboard.update_histogram_panel = MagicMock()

    mock_dashboard.handle_single_request(
        live_metrics, total_requests=10, error_code=None
    )

    mock_dashboard.update_metrics_panels.assert_called_once_with(live_metrics)
    mock_dashboard.update_histogram_panel.assert_called_once_with(live_metrics)


# Test for handle_single_request method when an error occurs
def test_handle_single_request_with_error(mock_dashboard: create_dashboard):
    live_metrics = {
        "ttft": [],
        "input_throughput": [],
        "output_throughput": [],
        "output_latency": [],
        "stats": {},
    }
    mock_dashboard.calculate_time_based_progress = MagicMock(return_value=50)
    mock_dashboard.update_benchmark_progress_bars = MagicMock()
    mock_dashboard.update_metrics_panels = MagicMock()
    mock_dashboard.update_histogram_panel = MagicMock()

    mock_dashboard.handle_single_request(
        live_metrics, total_requests=10, error_code=500
    )

    # update_metrics_panels and update_histogram_panel should NOT be called
    # when there's an error
    mock_dashboard.update_metrics_panels.assert_not_called()
    mock_dashboard.update_histogram_panel.assert_not_called()


# Test for update_metrics_panels method when stats are provided
def test_update_metrics_panels_with_stats(mock_dashboard: create_dashboard):
    live_metrics = {
        "ttft": [0.5],
        "input_throughput": [100],
        "output_throughput": [200],
        "output_latency": [1.5],
        "stats": {
            "ttft": calculate_stats([0.5]),
            "input_throughput": calculate_stats([100]),
            "output_latency": calculate_stats([1.5]),
            "output_throughput": calculate_stats([200]),
        },
    }

    mock_dashboard.layout["input_throughput"].update = MagicMock()
    mock_dashboard.layout["input_latency"].update = MagicMock()
    mock_dashboard.layout["output_throughput"].update = MagicMock()
    mock_dashboard.layout["output_latency"].update = MagicMock()

    mock_dashboard.update_metrics_panels(live_metrics)

    mock_dashboard.layout["input_throughput"].update.assert_called()
    mock_dashboard.layout["input_latency"].update.assert_called()
    mock_dashboard.layout["output_throughput"].update.assert_called()
    mock_dashboard.layout["output_latency"].update.assert_called()


# Test for update_metrics_panels method when no data is provided
def test_update_metrics_panels_empty(mock_dashboard: create_dashboard):
    live_metrics = {
        "ttft": [],
        "input_throughput": [],
        "output_throughput": [],
        "output_latency": [],
        "stats": {},
    }

    mock_dashboard.layout["input_throughput"].update = MagicMock()
    mock_dashboard.layout["input_latency"].update = MagicMock()
    mock_dashboard.layout["output_throughput"].update = MagicMock()
    mock_dashboard.layout["output_latency"].update = MagicMock()

    mock_dashboard.update_metrics_panels(live_metrics)

    mock_dashboard.layout["input_throughput"].update.assert_not_called()
    mock_dashboard.layout["input_latency"].update.assert_not_called()
    mock_dashboard.layout["output_throughput"].update.assert_not_called()
    mock_dashboard.layout["output_latency"].update.assert_not_called()


# Test for update_histogram_panel method when data is present
def test_update_histogram_panel(mock_dashboard: create_dashboard):
    live_metrics = {
        "ttft": [0.5],
        "input_throughput": [100],
        "output_throughput": [200],
        "output_latency": [1.5],
        "stats": {},
    }

    mock_dashboard.layout["input_histogram"].update = MagicMock()
    mock_dashboard.layout["output_histogram"].update = MagicMock()

    mock_dashboard.update_histogram_panel(live_metrics)

    mock_dashboard.layout["input_histogram"].update.assert_called()
    mock_dashboard.layout["output_histogram"].update.assert_called()


@pytest.mark.parametrize(
    "enable_ui,expected_type",
    [
        ("true", RichLiveDashboard),
        ("false", MinimalDashboard),
        ("", MinimalDashboard),  # Test default case when env var is not set
    ],
)
def test_dashboard_factory_with_env_var(monkeypatch, enable_ui, expected_type):
    """
    Test that create_dashboard returns the correct dashboard type based on ENABLE_UI
    env var.
    """
    monkeypatch.setenv("ENABLE_UI", enable_ui)

    dashboard = create_dashboard()
    assert isinstance(dashboard, expected_type)
