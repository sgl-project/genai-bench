import os
from unittest.mock import MagicMock

import numpy as np
import pytest

from genai_bench.ui.dashboard import (
    MinimalDashboard,
    RichLiveDashboard,
    create_dashboard,
)
from genai_bench.ui.plots import create_scatter_plot


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
    dashboard = create_dashboard("s")
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

    mock_dashboard.update_metrics_panels.assert_called_once_with(live_metrics, "s")
    mock_dashboard.update_histogram_panel.assert_called_once_with(live_metrics, "s")


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
        "stats": {
            "ttft": {
                "mean": 0.1,
                "min": 0.05,
                "max": 0.2,
                "p50": 0.1,
                "p90": 0.15,
                "p99": 0.2,
            },
            "input_throughput": {"mean": 100, "min": 80, "max": 120},
            "output_latency": {
                "mean": 0.5,
                "min": 0.3,
                "max": 0.8,
                "p50": 0.5,
                "p90": 0.7,
                "p99": 0.8,
            },
            "output_throughput": {"mean": 50, "min": 40, "max": 60},
        }
    }
    mock_dashboard.update_metrics_panels = MagicMock()
    mock_dashboard.update_metrics_panels(live_metrics, "s")
    mock_dashboard.update_metrics_panels.assert_called_once_with(live_metrics, "s")


# Test for update_metrics_panels method when no data is provided
def test_update_metrics_panels_empty(mock_dashboard: create_dashboard):
    live_metrics = {"stats": []}
    mock_dashboard.update_metrics_panels = MagicMock()
    mock_dashboard.update_metrics_panels(live_metrics, "s")
    mock_dashboard.update_metrics_panels.assert_called_once_with(live_metrics, "s")


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

    mock_dashboard.update_histogram_panel(live_metrics, "s")

    mock_dashboard.layout["input_histogram"].update.assert_called()
    mock_dashboard.layout["output_histogram"].update.assert_called()


@pytest.mark.parametrize(
    "enable_ui,expected_type",
    [
        ("true", RichLiveDashboard),
        ("TRUE", RichLiveDashboard),  # Test uppercase
        ("True", RichLiveDashboard),  # Test mixed case
        ("1", RichLiveDashboard),  # Test numeric
        ("yes", RichLiveDashboard),  # Test yes
        ("YES", RichLiveDashboard),  # Test uppercase yes
        ("Yes", RichLiveDashboard),  # Test mixed case yes
        ("on", RichLiveDashboard),  # Test on
        ("ON", RichLiveDashboard),  # Test uppercase on
        ("On", RichLiveDashboard),  # Test mixed case on
        ("false", MinimalDashboard),
        ("FALSE", MinimalDashboard),  # Test uppercase false
        ("0", MinimalDashboard),  # Test numeric false
        ("no", MinimalDashboard),  # Test no
        ("off", MinimalDashboard),  # Test off
        ("", MinimalDashboard),  # Test default case when env var is not set
    ],
)
def test_dashboard_factory_with_env_var(monkeypatch, enable_ui, expected_type):
    """
    Test that create_dashboard returns the correct dashboard type based on ENABLE_UI
    env var. Supports multiple truthy values: true, TRUE, 1, yes, YES, on, ON (case-insensitive).
    """
    monkeypatch.setenv("ENABLE_UI", enable_ui)

    dashboard = create_dashboard("s")
    assert isinstance(dashboard, expected_type)


def test_scatter_plot_spacing_for_different_time_units():
    """Test that scatter plot spacing is correct for seconds vs milliseconds."""
    # Test data
    x_values = [100, 200, 300, 400, 500]
    y_values = [0.5, 1.0, 1.5, 2.0, 2.5]

    # Test with seconds
    plot_s = create_scatter_plot(x_values, y_values, y_unit="s")
    plot_s_str = str(plot_s)

    # Test with milliseconds
    plot_ms = create_scatter_plot(x_values, y_values, y_unit="ms")
    plot_ms_str = str(plot_ms)

    # Check that seconds plot uses 7 spaces for labels
    lines_s = plot_s_str.split("\n")
    label_line_s = None
    for line in lines_s:
        if "2.5" in line and "s" in line:
            label_line_s = line
            break

    assert label_line_s is not None, "Could not find label line with seconds"

    # Check that milliseconds plot uses 9 spaces for labels
    lines_ms = plot_ms_str.split("\n")
    label_line_ms = None
    for line in lines_ms:
        if "2.5" in line and "ms" in line:  # Scatter plot doesn't convert values
            label_line_ms = line
            break

    assert label_line_ms is not None, "Could not find label line with milliseconds"

    # Verify the label spacing
    assert (
        label_line_s.index("|") == 7
    ), f"Expected 7 spaces for seconds, got: {label_line_s.index('|')}"
    assert (
        label_line_ms.index("|") == 9
    ), f"Expected 9 spaces for milliseconds, got: {label_line_ms.index('|')}"


def test_minimal_dashboard_update_scatter_plot_does_not_crash():
    """
    Ensure MinimalDashboard.update_scatter_plot_panel works without a crash
    """
    dashboard = MinimalDashboard("s")

    mock_metrics = [0.1, 0.2, 10.0, 20.0]

    # Calling with metrics and explicit time unit should not raise
    dashboard.update_scatter_plot_panel(mock_metrics, "s")

    dashboard.update_scatter_plot_panel(mock_metrics, "ms")

    dashboard.update_scatter_plot_panel(None, "s")
