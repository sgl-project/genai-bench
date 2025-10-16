from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from genai_bench.cli.report import plot


@pytest.fixture
def cli_runner():
    """Create a CLI runner for testing."""
    return CliRunner()


@pytest.fixture
def mock_experiment_data():
    """Create mock experiment data."""
    metadata = Mock()
    metadata.metrics_time_unit = "s"  # Source time unit from experiment

    run_data = {
        "scenario1": {
            1: {
                "aggregated_metrics": Mock(
                    stats=Mock(ttft=Mock(mean=0.1), e2e_latency=Mock(mean=0.5)),
                    mean_output_throughput_tokens_per_s=100.0,
                ),
                "_time_unit": "s",
            }
        }
    }

    return [(metadata, run_data)]


def test_cli_metrics_time_unit_parameter_passed(cli_runner, mock_experiment_data):
    """Test that the --metrics-time-unit CLI parameter is correctly passed through."""
    with (
        patch(
            "genai_bench.cli.report.load_multiple_experiments",
            return_value=mock_experiment_data,
        ),
        patch(
            "genai_bench.analysis.flexible_plot_report.plot_experiment_data_flexible"
        ) as mock_plot,
    ):
        cli_runner.invoke(
            plot,
            [
                "--experiments-folder",
                "/tmp",
                "--group-key",
                "traffic_scenario",
                "--metrics-time-unit",
                "ms",
            ],
        )

        # Verify the plotting function was called with the correct time unit
        call_args = mock_plot.call_args
        assert call_args[1]["metrics_time_unit"] == "ms"


def test_cli_metrics_time_unit_default_value(cli_runner, mock_experiment_data):
    """Test that the default time unit is 's' when not specified."""
    with (
        patch(
            "genai_bench.cli.report.load_multiple_experiments",
            return_value=mock_experiment_data,
        ),
        patch(
            "genai_bench.analysis.flexible_plot_report.plot_experiment_data_flexible"
        ) as mock_plot,
    ):
        cli_runner.invoke(
            plot,
            [
                "--experiments-folder",
                "/tmp",
                "--group-key",
                "traffic_scenario",
                # No --metrics-time-unit specified
            ],
        )

        # Verify the plotting function was called with default time unit
        call_args = mock_plot.call_args
        assert call_args[1]["metrics_time_unit"] == "s"


def test_cli_metrics_time_unit_parameter_validation(cli_runner):
    """Test that invalid time unit values are rejected."""
    result = cli_runner.invoke(
        plot,
        [
            "--experiments-folder",
            "/tmp",
            "--group-key",
            "traffic_scenario",
            "--metrics-time-unit",
            "invalid_unit",
        ],
    )

    # Should fail with invalid choice error
    assert result.exit_code != 0
    assert "Invalid value for '--metrics-time-unit'" in result.output


def test_cli_metrics_time_unit_with_preset(cli_runner, mock_experiment_data):
    """Test that time unit works correctly with preset configurations."""
    with (
        patch(
            "genai_bench.cli.report.load_multiple_experiments",
            return_value=mock_experiment_data,
        ),
        patch(
            "genai_bench.analysis.flexible_plot_report.plot_experiment_data_flexible"
        ) as mock_plot,
    ):
        cli_runner.invoke(
            plot,
            [
                "--experiments-folder",
                "/tmp",
                "--group-key",
                "traffic_scenario",
                "--preset",
                "2x4_default",
                "--metrics-time-unit",
                "ms",
            ],
        )

        # Verify the plotting function was called with correct parameters
        mock_plot.assert_called_once()
        call_args = mock_plot.call_args
        assert call_args[1]["metrics_time_unit"] == "ms"
