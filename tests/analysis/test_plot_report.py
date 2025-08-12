import logging
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from genai_bench.analysis.plot_report import (
    extract_traffic_scenarios,
    finalize_and_save_plots,
    get_group_data,
    get_scenario_data,
    plot_error_rates,
    plot_experiment_data,
    plot_graph,
    plot_metrics,
    plot_single_scenario_inference_speed_vs_throughput,
    save_individual_subplots,
)
from genai_bench.metrics.metrics import AggregatedMetrics


@pytest.fixture
def mock_metrics_data():
    """Create mock metrics data with valid and invalid entries."""
    # Create a mock StatField for output_inference_speed
    mock_stat_field = MagicMock()
    mock_stat_field.mean = 10.0

    # Create another mock StatField for the second entry
    mock_stat_field2 = MagicMock()
    mock_stat_field2.mean = 20.0

    # Create mock MetricStats
    mock_stats1 = MagicMock()
    mock_stats1.output_inference_speed = mock_stat_field

    mock_stats2 = MagicMock()
    mock_stats2.output_inference_speed = mock_stat_field2

    # Create MetricStats for concurrency level 3 with missing output_inference_speed
    mock_stats3 = MagicMock()
    mock_stats3.output_inference_speed = None

    return {
        1: {
            "aggregated_metrics": MagicMock(
                stats=mock_stats1,
                mean_output_throughput_tokens_per_s=100.0,
            )
        },
        2: {
            "aggregated_metrics": MagicMock(
                stats=mock_stats2,
                mean_output_throughput_tokens_per_s=200.0,
            )
        },
        # Concurrency level 3 has missing output_inference_speed
        3: {
            "aggregated_metrics": MagicMock(
                stats=mock_stats3,
                mean_output_throughput_tokens_per_s=300.0,
            )
        },
        # Concurrency level 4 has invalid structure
        4: {},
    }


@pytest.fixture
def mock_scenario_metrics_data(mock_metrics_data):
    return {"data": mock_metrics_data, "num_concurrency": [1, 2]}


@patch("genai_bench.analysis.plot_report.plt")
@patch("genai_bench.analysis.plot_report.plot_graph")
def test_plot_single_scenario_success(
    mock_plot_graph, mock_plt, mock_scenario_metrics_data, tmp_path, caplog
):
    """Test successful plotting with valid data."""
    # Setup
    task = "text-to-text"
    iteration_type = "num_concurrency"
    scenario_label = "test_scenario"
    experiment_folder = str(tmp_path)

    # Mock the subplot creation
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)

    with caplog.at_level(logging.WARNING):
        plot_single_scenario_inference_speed_vs_throughput(
            scenario_label,
            experiment_folder,
            task,
            mock_scenario_metrics_data,
            iteration_type,
        )

    mock_plot_graph.assert_called_once_with(
        ax=mock_ax,
        x_data=[100.0, 200.0],
        y_data=[10.0, 20.0],
        x_label="Output Throughput of Server (tokens/s)",
        y_label="Output Inference Speed per Request (tokens/s)",
        title="Output Inference Speed per Request vs Output Throughput of Server "
        "- test_scenario",
        concurrency_levels=[1, 2],
        label="Scenario: test_scenario",
        plot_type="line",
    )
    assert mock_plt.savefig.called
    assert mock_plt.close.called
    assert len(caplog.records) == 0  # No warnings should be logged


@patch("genai_bench.analysis.plot_report.plt")
@patch("genai_bench.analysis.plot_report.plot_graph")
def test_plot_single_scenario_with_missing_data(
    mock_plot_graph, mock_plt, mock_metrics_data, tmp_path, caplog
):
    """Test plotting with some missing or invalid data."""
    # Setup
    task = "text-to-text"
    iteration_type = "num_concurrency"
    concurrency_levels = [1, 2, 3, 4]  # Including levels with missing/invalid data
    scenario_metrics = {"data": mock_metrics_data, iteration_type: concurrency_levels}
    scenario_label = "test_scenario"
    experiment_folder = str(tmp_path)

    # Mock the subplot creation
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)

    with caplog.at_level(logging.WARNING):
        plot_single_scenario_inference_speed_vs_throughput(
            scenario_label, experiment_folder, task, scenario_metrics, iteration_type
        )

    # Should still plot with valid data points
    mock_plot_graph.assert_called_once_with(
        ax=mock_ax,
        x_data=[100.0, 200.0],  # Only data from concurrency levels 1 and 2
        y_data=[10.0, 20.0],
        x_label="Output Throughput of Server (tokens/s)",
        y_label="Output Inference Speed per Request (tokens/s)",
        title="Output Inference Speed per Request vs Output Throughput of Server "
        "- test_scenario",
        concurrency_levels=[1, 2],
        label="Scenario: test_scenario",
        plot_type="line",
    )

    # Verify warnings were logged for missing data
    assert len(caplog.records) == 2  # Two warning messages
    assert "Missing inference speed data for concurrency level 3" in caplog.text
    assert "Missing inference speed data for concurrency level 4" in caplog.text


@patch("genai_bench.analysis.plot_report.plt")
@patch("genai_bench.analysis.plot_report.plot_graph")
def test_plot_single_scenario_all_invalid_data(
    mock_plot_graph, mock_plt, tmp_path, caplog
):
    """Test handling of case where all data points are invalid."""
    # Setup
    task = "text-to-text"
    iteration_type = "num_concurrency"
    invalid_metrics_data = {
        1: {
            "aggregated_metrics": MagicMock(stats=AggregatedMetrics)
        },  # Missing required data
        2: {},  # Invalid structure
    }
    concurrency_levels = [1, 2]
    scenario_metrics = {
        "data": invalid_metrics_data,
        iteration_type: concurrency_levels,
    }
    scenario_label = "test_scenario"
    experiment_folder = str(tmp_path)

    # Mock the subplot creation
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)

    with caplog.at_level(logging.WARNING):
        plot_single_scenario_inference_speed_vs_throughput(
            scenario_label, experiment_folder, task, scenario_metrics, iteration_type
        )

    mock_plot_graph.assert_not_called()  # Should not attempt to plot
    assert mock_plt.close.called  # Should close figure
    assert "No valid inference speed data found" in caplog.text


@patch("genai_bench.analysis.plot_report.plt")
@patch("genai_bench.analysis.plot_report.plot_graph")
def test_plot_single_scenario_empty_input(mock_plot_graph, mock_plt, tmp_path, caplog):
    """Test handling of empty input data."""
    # Setup
    task = "text-to-text"
    iteration_type = "num_concurrency"
    scenario_metrics = {"data": {}, iteration_type: []}
    scenario_label = "test_scenario"
    experiment_folder = str(tmp_path)

    # Mock the subplot creation
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)

    with caplog.at_level(logging.WARNING):
        plot_single_scenario_inference_speed_vs_throughput(
            scenario_label, experiment_folder, task, scenario_metrics, iteration_type
        )

    mock_plot_graph.assert_not_called()  # Should not attempt to plot
    assert mock_plt.close.called  # Should close figure
    assert "No valid inference speed data found" in caplog.text


@patch("genai_bench.analysis.plot_report.plt")
@patch("genai_bench.analysis.plot_report.plot_graph")
def test_plot_single_scenario_rerank(mock_plot_graph, mock_plt, tmp_path, caplog):
    """Test handling of empty input data."""
    # Setup
    task = "text-to-rerank"
    iteration_type = "num_concurrency"
    scenario_metrics = {"data": {}, iteration_type: []}
    scenario_label = "test_scenario"
    experiment_folder = str(tmp_path)

    # Mock the subplot creation
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)

    with caplog.at_level(logging.WARNING):
        plot_single_scenario_inference_speed_vs_throughput(
            scenario_label, experiment_folder, task, scenario_metrics, iteration_type
        )

    mock_plot_graph.assert_not_called()
    mock_plt.close.assert_not_called()


def test_plot_graph_line():
    ax = MagicMock()
    ax.get_xlim.return_value = (0, 10)
    ax.get_ylim.return_value = (0, 10)
    ax.get_yscale.return_value = "linear"
    x_data = [1, 2, 3]
    y_data = [10, 20, 30]
    x_label = "X Axis"
    y_label = "Y Axis"
    title = "Line Plot Test"
    concurrency_levels = [1, 2, 3]
    label = "LineLabel"

    plot_graph(
        ax,
        x_data,
        y_data,
        x_label,
        y_label,
        title,
        concurrency_levels,
        label,
        plot_type="line",
    )

    # For a line plot (and non-"Concurrency" x_label) expect ax.plot and annotations.
    ax.plot.assert_called_once_with(x_data, y_data, "-o", label=label)
    assert ax.annotate.call_count == 3
    ax.set_xlabel.assert_called_with(x_label)
    ax.set_ylabel.assert_called_with(y_label)
    ax.set_title.assert_called_with(title)


def test_plot_graph_scatter():
    ax = MagicMock()
    ax.get_xlim.return_value = (0, 10)
    ax.get_ylim.return_value = (0, 10)
    ax.get_yscale.return_value = "linear"
    x_data = [1, 2, 3]
    y_data = [10, 20, 30]
    x_label = "X Axis"
    y_label = "Y Axis"
    title = "Scatter Plot Test"
    concurrency_levels = [1, 2, 3]
    label = "ScatterLabel"

    plot_graph(
        ax,
        x_data,
        y_data,
        x_label,
        y_label,
        title,
        concurrency_levels,
        label,
        plot_type="scatter",
    )

    ax.scatter.assert_called_once_with(x_data, y_data, label=label)
    assert ax.annotate.call_count == 3


def test_plot_graph_concurrency():
    """When x_label is 'Concurrency', x_data is replaced by evenly spaced positions."""
    ax = MagicMock()
    ax.get_xlim.return_value = (0, 10)
    ax.get_ylim.return_value = (0, 10)
    ax.get_yscale.return_value = "linear"
    x_data = [10, 20, 30]
    y_data = [0.5, 1.0, 2.0]
    x_label = "Concurrency"
    y_label = "Some Y"
    title = "Concurrency Plot"
    concurrency_levels = [100, 200, 300]
    label = "ConcurrencyTest"

    plot_graph(ax, x_data, y_data, x_label, y_label, title, concurrency_levels, label)

    # Check that custom x-tick positions and labels are set
    ax.set_xticks.assert_called_once_with(range(len(concurrency_levels)))
    ax.set_xticklabels.assert_called_once_with(concurrency_levels)
    # Check that the plot was made (using ax.plot by default)
    ax.plot.assert_called_once()


@patch("genai_bench.analysis.plot_report.plot_graph")
@patch("genai_bench.analysis.plot_report.plot_error_rates")
def test_plot_metrics(mock_plot_error_rates, mock_plot_graph):
    # Create dummy aggregated metrics for two concurrency levels.
    def create_dummy_metrics(
        speed, ttft, e2e_mean, e2e_p90, e2e_p99, throughput, total_throughput, rps
    ):
        agg = MagicMock()
        stats = MagicMock()
        stats.output_inference_speed.mean = speed
        stats.ttft.mean = ttft
        e2e = MagicMock()
        e2e.mean = e2e_mean
        e2e.p90 = e2e_p90
        e2e.p99 = e2e_p99
        stats.e2e_latency = e2e
        agg.stats = stats
        agg.mean_output_throughput_tokens_per_s = throughput
        agg.mean_total_tokens_throughput_tokens_per_s = total_throughput
        agg.requests_per_second = rps
        # For error rates:
        agg.error_codes_frequency = {404: 1}
        agg.num_requests = 10
        return agg

    # Note that in the current implementation there are 7 plot specifications.
    concurrency_data = {
        1: {
            "aggregated_metrics": create_dummy_metrics(
                10, 1, 0.5, 0.7, 0.9, 100, 300, 50
            )
        },
        2: {
            "aggregated_metrics": create_dummy_metrics(
                20, 2, 0.6, 0.8, 1.0, 200, 400, 60
            )
        },
    }
    concurrency_data_list = [concurrency_data]
    label_to_concurrency_map = {"Scenario: test": [1, 2]}
    labels = ["Scenario: test"]

    # Create a dummy 2x4 array of axes (using numpy for convenience)
    axs = np.empty((2, 4), dtype=object)
    for i in range(2):
        for j in range(4):
            axs[i, j] = MagicMock()

    plot_metrics(axs, concurrency_data_list, label_to_concurrency_map, labels)

    # There are now 7 plot specifications per scenario.
    assert mock_plot_graph.call_count == 7
    # And one call to plot_error_rates per scenario.
    mock_plot_error_rates.assert_called_once()


def test_get_scenario_data():
    dummy_metadata = MagicMock()
    run_data_list = [
        (dummy_metadata, {"scenario1": {1: "data1", 2: "data2"}}),
        (dummy_metadata, {"scenario2": {3: "data3"}}),
    ]
    label_to_concurrency_map, concurrency_data_list, labels = get_scenario_data(
        run_data_list
    )

    assert label_to_concurrency_map == {
        "Scenario: scenario1": [1, 2],
        "Scenario: scenario2": [3],
    }
    # The concurrency_data_list contains the raw dictionaries.
    assert concurrency_data_list == [{1: "data1", 2: "data2"}, {3: "data3"}]
    assert labels == ["Scenario: scenario1", "Scenario: scenario2"]


class DummyMetadata:
    def __init__(self, server_version, experiment_folder_name):
        self.server_version = server_version
        self.experiment_folder_name = experiment_folder_name


def test_get_group_data():
    metadata1 = DummyMetadata("v1", "/path/to/experiment1")
    metadata2 = DummyMetadata("v2", "/path/to/experiment2")
    run_data_list = [
        (metadata1, {"scenarioA": {1: "data1", 2: "data2"}}),
        (metadata2, {"scenarioA": {3: "data3"}}),
    ]

    # Test with a group key that is not "experiment_folder_name"
    label_to_concurrency_map, concurrency_data_list, labels = get_group_data(
        run_data_list, "scenarioA", "server_version"
    )
    assert label_to_concurrency_map == {
        "server_version: v1": [1, 2],
        "server_version: v2": [3],
    }
    assert labels == ["server_version: v1", "server_version: v2"]

    # Test with group_key "experiment_folder_name" (which uses os.path.basename)
    label_to_concurrency_map2, concurrency_data_list2, labels2 = get_group_data(
        run_data_list, "scenarioA", "experiment_folder_name"
    )
    assert label_to_concurrency_map2 == {
        "experiment_folder_name: experiment1": [1, 2],
        "experiment_folder_name: experiment2": [3],
    }
    assert labels2 == [
        "experiment_folder_name: experiment1",
        "experiment_folder_name: experiment2",
    ]


def test_extract_traffic_scenarios():
    dummy_metadata = MagicMock()
    run_data_list = [
        (dummy_metadata, {"scenarioA": {1: "data1"}}),
        (dummy_metadata, {"scenarioB": {2: "data2"}}),
        (dummy_metadata, {"scenarioA": {3: "data3"}}),
    ]
    scenarios = extract_traffic_scenarios(run_data_list)
    assert scenarios == {"scenarioA", "scenarioB"}


@patch("genai_bench.analysis.plot_report.save_individual_subplots")
@patch("genai_bench.analysis.plot_report.plt")
def test_finalize_and_save_plots(mock_plt, mock_save_individual_subplots, tmp_path):
    axs = MagicMock()
    fig = MagicMock()
    labels = ["Test Label"]
    experiment_folder = str(tmp_path)
    output_file_prefix = "test_prefix"

    finalize_and_save_plots(axs, fig, labels, experiment_folder, output_file_prefix)

    # Check that individual subplots were saved and the combined plot was saved.
    mock_save_individual_subplots.assert_called_once_with(
        axs, experiment_folder, output_file_prefix
    )
    expected_file = os.path.join(
        experiment_folder, f"{output_file_prefix}_combined_plots_2x4.png"
    )
    mock_plt.savefig.assert_called_with(expected_file)
    mock_plt.close.assert_called()


@patch("genai_bench.analysis.plot_report.plt")
def test_save_individual_subplots(mock_plt, tmp_path):
    # Set up a dummy axis with required methods/attributes.
    dummy_ax = MagicMock()
    dummy_ax.get_title.return_value = "Test_Title"
    # Set up dummy line and annotation objects for the axis.
    dummy_line = MagicMock()
    dummy_line.get_xdata.return_value = [1, 2, 3]
    dummy_line.get_ydata.return_value = [4, 5, 6]
    dummy_line.get_label.return_value = "line_label"
    dummy_line.get_marker.return_value = "o"
    dummy_annotation = MagicMock()
    dummy_annotation.get_text.return_value = "annotation_text"
    dummy_annotation.get_position.return_value = (1, 4)
    dummy_annotation.get_fontsize.return_value = 9
    dummy_annotation.get_ha.return_value = "center"
    dummy_annotation.get_va.return_value = "center"

    dummy_ax.get_lines.return_value = [dummy_line]
    dummy_ax.texts = [dummy_annotation]
    dummy_ax.get_yscale.return_value = "linear"
    dummy_ax.get_xlim.return_value = (0, 10)
    dummy_ax.get_ylim.return_value = (0, 10)
    dummy_ax.get_xlabel.return_value = "X Axis"
    dummy_ax.get_xticks.return_value = [0, 5, 10]
    dummy_ax.get_xticklabels.return_value = ["0", "5", "10"]
    dummy_ax.get_ylabel.return_value = "Y Axis"
    dummy_ax.get_legend_handles_labels.return_value = ([], [])

    class DummyAxesArray:
        @property
        def flat(self):
            return [dummy_ax]

    axs = DummyAxesArray()
    experiment_folder = str(tmp_path)
    output_file_prefix = "prefix"

    captured_figs = []

    def subplots_side_effect(*args, **kwargs):
        dummy_fig = MagicMock(name="dummy_fig")
        dummy_fig.get_figwidth.return_value = 3
        dummy_ax = MagicMock(name="dummy_ax")
        captured_figs.append(dummy_fig)
        return dummy_fig, dummy_ax

    mock_plt.subplots.side_effect = subplots_side_effect

    save_individual_subplots(axs, experiment_folder, output_file_prefix)
    expected_file = os.path.join(
        experiment_folder, f"{output_file_prefix}_Test_Title.png"
    )
    captured_figs[0].savefig.assert_called_with(expected_file)


def test_plot_error_rates():
    ax = MagicMock()
    # Ensure unpacking of legend handles/labels works.
    ax.get_legend_handles_labels.return_value = ([], [])
    # Provide y-limits for autoscale+pin logic
    ax.get_ylim.return_value = (0, 1)

    def create_agg(freq, num_requests):
        agg = MagicMock()
        agg.error_codes_frequency = freq
        agg.num_requests = num_requests
        return agg

    concurrency_data = {
        1: {"aggregated_metrics": create_agg({404: 1}, 10)},
        2: {"aggregated_metrics": create_agg({404: 2, 500: 1}, 20)},
    }
    concurrency_levels = [1, 2]
    label = "Error Rate Test"

    plot_error_rates(ax, concurrency_data, concurrency_levels, label)

    # Two error codes (404 and 500) should yield two calls to ax.bar.
    assert ax.bar.call_count == 2
    ax.set_xlabel.assert_called_with("Concurrency")
    ax.set_ylabel.assert_called_with("Error Rate")
    ax.set_title.assert_called_with("Error Rates by HTTP Status vs Concurrency")
    ax.set_ylim.assert_called()  # Ensuring y-limit is set (with bottom=0)
    ax.legend.assert_called()
    ax.grid.assert_called_with(True)


def test_plot_experiment_data_invalid_group_key(tmp_path):
    """Using an invalid group_key should raise a ValueError."""
    run_data_list = []
    experiment_folder = str(tmp_path)
    with pytest.raises(ValueError):
        plot_experiment_data(run_data_list, "invalid_key", experiment_folder)
