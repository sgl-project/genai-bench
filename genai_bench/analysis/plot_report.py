import os
import textwrap
from http import HTTPStatus
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from genai_bench.analysis.experiment_loader import ExperimentMetrics, MetricsData
from genai_bench.logging import init_logger
from genai_bench.protocol import ExperimentMetadata
from genai_bench.utils import sanitize_string

logger = init_logger(__name__)


def plot_graph(
    ax: Axes,
    x_data: List[Any],
    y_data: List[Any],
    x_label: str,
    y_label: str,
    title: str,
    concurrency_levels: List[int],
    label: str,
    plot_type: str = "line",
) -> None:
    """
    Generalized plotting function that can handle both line and scatter plots.

    Args:
        ax (Axes): Matplotlib axis on which to plot.
        x_data (list): Data for x-axis.
        y_data (list): Data for y-axis.
        x_label (str): Label for x-axis.
        y_label (str): Label for y-axis.
        title (str): Title of the plot.
        concurrency_levels (list): List of concurrency levels for annotations.
        label (str): Label for the legend.
        plot_type (str): Type of plot, either "line" or "scatter".
    """
    if x_label == "Concurrency":
        # Set x-axis for concurrency levels to have even spacing
        x_positions = range(len(concurrency_levels))
        ax.set_xticks(x_positions)
        ax.set_xticklabels(concurrency_levels)
        x_data = x_positions  # type: ignore[assignment]  # Replace x_data with even positions for plotting
    else:
        x_positions = x_data  # type: ignore[assignment]

    valid_x = x_data
    valid_y = y_data
    valid_concurrency = concurrency_levels

    # Plot data
    if plot_type == "line":
        ax.plot(valid_x, valid_y, "-o", label=label)
    else:
        ax.scatter(valid_x, valid_y, label=label)

    # Annotate
    for xx, yy, cc in zip(valid_x, valid_y, valid_concurrency, strict=False):
        annotation = f"{yy:.2f}" if x_label == "Concurrency" else f"{cc}"
        ax.annotate(
            annotation,
            (xx, yy),
            fontsize=9,
            xytext=(4, 4),
            textcoords="offset points",
            ha="left",
            bbox=dict(
                boxstyle="round,pad=0.2", facecolor="white", alpha=0.0, edgecolor="none"
            ),
        )

    if y_label == "TTFT":
        ax.set_yscale("log", base=10)
        ax.yaxis.set_major_locator(
            mticker.LogLocator(base=10.0, subs=[1.0], numticks=10)
        )
        ax.yaxis.set_minor_locator(
            mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100)
        )

    # Axis limits handling with autoscale re-enabled every draw
    # X-axis: allow Matplotlib to autoscale to include new data, then pin left=0
    ax.autoscale(enable=True, axis="x", tight=False)
    x_left, x_right = ax.get_xlim()
    ax.set_xlim(left=0.0, right=x_right)

    # Y-axis: re-autoscale first, then pin bottom=0 for linear scale only
    ax.autoscale(enable=True, axis="y", tight=False)
    if ax.get_yscale() != "log":
        y_bottom, y_top = ax.get_ylim()
        ax.set_ylim(bottom=0.0, top=y_top)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True)
    # Turn on minor ticks. Combined with the LogLocator above,
    # this gives fine-grained, labeled minor ticks
    ax.minorticks_on()


def plot_metrics(
    axs: Any,
    concurrency_data_list: List[Dict[int, MetricsData]],
    label_to_concurrency_map: Dict[str, List[int]],
    labels: List[str],
) -> None:
    """
    Plots various metrics on provided axes and saves each plot individually.

    Args:
        axs (array-like): Array of matplotlib axes on which the plots are drawn.
        concurrency_data_list (list): List of concurrency data dictionaries for
            each label, where each dictionary contains the data for a
            specific concurrency level.
        label_to_concurrency_map (dict): Mapping of each label to its associated
            concurrency levels. Keys are labels (e.g., traffic scenarios or
            group names), and values are lists of concurrency levels associated
            with each label.
        labels (list): List of labels corresponding to each set of concurrency
            data, such as different scenarios or experiment groups.
    """
    for i, concurrency_data in enumerate(concurrency_data_list):
        concurrency_levels = label_to_concurrency_map[labels[i]]

        # Define all plot specifications in a single list
        plot_specs = [  # type: ignore[union-attr,index]
            # First row
            {
                "y_data": [
                    concurrency_data[c][
                        "aggregated_metrics"
                    ].stats.output_inference_speed.mean
                    for c in concurrency_levels
                ],
                "x_data": [
                    concurrency_data[c][
                        "aggregated_metrics"
                    ].mean_output_throughput_tokens_per_s
                    for c in concurrency_levels
                ],
                "x_label": "Output Throughput of Server (tokens/s)",
                "y_label": "Output Inference Speed per Request (tokens/s)",
                "title": "Output Inference Speed per Request "
                "vs Output Throughput of Server",
                "plot_type": "line",
                "ax": axs[0, 0],
            },
            {
                "y_data": [
                    concurrency_data[c]["aggregated_metrics"].stats.ttft.mean
                    for c in concurrency_levels
                ],
                "x_data": [
                    concurrency_data[c][
                        "aggregated_metrics"
                    ].mean_output_throughput_tokens_per_s
                    for c in concurrency_levels
                ],
                "x_label": "Output Throughput of Server (tokens/s)",
                "y_label": "TTFT",
                "title": "TTFT vs Output Throughput of Server",
                "plot_type": "line",
                "ax": axs[0, 1],
            },
            {
                "y_data": [
                    concurrency_data[c]["aggregated_metrics"].stats.e2e_latency.mean
                    for c in concurrency_levels
                ],
                "x_data": [
                    concurrency_data[c]["aggregated_metrics"].requests_per_second
                    for c in concurrency_levels
                ],
                "x_label": "RPS (req/s)",
                "y_label": "Mean E2E Latency per Request (s)",
                "title": "Mean E2E Latency per Request vs RPS",
                "plot_type": "line",
                "ax": axs[0, 2],
            },
            # Second row
            {
                "y_data": [
                    concurrency_data[c][
                        "aggregated_metrics"
                    ].stats.output_inference_speed.mean
                    for c in concurrency_levels
                ],
                "x_data": [
                    concurrency_data[c][
                        "aggregated_metrics"
                    ].mean_total_tokens_throughput_tokens_per_s
                    for c in concurrency_levels
                ],
                "x_label": "Total Throughput (Input + Output) of Server (tokens/s)",
                "y_label": "Output Inference Speed per Request (tokens/s)",
                "title": "Output Inference Speed per Request vs "
                "Total Throughput (Input + Output) of Server",
                "plot_type": "line",
                "ax": axs[1, 0],
            },
            {
                "y_data": [
                    concurrency_data[c]["aggregated_metrics"].stats.ttft.mean
                    for c in concurrency_levels
                ],
                "x_data": [
                    concurrency_data[c][
                        "aggregated_metrics"
                    ].mean_total_tokens_throughput_tokens_per_s
                    for c in concurrency_levels
                ],
                "x_label": "Total Throughput (Input + Output) of Server (tokens/s)",
                "y_label": "TTFT",
                "title": "TTFT vs Total Throughput (Input + Output) of Server",
                "plot_type": "line",
                "ax": axs[1, 1],
            },
            {
                "y_data": [
                    concurrency_data[c]["aggregated_metrics"].stats.e2e_latency.p90
                    for c in concurrency_levels
                ],
                "x_data": [
                    concurrency_data[c]["aggregated_metrics"].requests_per_second
                    for c in concurrency_levels
                ],
                "x_label": "RPS (req/s)",
                "y_label": "P90 E2E Latency per Request (s)",
                "title": "P90 E2E Latency per Request vs RPS",
                "plot_type": "line",
                "ax": axs[1, 2],
            },
            {
                "y_data": [
                    concurrency_data[c]["aggregated_metrics"].stats.e2e_latency.p99
                    for c in concurrency_levels
                ],
                "x_data": [
                    concurrency_data[c]["aggregated_metrics"].requests_per_second
                    for c in concurrency_levels
                ],
                "x_label": "RPS (req/s)",
                "y_label": "P99 E2E Latency per Request (s)",
                "title": "P99 E2E Latency per Request vs RPS",
                "plot_type": "line",
                "ax": axs[1, 3],
            },
        ]

        # Generate all plots
        for spec in plot_specs:
            plot_graph(
                ax=spec["ax"],
                x_data=spec["x_data"],
                y_data=spec["y_data"],
                x_label=spec["x_label"],
                y_label=spec["y_label"],
                title=spec["title"],
                concurrency_levels=concurrency_levels,
                label=labels[i],
                plot_type=spec["plot_type"],
            )

        # Add stacked error rate plot
        plot_error_rates(
            ax=axs[0, 3],
            concurrency_data=concurrency_data,
            concurrency_levels=concurrency_levels,
            label=labels[i],
        )


def plot_experiment_data(
    run_data_list: List[Tuple[ExperimentMetadata, ExperimentMetrics]],
    group_key: str,
    experiment_folder: str,
) -> None:
    """
    Plots experiment data in a 2x3 grid, grouping by traffic scenario or the
    specified group key.

    Args:
        run_data_list (list): A list of tuples, each containing data from
            individual experiments and their associated metadata.
            Each tuple follows this structure:

                [
                    (experiment_1_metadata, experiment_1_run_data),
                    (experiment_2_metadata, experiment_2_run_data),
                    ...
                ]

            In each experiment tuple:

            - `experiment_n_metadata` (dict): Metadata for the nth experiment.
            - `experiment_n_run_data` (dict): Metrics for the nth experiment,
              organized as follows:

                {
                    "<traffic-scenario>": {
                        "<concurrency-level>": {
                            "aggregated_metrics": <aggregated metrics>,
                            "individual_request_metrics": <individual metrics>
                        }
                    }
                }
        group_key (str): Key to group the data by (e.g., 'traffic_scenario',
            'server_version').
        experiment_folder (str): Path to the folder where the plots will be
            saved.
    """
    if group_key not in ExperimentMetadata.model_fields:
        raise ValueError(
            f"Invalid group_key '{group_key}' in ExperimentMetadata fields."
        )

    if group_key == "traffic_scenario":
        fig, axs = plt.subplots(2, 4, figsize=(32, 12))
        fig.suptitle("Grouped by Traffic Scenario", fontsize=14)
        label_to_concurrency_map, concurrency_data_list, labels = get_scenario_data(
            run_data_list
        )
        plot_metrics(axs, concurrency_data_list, label_to_concurrency_map, labels)
        finalize_and_save_plots(axs, fig, labels, experiment_folder, "traffic_scenario")
    else:
        traffic_scenarios = extract_traffic_scenarios(run_data_list)
        for traffic_scenario in traffic_scenarios:
            fig, axs = plt.subplots(2, 4, figsize=(32, 12))
            fig.suptitle(f"Traffic Scenario: {traffic_scenario}", fontsize=14)
            label_to_concurrency_map, concurrency_data_list, labels = get_group_data(
                run_data_list, traffic_scenario, group_key
            )
            plot_metrics(axs, concurrency_data_list, label_to_concurrency_map, labels)
            finalize_and_save_plots(
                axs,
                fig,
                labels,
                experiment_folder,
                f"{sanitize_string(traffic_scenario)}_group_by_{group_key}",
            )


def get_scenario_data(
    run_data_list: List[Tuple[ExperimentMetadata, ExperimentMetrics]],
) -> Tuple[Dict[str, List[int]], List[Dict[int, MetricsData]], List[str]]:
    """
    Collects all traffic scenarios and returns the required data structures for
    plotting.

    Args:
        run_data_list (list): List of tuples containing run data and experiment
        metadata.

    Returns:
        tuple:
            - label_to_concurrency_map (dict): A dictionary mapping each
              traffic scenario  label (as a string) to a sorted list of
              concurrency levels (integers) associated with that scenario.
              The key is in the format `"Scenario: <scenario_name>"`.

            - concurrency_data_list (list): A list of dictionaries, where each
              dictionary contains the run data (metrics) for a specific
              scenario across various concurrency levels. Each item in this
              list corresponds to one traffic scenario, and the dictionary keys
              represent different concurrency levels, while the values contain
              the aggregated and individual metrics at each level.

            - labels (list): A list of labels for each traffic scenario,
              formatted as `"Scenario: <scenario_name>"`. Each label in this
              list directly matches the keys in `label_to_concurrency_map` and
              is used for labeling plots.

    """
    label_to_concurrency_map = {}
    concurrency_data_list = []
    labels = []
    for metadata, run_data in run_data_list:
        for scenario, concurrency_data in run_data.items():
            label = f"Scenario: {scenario}"
            label_to_concurrency_map[label] = sorted(concurrency_data.keys())
            concurrency_data_list.append(concurrency_data)
            labels.append(label)
    return label_to_concurrency_map, concurrency_data_list, labels


def get_group_data(
    run_data_list: List[Tuple[ExperimentMetadata, ExperimentMetrics]],
    traffic_scenario: str,
    group_key: str,
) -> Tuple[Dict[str, List[int]], List[Dict[int, MetricsData]], List[str]]:
    """
    Collects data grouped by a specific group key for a particular traffic
    scenario.

    Args:
        run_data_list (list): List of tuples containing run data and experiment
            metadata.
        traffic_scenario (str): The traffic scenario to filter by.
        group_key (str): The group key to use for grouping the data.

    Returns:
        tuple: A tuple containing label-to-concurrency mapping, concurrency
            data list, and labels.
    """
    label_to_concurrency_map = {}
    concurrency_data_list = []
    labels = []
    for metadata, run_data in run_data_list:
        if traffic_scenario not in run_data:
            continue
        group_label = getattr(metadata, group_key, "Unknown")

        # extra base folder name from abspath
        if group_key == "experiment_folder_name":
            group_label = os.path.basename(group_label)

        label = f"{group_key}: {group_label}"
        label_to_concurrency_map[label] = sorted(run_data[traffic_scenario].keys())
        concurrency_data_list.append(run_data[traffic_scenario])
        labels.append(label)
    return label_to_concurrency_map, concurrency_data_list, labels


def extract_traffic_scenarios(
    run_data_list: List[Tuple[ExperimentMetadata, ExperimentMetrics]],
) -> set:
    """
    Extracts a set of unique traffic scenarios from the run data list.

    Args:
        run_data_list (list): List of tuples containing run data and experiment
            metadata.

    Returns:
        set: A set of unique traffic scenarios found in the run data.
    """
    traffic_scenarios: set[str] = set()
    for _, run_data in run_data_list:
        traffic_scenarios.update(run_data.keys())
    return traffic_scenarios


def finalize_and_save_plots(
    axs: Any,
    fig: Figure,
    labels: List[str],
    experiment_folder: str,
    output_file_prefix: str,
) -> None:
    """
    Finalizes plot settings and saves each plot as individual images and a
    combined image.

    Args:
        axs (array-like): Array of matplotlib axes from the main figure.
        fig (Figure): The main matplotlib figure object containing the subplots.
        labels (list): List of labels for the plot legend.
        experiment_folder (str): Path to the folder where the plots will be
            saved.
        output_file_prefix (str): Prefix for the output file names.
    """
    save_individual_subplots(axs, experiment_folder, output_file_prefix)

    fig.legend(labels=labels, loc="lower center", ncol=3, fancybox=True)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)

    output_file = os.path.join(
        experiment_folder, f"{output_file_prefix}_combined_plots_2x4.png"
    )
    logger.info(f"🎨 Saving {output_file}")
    plt.savefig(output_file)
    plt.close()


def save_individual_subplots(
    axs: Any, experiment_folder: str, output_file_prefix: str
) -> None:
    """
    Extracts and saves each subplot from a populated 2x3 grid figure as
    individual PNGs, using each subplot's title as the filename.

    Args:
        axs (array-like): The array of matplotlib axes from the main figure.
        experiment_folder (str): Directory path to save each individual subplot.
        output_file_prefix (str): Prefix of the output file to save individual.
    """
    for ax in axs.flat:
        title = ax.get_title()
        sanitized_title = sanitize_string(title)

        fig_temp, ax_temp = plt.subplots()

        # Copy the content of the original axis to the new axis
        for line in ax.get_lines():
            ax_temp.plot(
                line.get_xdata(),
                line.get_ydata(),
                label=line.get_label(),
                marker=line.get_marker(),
            )

        # Copy each annotation from the original axis to the new axis
        for annotation in ax.texts:
            ax_temp.annotate(
                annotation.get_text(),
                xy=annotation.get_position(),
                fontsize=annotation.get_fontsize(),
                ha=annotation.get_ha(),
                va=annotation.get_va(),
            )

        # Copy scale settings
        if ax.get_yscale() == "log":
            ax_temp.set_yscale("log")

        # Copy axis limits
        ax_temp.set_xlim(ax.get_xlim())
        ax_temp.set_ylim(ax.get_ylim())

        ax_temp.set_title(title)
        ax_temp.set_xlabel(ax.get_xlabel())
        ax_temp.set_xticks(ax.get_xticks())
        ax_temp.set_xticklabels(ax.get_xticklabels())
        ax_temp.set_ylabel(ax.get_ylabel())
        ax_temp.grid(ax.get_xgridlines())
        ax_temp.minorticks_on()

        # Calculate max_label_length based on figure width
        fig_width = fig_temp.get_figwidth()
        max_label_length = int(fig_width * 10)

        handles, labels = ax.get_legend_handles_labels()
        if handles and labels:
            # Wrap each label if it exceeds the calculated max_label_length
            wrapped_labels = [
                "\n".join(textwrap.wrap(label, max_label_length)) for label in labels
            ]
            ax_temp.legend(handles, wrapped_labels, fontsize="x-small", fancybox=True)

        fig_temp.tight_layout()
        output_file = os.path.join(
            experiment_folder, f"{output_file_prefix}_{sanitized_title}.png"
        )
        fig_temp.savefig(output_file)
        plt.close(fig_temp)
        logger.info(f"🎨 Saving {output_file}")


def plot_single_scenario_inference_speed_vs_throughput(
    scenario_label: str,
    experiment_folder: str,
    task: str,
    scenario_metrics: Dict[str, Any],
    iteration_type: str,
) -> None:
    """
    Plots metrics for a single scenario immediately after it completes.

    Args:
        scenario_label (str): Label for the scenario being plotted
        experiment_folder (str): Path to save the plot
        task: The benchmark task e.g. text-to-text
        scenario_metrics: A dict with two keys:
            "Data": Dictionary containing metrics data for different concurrency levels
                or batch
             iteration_type: List of concurrency levels/batch
        iteration_type: Type of iteration for the benchmark. E.g. concurrency levels.
    """
    # TODO: This logic should be de-coupled.
    if (
        task.split("-to-")[-1] == "embeddings" or task.split("-to-")[-1] == "rerank"
    ) and not scenario_metrics["data"]:
        return

    concurrency_data = scenario_metrics["data"]
    concurrency_levels = sorted(scenario_metrics[f"{iteration_type}"])
    fig, ax = plt.subplots(figsize=(10, 6))

    # Filter out concurrency levels with missing data and collect valid data points
    valid_x_data = []
    valid_y_data = []
    valid_concurrency = []

    for c in concurrency_levels:
        try:
            speed = concurrency_data[c][
                "aggregated_metrics"
            ].stats.output_inference_speed.mean
            output_throughput = concurrency_data[c][
                "aggregated_metrics"
            ].mean_output_throughput_tokens_per_s

            if speed is not None and output_throughput is not None:
                valid_y_data.append(speed)
                valid_x_data.append(output_throughput)
                valid_concurrency.append(c)
        except (KeyError, AttributeError) as e:
            logger.warning(
                f"Missing inference speed data for concurrency level {c} in scenario "
                f"{scenario_label}: {str(e)}"
            )
            continue

    if not valid_y_data or not valid_x_data:
        logger.warning(
            f"No valid inference speed data found for any concurrency level in "
            f"scenario {scenario_label}"
        )
        plt.close()
        return

    # Plot using only valid data points
    plot_graph(
        ax=ax,
        x_data=valid_x_data,
        y_data=valid_y_data,
        x_label="Output Throughput of Server (tokens/s)",
        y_label="Output Inference Speed per Request (tokens/s)",
        title=f"Output Inference Speed per Request vs "
        f"Output Throughput of Server - {scenario_label}",
        concurrency_levels=valid_concurrency,
        label=f"Scenario: {scenario_label}",
        plot_type="line",
    )

    ax.legend()
    ax.minorticks_on()
    plt.tight_layout()

    output_file = os.path.join(
        experiment_folder,
        f"interim_{sanitize_string(scenario_label)}_output_speed_vs_throughput.png",
    )
    logger.info(f"🎨 Saving interim plot: {output_file}")
    plt.savefig(output_file)
    plt.close()


def plot_error_rates(
    ax: Axes,
    concurrency_data: Dict[int, dict],
    concurrency_levels: List[int],
    label: str,
) -> None:
    """
    Creates a stacked bar plot of error rates by actual HTTP status code
    across concurrency levels. Uses http.HTTPStatus to label each code.

    Args:
        ax (Axes): Matplotlib axis to plot on
        concurrency_data (dict): Dictionary containing error data for each concurrency
        concurrency_levels (list): List of concurrency levels
        label (str): Label for the legend
    """
    all_codes = set()
    for c in concurrency_levels:
        aggregated = concurrency_data[c]["aggregated_metrics"]
        error_freq = aggregated.error_codes_frequency
        all_codes.update(error_freq.keys())

    # Convert to a sorted list so each code is consistently stacked
    all_codes = sorted(all_codes)  # type: ignore[assignment]

    bottom = np.zeros(len(concurrency_levels))
    color_map = plt.get_cmap("tab20")
    existing_handles, existing_labels = ax.get_legend_handles_labels()

    for idx, code in enumerate(all_codes):
        # Use HTTPStatus to get a nice textual label, e.g. "404 Not Found".
        # If Python's HTTPStatus doesn't recognize it, just show the numeric code.
        try:
            code_label = f"{code} {HTTPStatus(code).phrase}"
        except ValueError:
            code_label = str(code)

        error_rates = []
        for c_index, c_val in enumerate(concurrency_levels):
            aggregated = concurrency_data[c_val]["aggregated_metrics"]
            total_requests = aggregated.num_requests
            freq_dict = aggregated.error_codes_frequency

            count_this_code = freq_dict.get(code, 0)
            error_rate = count_this_code / total_requests if total_requests > 0 else 0
            error_rates.append(error_rate)

        ax.bar(
            concurrency_levels,
            error_rates,
            bottom=bottom,
            color=color_map(idx / len(all_codes)),
            label=code_label,
        )

        bottom += np.array(error_rates)

    ax.set_xlabel("Concurrency")
    ax.set_ylabel("Error Rate")
    ax.set_title("Error Rates by HTTP Status vs Concurrency")
    # Re-enable autoscale for y so subsequent groups can extend the top,
    # then pin bottom at 0 (valid for linear scale used here)
    ax.autoscale(enable=True, axis="y", tight=False)
    y_bottom, y_top = ax.get_ylim()
    ax.set_ylim(bottom=0.0, top=y_top)
    ax.legend()
    ax.grid(True)
