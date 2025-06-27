"""Flexible plotting system supporting user-defined plot configurations."""

import os
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from genai_bench.analysis.experiment_loader import ExperimentMetrics, MetricsData
from genai_bench.analysis.plot_config import PlotConfig, PlotConfigManager, PlotSpec
from genai_bench.analysis.plot_report import plot_error_rates, plot_graph
from genai_bench.logging import init_logger
from genai_bench.protocol import ExperimentMetadata
from genai_bench.utils import sanitize_string

logger = init_logger(__name__)


class FlexiblePlotGenerator:
    """Generate plots based on flexible configuration."""

    def __init__(self, config: PlotConfig):
        self.config = config

    def generate_plots(
        self,
        run_data_list: List[Tuple[ExperimentMetadata, ExperimentMetrics]],
        group_key: str,
        experiment_folder: str,
    ) -> None:
        """Generate plots based on configuration."""
        if group_key == "traffic_scenario":
            self._plot_by_scenario(run_data_list, experiment_folder)
        else:
            self._plot_by_group(run_data_list, group_key, experiment_folder)

    def _plot_by_scenario(
        self,
        run_data_list: List[Tuple[ExperimentMetadata, ExperimentMetrics]],
        experiment_folder: str,
    ) -> None:
        """Plot grouped by traffic scenario."""
        from genai_bench.analysis.plot_report import get_scenario_data

        label_to_concurrency_map, concurrency_data_list, labels = get_scenario_data(
            run_data_list
        )

        fig, axs = self._create_figure()
        fig.suptitle("Grouped by Traffic Scenario", fontsize=14)

        self._plot_metrics(axs, concurrency_data_list, label_to_concurrency_map, labels)
        self._finalize_and_save_plots(
            axs, fig, labels, experiment_folder, "traffic_scenario"
        )

    def _plot_by_group(
        self,
        run_data_list: List[Tuple[ExperimentMetadata, ExperimentMetrics]],
        group_key: str,
        experiment_folder: str,
    ) -> None:
        """Plot grouped by specified group key."""
        from genai_bench.analysis.plot_report import (
            extract_traffic_scenarios,
            get_group_data,
        )

        traffic_scenarios = extract_traffic_scenarios(run_data_list)
        for traffic_scenario in traffic_scenarios:
            fig, axs = self._create_figure()
            fig.suptitle(f"Traffic Scenario: {traffic_scenario}", fontsize=14)

            label_to_concurrency_map, concurrency_data_list, labels = get_group_data(
                run_data_list, traffic_scenario, group_key
            )

            self._plot_metrics(
                axs, concurrency_data_list, label_to_concurrency_map, labels
            )
            self._finalize_and_save_plots(
                axs,
                fig,
                labels,
                experiment_folder,
                f"{sanitize_string(traffic_scenario)}_group_by_{group_key}",
            )

    def _create_figure(self) -> Tuple[Figure, Any]:
        """Create figure with configured layout."""
        layout = self.config.layout
        figsize = layout.figsize or (8 * layout.cols, 6 * layout.rows)

        fig, axs = plt.subplots(layout.rows, layout.cols, figsize=figsize)

        # Ensure axs is always 2D array for consistent indexing
        if layout.rows == 1 and layout.cols == 1:
            axs = np.array([[axs]])
        elif layout.rows == 1:
            axs = axs.reshape(1, -1)
        elif layout.cols == 1:
            axs = axs.reshape(-1, 1)

        return fig, axs

    def _plot_metrics(
        self,
        axs: Any,
        concurrency_data_list: List[Dict[int, MetricsData]],
        label_to_concurrency_map: Dict[str, List[int]],
        labels: List[str],
    ) -> None:
        """Plot metrics based on configuration."""
        for i, concurrency_data in enumerate(concurrency_data_list):
            concurrency_levels = label_to_concurrency_map[labels[i]]

            for plot_spec in self.config.plots:
                self._plot_single_metric(
                    plot_spec=plot_spec,
                    ax=axs[plot_spec.position[0], plot_spec.position[1]],
                    concurrency_data=concurrency_data,
                    concurrency_levels=concurrency_levels,
                    label=labels[i],
                )

    def _plot_single_metric(
        self,
        plot_spec: PlotSpec,
        ax: Axes,
        concurrency_data: Dict[int, MetricsData],
        concurrency_levels: List[int],
        label: str,
    ) -> None:
        """Plot a single metric based on plot specification."""
        try:
            # Extract data using field paths
            x_data = []
            y_data = []
            valid_concurrency = []

            for c in concurrency_levels:
                try:
                    metrics = concurrency_data[c]["aggregated_metrics"]

                    x_val = PlotConfigManager.get_field_value(
                        metrics, plot_spec.x_field
                    )
                    y_val = PlotConfigManager.get_field_value(
                        metrics, plot_spec.y_field
                    )

                    if x_val is not None and y_val is not None:
                        x_data.append(x_val)
                        y_data.append(y_val)
                        valid_concurrency.append(c)

                except (AttributeError, KeyError) as e:
                    logger.warning(f"Missing data for concurrency {c}: {e}")
                    continue

            if not x_data or not y_data:
                logger.warning(f"No valid data for plot: {plot_spec.title}")
                return

            # Handle special error rate plot
            if plot_spec.y_field == "error_rate" and plot_spec.plot_type == "bar":
                plot_error_rates(ax, concurrency_data, concurrency_levels, label)
            else:
                # Use existing plot_graph function
                plot_graph(
                    ax=ax,
                    x_data=x_data,
                    y_data=y_data,
                    x_label=plot_spec.x_label
                    or self._generate_label(plot_spec.x_field),
                    y_label=plot_spec.y_label
                    or self._generate_label(plot_spec.y_field),
                    title=plot_spec.title,
                    concurrency_levels=valid_concurrency,
                    label=label,
                    plot_type=plot_spec.plot_type,
                )

        except Exception as e:
            logger.error(f"Error plotting {plot_spec.title}: {e}")
            ax.text(
                0.5,
                0.5,
                f"Error: {str(e)}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

    def _generate_label(self, field_path: str) -> str:
        """Generate a human-readable label from field path."""
        # Simple label generation - can be enhanced
        parts = field_path.split(".")
        if len(parts) == 1:
            return field_path.replace("_", " ").title()

        # Handle stats fields
        if parts[0] == "stats":
            metric = parts[1].replace("_", " ").title()
            if len(parts) > 2:
                stat = parts[2].upper()
                return f"{metric} ({stat})"
            return metric

        return field_path.replace("_", " ").title()

    def _finalize_and_save_plots(
        self,
        axs: Any,
        fig: Figure,
        labels: List[str],
        experiment_folder: str,
        output_file_prefix: str,
    ) -> None:
        """Finalize and save plots."""
        from genai_bench.analysis.plot_report import save_individual_subplots

        # Hide unused subplots
        layout = self.config.layout
        used_positions = {plot.position for plot in self.config.plots}

        for row in range(layout.rows):
            for col in range(layout.cols):
                if (row, col) not in used_positions:
                    axs[row, col].set_visible(False)

        # Save individual subplots
        save_individual_subplots(axs, experiment_folder, output_file_prefix)

        # Add legend and save combined plot
        fig.legend(labels=labels, loc="lower center", ncol=3, fancybox=True)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)

        output_file = os.path.join(
            experiment_folder,
            f"{output_file_prefix}_combined_plots_{layout.rows}x{layout.cols}.png",
        )
        logger.info(f"🎨 Saving {output_file}")
        plt.savefig(output_file)
        plt.close()


def plot_experiment_data_flexible(
    run_data_list: List[Tuple[ExperimentMetadata, ExperimentMetrics]],
    group_key: str,
    experiment_folder: str,
    plot_config: PlotConfig = None,
) -> None:
    """
    Plot experiment data using flexible configuration.

    Args:
        run_data_list: List of experiment data tuples
        group_key: Key to group data by
        experiment_folder: Output folder path
        plot_config: Plot configuration (uses default if None)
    """
    if plot_config is None:
        plot_config = PlotConfigManager.load_preset("2x4_default")

    generator = FlexiblePlotGenerator(plot_config)
    generator.generate_plots(run_data_list, group_key, experiment_folder)


def validate_plot_config_with_data(
    config: PlotConfig,
    sample_metrics: List[Tuple[ExperimentMetadata, ExperimentMetrics]],
) -> List[str]:
    """
    Validate plot configuration against actual data.

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    if not sample_metrics:
        errors.append("No sample data provided for validation")
        return errors

    # Get a sample aggregated metrics object
    try:
        _, experiment_metrics = sample_metrics[0]
        first_scenario = list(experiment_metrics.keys())[0]
        first_concurrency = list(experiment_metrics[first_scenario].keys())[0]
        sample_agg_metrics = experiment_metrics[first_scenario][first_concurrency][
            "aggregated_metrics"
        ]
    except (IndexError, KeyError) as e:
        errors.append(f"Cannot extract sample metrics for validation: {e}")
        return errors

    # Validate each plot specification
    for i, plot_spec in enumerate(config.plots):
        # Validate field paths
        if not PlotConfigManager.validate_field_path(
            plot_spec.x_field, sample_agg_metrics
        ):
            errors.append(f"Plot {i+1}: Invalid x_field '{plot_spec.x_field}'")

        if not PlotConfigManager.validate_field_path(
            plot_spec.y_field, sample_agg_metrics
        ):
            errors.append(f"Plot {i+1}: Invalid y_field '{plot_spec.y_field}'")

        # Validate position bounds
        layout = config.layout
        row, col = plot_spec.position
        if row >= layout.rows or col >= layout.cols:
            errors.append(
                f"Plot {i+1}: Position ({row}, {col}) exceeds layout bounds "
                f"({layout.rows-1}, {layout.cols-1})"
            )

    return errors
