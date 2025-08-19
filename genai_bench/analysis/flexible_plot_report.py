"""Flexible plotting system supporting user-defined plot configurations."""

import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from genai_bench.analysis.experiment_loader import ExperimentMetrics, MetricsData
from genai_bench.analysis.plot_config import PlotConfig, PlotConfigManager, PlotSpec
from genai_bench.analysis.plot_report import (
    get_scenario_data,
    plot_error_rates,
    plot_graph,
)
from genai_bench.logging import init_logger
from genai_bench.protocol import ExperimentMetadata
from genai_bench.utils import sanitize_string
from genai_bench.time_units import TimeUnitConverter

logger = init_logger(__name__)


class FlexiblePlotGenerator:
    """Generates plots using flexible configuration."""

    def __init__(self, config: PlotConfig, time_unit: str = "s"):
        """Initialize with plot configuration and time unit."""
        self.config = config
        self.time_unit = time_unit

    def generate_plots(
        self,
        run_data_list: List[Tuple[ExperimentMetadata, ExperimentMetrics]],
        group_key: str,
        experiment_folder: str,
    ) -> None:
        """Generate plots based on configuration."""
        if group_key == "none":
            self._plot_single_analysis(run_data_list, experiment_folder)
        elif group_key == "traffic_scenario":
            self._plot_by_scenario(run_data_list, experiment_folder)
        else:
            self._plot_by_group(run_data_list, group_key, experiment_folder)

    def _plot_single_analysis(
        self,
        run_data_list: List[Tuple[ExperimentMetadata, ExperimentMetrics]],
        experiment_folder: str,
    ) -> None:
        """Plot for single scenario analysis without grouping - plots ALL scenarios."""
        if not run_data_list:
            logger.warning("No experiment data found for single analysis")
            return

        # Get the first experiment
        metadata, experiment_metrics = run_data_list[0]
        all_scenarios = list(experiment_metrics.keys())

        logger.info(
            f"Single scenario analysis mode - plotting all {len(all_scenarios)} "
            f"scenarios: {all_scenarios}"
        )

        # Plot each scenario individually for clean multi-line analysis
        for scenario in all_scenarios:
            concurrency_data = experiment_metrics[scenario]
            concurrency_levels = sorted(concurrency_data.keys())
            # Use stored time_unit instead of metadata.time_unit
            time_unit = self.time_unit

            fig, axs = self._create_figure()
            fig.suptitle(f"Single Scenario Analysis: {scenario}", fontsize=14)

            # Single scenario = no grouping, perfect for multi-line plots
            # Pass empty label since we're not grouping
            self._plot_metrics(axs, [concurrency_data], {"": concurrency_levels}, [""], time_unit)
            self._finalize_and_save_plots(
                axs,
                fig,
                [""],
                experiment_folder,
                f"single_analysis_{sanitize_string(scenario)}",
            )

    def _plot_by_scenario(
        self,
        run_data_list: List[Tuple[ExperimentMetadata, ExperimentMetrics]],
        experiment_folder: str,
    ) -> None:
        """Plot grouped by traffic scenario."""
        label_to_concurrency_map, concurrency_data_list, labels = get_scenario_data(
            run_data_list
        )

        # Use stored time_unit instead of extracting from metadata
        time_unit = self.time_unit

        fig, axs = self._create_figure()
        fig.suptitle("Grouped by Traffic Scenario", fontsize=14)

        self._plot_metrics(axs, concurrency_data_list, label_to_concurrency_map, labels, time_unit)
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

        # Use stored time_unit instead of extracting from metadata
        time_unit = self.time_unit

        traffic_scenarios = extract_traffic_scenarios(run_data_list)
        for traffic_scenario in traffic_scenarios:
            fig, axs = self._create_figure()
            fig.suptitle(f"Traffic Scenario: {traffic_scenario}", fontsize=14)

            label_to_concurrency_map, concurrency_data_list, labels = get_group_data(
                run_data_list, traffic_scenario, group_key
            )

            self._plot_metrics(
                axs, concurrency_data_list, label_to_concurrency_map, labels, time_unit
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
        time_unit: str = "s",
    ) -> None:
        """Plot metrics based on configuration."""
        # Check if we have multi-line plots with multiple scenarios/groups
        has_multi_line = any(plot.is_multi_line() for plot in self.config.plots)
        has_multiple_groups = len(labels) > 1

        if has_multi_line and has_multiple_groups:
            logger.warning(
                f"Multi-line plots detected with {len(labels)} groups/scenarios. "
                f"Multi-line plots work best with single scenarios. "
                f"Converting to single-line plots for better visualization."
            )
            # Convert multi-line plots to single-line plots automatically
            self._plot_metrics_single_line_fallback(
                axs, concurrency_data_list, label_to_concurrency_map, labels, time_unit
            )
            return

        for i, concurrency_data in enumerate(concurrency_data_list):
            concurrency_levels = label_to_concurrency_map[labels[i]]

            for plot_spec in self.config.plots:
                self._plot_single_metric(
                    plot_spec=plot_spec,
                    ax=axs[plot_spec.position[0], plot_spec.position[1]],
                    concurrency_data=concurrency_data,
                    concurrency_levels=concurrency_levels,
                    label=labels[i],
                    time_unit=time_unit,
                )

    def _plot_metrics_single_line_fallback(
        self,
        axs: Any,
        concurrency_data_list: List[Dict[int, MetricsData]],
        label_to_concurrency_map: Dict[str, List[int]],
        labels: List[str],
        time_unit: str = "s",
    ) -> None:
        """Fallback to single-line plotting when multi-line conflicts with grouping."""
        logger.info(
            "Converting multi-line plots to single-line plots for better "
            "visualization with multiple groups"
        )

        for i, concurrency_data in enumerate(concurrency_data_list):
            concurrency_levels = label_to_concurrency_map[labels[i]]

            for plot_spec in self.config.plots:
                if plot_spec.is_multi_line():
                    # For multi-line plots, just use the first Y-field
                    y_field_specs = plot_spec.get_y_field_specs()
                    first_field = y_field_specs[0]

                    # Create a temporary single-line plot spec
                    from genai_bench.analysis.plot_config import PlotSpec

                    single_line_spec = PlotSpec(
                        title=f"{plot_spec.title} "
                        f"({first_field.label or self._generate_label(first_field.field, time_unit)})",  # noqa: E501
                        x_field=plot_spec.x_field,
                        y_field=first_field.field,
                        x_label=plot_spec.x_label,
                        y_fields=None,
                        position=plot_spec.position,
                        y_scale=plot_spec.y_scale,
                        y_label=plot_spec.y_label,
                        plot_type=plot_spec.plot_type,
                    )

                    self._plot_single_line_metric(
                        plot_spec=single_line_spec,
                        ax=axs[plot_spec.position[0], plot_spec.position[1]],
                        concurrency_data=concurrency_data,
                        concurrency_levels=concurrency_levels,
                        label=labels[i],
                        time_unit=time_unit,
                    )
                else:
                    # Regular single-line plot
                    self._plot_single_metric(
                        plot_spec=plot_spec,
                        ax=axs[plot_spec.position[0], plot_spec.position[1]],
                        concurrency_data=concurrency_data,
                        concurrency_levels=concurrency_levels,
                        label=labels[i],
                        time_unit=time_unit,
                    )

    def _plot_single_metric(
        self,
        plot_spec: PlotSpec,
        ax: Axes,
        concurrency_data: Dict[int, MetricsData],
        concurrency_levels: List[int],
        label: str,
        time_unit: str = "s",
    ) -> None:
        """Plot a single metric based on plot specification."""
        try:
            if plot_spec.is_multi_line():
                self._plot_multi_line_metric(
                    plot_spec, ax, concurrency_data, concurrency_levels, label, time_unit
                )
            else:
                self._plot_single_line_metric(
                    plot_spec, ax, concurrency_data, concurrency_levels, label, time_unit
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

    def _plot_single_line_metric(
        self,
        plot_spec: PlotSpec,
        ax: Axes,
        concurrency_data: Dict[int, MetricsData],
        concurrency_levels: List[int],
        label: str,
        time_unit: str = "s",
    ) -> None:
        """Plot a single line metric (original behavior)."""
        # Extract data using field paths
        x_data = []
        y_data = []
        valid_concurrency = []

        y_field_spec = plot_spec.get_y_field_specs()[0]  # Single field

        for c in concurrency_levels:
            try:
                scenario_data = concurrency_data[c]
                if "aggregated_metrics" not in scenario_data:
                    continue
                metrics = scenario_data["aggregated_metrics"]  # type: ignore[index]

                x_val = PlotConfigManager.get_field_value(metrics, plot_spec.x_field)  # type: ignore[arg-type]
                y_val = PlotConfigManager.get_field_value(metrics, y_field_spec.field)  # type: ignore[arg-type]

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

        # Convert time data values if this is a latency field
        is_latency = TimeUnitConverter.is_latency_field(y_field_spec.field)
        
        if is_latency:
            # Check if data appears to already be in milliseconds (values > 1000)
            # If so, convert from ms to target unit, otherwise assume seconds
            if any(val > 1000 for val in y_data):
                y_data = [
                    TimeUnitConverter.convert_value(val, "ms", time_unit) for val in y_data
                ]
            else:
                y_data = [
                    TimeUnitConverter.convert_value(val, "s", time_unit) for val in y_data
                ]

        # Handle special error rate plot
        if y_field_spec.field == "error_rate" and plot_spec.plot_type == "bar":
            plot_error_rates(ax, concurrency_data, concurrency_levels, label)
        else:
            # Apply time unit conversion to labels and title if this is a latency field
            y_label = plot_spec.y_label or self._generate_label(y_field_spec.field, time_unit)
            title = plot_spec.title
            
            # Convert labels to correct time unit if this is a latency field
            if TimeUnitConverter.is_latency_field(y_field_spec.field):
                y_label = TimeUnitConverter.get_unit_label(y_label, time_unit)
                title = TimeUnitConverter.get_unit_label(title, time_unit)
            
            # Use existing plot_graph function
            plot_graph(
                ax=ax,
                x_data=x_data,
                y_data=y_data,
                x_label=plot_spec.x_label or self._generate_label(plot_spec.x_field, time_unit),
                y_label=y_label,
                title=title,
                concurrency_levels=valid_concurrency,
                label=label,
                plot_type=plot_spec.plot_type,
                time_unit=time_unit,
            )

    def _add_plot_annotations(
        self,
        ax: Axes,
        plot_spec: PlotSpec,
        valid_x: List[float],
        y_data: List[float],
        valid_concurrency: List[int],
        time_unit: str = "s",
    ) -> None:
        """Add annotations to plot points."""
        for x_val, y_val, c_val in zip(
            valid_x, y_data, valid_concurrency, strict=False
        ):
            # Show y-value when x-axis is concurrency, otherwise show concurrency
            if plot_spec.x_field == "num_concurrency":
                # Format y-value with appropriate precision based on time unit
                if time_unit == "ms":
                    annotation_text = f"{y_val:.1f}"
                else:
                    annotation_text = f"{y_val:.2f}"
            else:
                annotation_text = f"{c_val}"

            ax.annotate(
                annotation_text,
                (x_val, y_val),
                fontsize=8,
                alpha=1.0,
                xytext=(4, 4),
                textcoords="offset points",
                ha="left",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    alpha=0.1,
                    edgecolor="none",
                ),
            )

    def _plot_multi_line_metric(
        self,
        plot_spec: PlotSpec,
        ax: Axes,
        concurrency_data: Dict[int, MetricsData],
        concurrency_levels: List[int],
        label: str,
        time_unit: str = "s",
    ) -> None:
        """Plot multiple lines on the same subplot."""
        # Get colors from tab10 colormap
        cmap = plt.cm.get_cmap("tab10")
        colors = [cmap(i) for i in range(10)]  # Get first 10 colors
        linestyles = ["-", "--", "-.", ":"]

        x_data: List[Any] = []
        for c in concurrency_levels:
            try:
                scenario_data = concurrency_data[c]
                if "aggregated_metrics" not in scenario_data:
                    x_data.append(None)
                    continue
                metrics = scenario_data["aggregated_metrics"]  # type: ignore[index]
                x_val = PlotConfigManager.get_field_value(metrics, plot_spec.x_field)  # type: ignore[arg-type]
                if x_val is not None:
                    x_data.append(x_val)
                else:
                    x_data.append(None)
            except (AttributeError, KeyError):
                x_data.append(None)

        # Plot each Y-field as a separate line
        for i, y_field_spec in enumerate(plot_spec.get_y_field_specs()):
            y_data = []
            valid_x = []
            valid_concurrency = []

            for j, c in enumerate(concurrency_levels):
                if x_data[j] is None:
                    continue

                try:
                    scenario_data = concurrency_data[c]
                    if "aggregated_metrics" not in scenario_data:
                        continue
                    metrics = scenario_data["aggregated_metrics"]  # type: ignore[index]
                    y_val = PlotConfigManager.get_field_value(
                        metrics, y_field_spec.field
                    )  # type: ignore[arg-type]

                    if y_val is not None:
                        # Convert time values if this is a latency field
                        if TimeUnitConverter.is_latency_field(y_field_spec.field):
                            # Check if data appears to already be in milliseconds (values > 1000)
                            # If so, convert from ms to target unit, otherwise assume seconds
                            if y_val > 1000:
                                y_val = TimeUnitConverter.convert_value(y_val, "ms", time_unit)
                            else:
                                y_val = TimeUnitConverter.convert_value(y_val, "s", time_unit)
                        
                        y_data.append(y_val)
                        # Use evenly spaced positions for concurrency, actual values
                        # otherwise
                        if plot_spec.x_field == "num_concurrency":
                            valid_x.append(j)  # Even spacing: 0, 1, 2, 3...
                        else:
                            valid_x.append(x_data[j])  # Actual values
                        valid_concurrency.append(c)

                except (AttributeError, KeyError) as e:
                    logger.warning(
                        f"Missing data for field {y_field_spec.field}, "
                        f"concurrency {c}: {e}"
                    )
                    continue

            if not y_data:
                logger.warning(f"No valid data for field: {y_field_spec.field}")
                continue

            # Determine line styling
            color = y_field_spec.color or colors[i % len(colors)]
            linestyle = y_field_spec.linestyle or linestyles[i % len(linestyles)]
            line_label = y_field_spec.label or self._generate_label(y_field_spec.field, time_unit)

            # For multi-line plots, keep labels clean
            full_label = line_label

            # Plot the line
            if plot_spec.plot_type == "line":
                ax.plot(
                    valid_x,
                    y_data,
                    linestyle=linestyle,
                    color=color,
                    marker="o",
                    label=full_label,
                    markersize=4,
                )
                self._add_plot_annotations(
                    ax,
                    plot_spec,
                    valid_x,
                    y_data,
                    valid_concurrency,
                    time_unit,
                )
            elif plot_spec.plot_type == "scatter":
                ax.scatter(valid_x, y_data, color=color, label=full_label)
                self._add_plot_annotations(
                    ax,
                    plot_spec,
                    valid_x,
                    y_data,
                    valid_concurrency,
                    time_unit,
                )
            elif plot_spec.plot_type == "bar":
                # For bar plots with multiple fields, use grouped bars
                bar_width = 0.8 / len(plot_spec.get_y_field_specs())
                x_offset = (
                    i - len(plot_spec.get_y_field_specs()) / 2 + 0.5
                ) * bar_width
                ax.bar(
                    [x + x_offset for x in valid_x],
                    y_data,
                    width=bar_width,
                    color=color,
                    label=full_label,
                )

        # Set labels and title
        ax.set_xlabel(plot_spec.x_label or self._generate_label(plot_spec.x_field, time_unit))
        
        # Update y-axis label and title with correct time unit if this is a latency plot
        y_label = plot_spec.y_label or "Value"
        title = plot_spec.title
        
        # Check if any of the y-fields are latency fields to determine if we need time unit conversion
        has_latency_fields = any(
            TimeUnitConverter.is_latency_field(spec.field) 
            for spec in plot_spec.get_y_field_specs()
        )
        
        if has_latency_fields:
            y_label = TimeUnitConverter.get_unit_label(y_label, time_unit)
            title = TimeUnitConverter.get_unit_label(title, time_unit)
        
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        # Position legend outside plot area for multi-line plots to avoid overlap
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")

        # Apply y-axis scale from config or default based on field name
        if plot_spec.y_scale:
            # Use explicit scale from config
            if plot_spec.y_scale == "log":
                ax.set_yscale("log", base=10)
            # 'linear' is matplotlib default, no need to set explicitly
        else:
            # Backward compatibility: apply log scale for TTFT fields if not specified
            y_field_specs = plot_spec.get_y_field_specs()
            if any("ttft" in spec.field.lower() for spec in y_field_specs):
                ax.set_yscale("log", base=10)

        # Handle concurrency x-axis formatting - evenly spaced ticks
        if plot_spec.x_field == "num_concurrency":
            ax.set_xticks(range(len(concurrency_levels)))
            ax.set_xticklabels([str(level) for level in concurrency_levels])

    def _save_individual_subplots_multiline(
        self, axs: Any, experiment_folder: str, output_file_prefix: str
    ) -> None:
        """
        Custom subplot saving that properly handles multi-line plots.
        """

        for ax in axs.flat:
            title = ax.get_title()
            if not title:  # Skip empty subplots
                continue

            sanitized_title = sanitize_string(title)

            fig_temp, ax_temp = plt.subplots(figsize=(10, 6))

            # Copy all line plots with proper styling
            for line in ax.get_lines():
                ax_temp.plot(
                    line.get_xdata(),
                    line.get_ydata(),
                    label=line.get_label(),
                    marker=line.get_marker(),
                    linestyle=line.get_linestyle(),
                    color=line.get_color(),
                    markersize=line.get_markersize(),
                    alpha=line.get_alpha() or 1.0,
                )

            # Copy annotations - handle both multi-line and single-line plots
            # For all plots (multi-line and single-line), copy existing annotations
            for annotation in ax.texts:
                # Get the actual xy coordinates the annotation is pointing to
                xy_coords = (
                    annotation.xy
                    if hasattr(annotation, "xy")
                    else annotation.get_position()
                )

                # Get the original annotation properties
                original_xytext = (
                    annotation.xytext if hasattr(annotation, "xytext") else (4, 4)
                )
                original_textcoords = (
                    annotation.textcoords
                    if hasattr(annotation, "textcoords")
                    else "offset points"
                )

                ax_temp.annotate(
                    annotation.get_text(),
                    xy=xy_coords,  # Use the actual data coordinates
                    fontsize=annotation.get_fontsize(),
                    ha=annotation.get_ha(),
                    va=annotation.get_va(),
                    alpha=annotation.get_alpha() or 1.0,
                    xytext=original_xytext,
                    textcoords=original_textcoords,
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor="white",
                        alpha=0.1,
                        edgecolor="none",
                    ),
                )

            # Copy axis properties
            ax_temp.set_title(title)
            ax_temp.set_xlabel(ax.get_xlabel())
            ax_temp.set_ylabel(ax.get_ylabel())

            # Copy ticks and labels
            ax_temp.set_xticks(ax.get_xticks())
            ax_temp.set_xticklabels(ax.get_xticklabels())
            ax_temp.set_yticks(ax.get_yticks())
            ax_temp.set_yticklabels(ax.get_yticklabels())

            # Copy axis limits
            ax_temp.set_xlim(ax.get_xlim())
            ax_temp.set_ylim(ax.get_ylim())

            # Copy scale settings
            if ax.get_yscale() == "log":
                ax_temp.set_yscale("log", base=10)
            if ax.get_xscale() == "log":
                ax_temp.set_xscale("log", base=10)

            # Copy grid
            ax_temp.grid(
                ax.get_xgridlines()[0].get_visible() if ax.get_xgridlines() else True,
                alpha=0.3,
            )
            ax_temp.minorticks_on()

            # Handle legend properly for multi-line plots
            handles, labels = ax.get_legend_handles_labels()
            if handles and labels:
                # Clean up labels and position legend properly
                cleaned_labels = []
                for label in labels:
                    # Remove any scenario prefixes for cleaner individual plots
                    if " - " in label:
                        label = label.split(" - ")[-1]
                    cleaned_labels.append(label)

                ax_temp.legend(
                    handles,
                    cleaned_labels,
                    fontsize="small",
                    bbox_to_anchor=(1.05, 1),
                    loc="upper left",
                )

            plt.tight_layout()
            # Adjust for external legend
            if handles and labels:
                plt.subplots_adjust(right=0.85)

            output_file = os.path.join(
                experiment_folder, f"{output_file_prefix}_{sanitized_title}.png"
            )
            fig_temp.savefig(output_file, bbox_inches="tight")
            plt.close(fig_temp)
            logger.info(f"🎨 Saving {output_file}")

    def _generate_label(self, field_path: str, time_unit: str = "s") -> str:
        """Generate a human-readable label from field path."""
        # Simple label generation - can be enhanced
        parts = field_path.split(".")
        if len(parts) == 1:
            base_label = field_path.replace("_", " ").title()
            # Apply time unit conversion if this is a latency field
            if TimeUnitConverter.is_latency_field(field_path):
                base_label = TimeUnitConverter.get_unit_label(base_label, time_unit)
            return base_label

        # Handle stats fields
        if parts[0] == "stats":
            metric = parts[1].replace("_", " ").title()
            if len(parts) > 2:
                stat = parts[2].upper()
                base_label = f"{metric} ({stat})"
            else:
                base_label = metric
            
            # Apply time unit conversion if this is a latency field
            if TimeUnitConverter.is_latency_field(parts[1]):
                base_label = TimeUnitConverter.get_unit_label(base_label, time_unit)
            return base_label

        base_label = field_path.replace("_", " ").title()
        # Apply time unit conversion if this is a latency field
        if TimeUnitConverter.is_latency_field(field_path):
            base_label = TimeUnitConverter.get_unit_label(base_label, time_unit)
        return base_label

    def _finalize_and_save_plots(
        self,
        axs: Any,
        fig: Figure,
        labels: List[str],
        experiment_folder: str,
        output_file_prefix: str,
    ) -> None:
        """Finalize and save plots."""

        # Hide unused subplots
        layout = self.config.layout
        used_positions = {plot.position for plot in self.config.plots}

        for row in range(layout.rows):
            for col in range(layout.cols):
                if (row, col) not in used_positions:
                    axs[row, col].set_visible(False)

        # Save individual subplots (custom version for multi-line plots)
        self._save_individual_subplots_multiline(
            axs, experiment_folder, output_file_prefix
        )

        # For multi-line plots, legends are already positioned on individual subplots
        # Only add global legend for single-line plots
        has_multi_line = any(plot.is_multi_line() for plot in self.config.plots)

        if not has_multi_line:
            fig.legend(labels=labels, loc="lower center", ncol=3, fancybox=True)
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.1)
        else:
            plt.tight_layout()
            # Adjust layout to accommodate external legends
            plt.subplots_adjust(right=0.85)

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
    plot_config: Optional[PlotConfig] = None,
    time_unit: Optional[str] = None,
) -> None:
    """
    Plot experiment data using flexible configuration.

    Args:
        run_data_list: List of experiment data tuples
        group_key: Key to group data by
        experiment_folder: Output folder path
        plot_config: Plot configuration (uses default if None)
        time_unit: Time unit to use for plotting ('s' or 'ms'). If None, extracts from metadata.
    """
    # Use CLI time_unit parameter if provided, otherwise extract from experiment metadata
    if time_unit is None:
        extracted_time_unit = run_data_list[0][0].time_unit if run_data_list else "s"
        time_unit = extracted_time_unit
    
    if plot_config is None:
        plot_config = PlotConfigManager.load_preset("2x4_default", time_unit)

    generator = FlexiblePlotGenerator(plot_config, time_unit)
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
        scenario_data = experiment_metrics[first_scenario][first_concurrency]
        if "aggregated_metrics" not in scenario_data:
            raise KeyError("aggregated_metrics not found in sample data")
        sample_agg_metrics = scenario_data["aggregated_metrics"]  # type: ignore[index]
    except (IndexError, KeyError) as e:
        errors.append(f"Cannot extract sample metrics for validation: {e}")
        return errors

    # Validate each plot specification
    for i, plot_spec in enumerate(config.plots):
        # Validate X field path
        if not PlotConfigManager.validate_field_path(
            plot_spec.x_field,
            sample_agg_metrics,  # type: ignore[arg-type]
        ):
            errors.append(f"Plot {i+1}: Invalid x_field '{plot_spec.x_field}'")

        # Validate Y field paths (single or multiple)
        try:
            y_field_specs = plot_spec.get_y_field_specs()
            for j, y_field_spec in enumerate(y_field_specs):
                if not PlotConfigManager.validate_field_path(
                    y_field_spec.field,
                    sample_agg_metrics,  # type: ignore[arg-type]
                ):
                    if len(y_field_specs) == 1:
                        errors.append(
                            f"Plot {i+1}: Invalid y_field '{y_field_spec.field}'"
                        )
                    else:
                        errors.append(
                            f"Plot {i+1}: Invalid y_fields[{j}] '{y_field_spec.field}'"
                        )
        except Exception as e:
            errors.append(f"Plot {i+1}: Error validating Y-fields: {e}")

        # Validate position bounds
        layout = config.layout
        row, col = plot_spec.position
        if row >= layout.rows or col >= layout.cols:
            errors.append(
                f"Plot {i+1}: Position ({row}, {col}) exceeds layout bounds "
                f"({layout.rows-1}, {layout.cols-1})"
            )

    return errors
