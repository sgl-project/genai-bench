"""CLI for reports related commands."""

import os

import click

from genai_bench.analysis.excel_report import create_workbook
from genai_bench.analysis.experiment_loader import (
    load_multiple_experiments,
    load_one_experiment,
)
from genai_bench.analysis.plot_report import plot_experiment_data
from genai_bench.cli.validation import validate_filter_criteria
from genai_bench.logging import LoggingManager, init_logger
from genai_bench.utils import is_single_experiment_folder


@click.command()
@click.option(
    "--metric-percentile",
    type=click.Choice(["mean", "p25", "p50", "p75", "p90", "p95", "p99"]),
    default="mean",
    help="The percentile of the metric data to select from.",
)
@click.option(
    "--experiment-folder",
    type=click.Path(exists=True),
    required=True,
    prompt=True,
    help="Path to folder of one experiment",
)
@click.option(
    "--excel-name",
    type=str,
    required=True,
    prompt=True,
    help="Name of the Excel file. The system will create a <excel-name>.xlsx.",
)
@click.option(
    "--time-unit",
    type=click.Choice(["s", "ms"], case_sensitive=False),
    default="s",
    help="Time unit for latency metrics in the spreadsheet. "
    "Options: 's' (seconds), 'ms' (milliseconds). Default: s",
)
@click.pass_context
def excel(ctx, experiment_folder, excel_name, metric_percentile, time_unit):
    """
    Exports the experiment results to an Excel file.
    """
    LoggingManager("excel")
    _ = init_logger("genai_bench.excel")
    excel_path = os.path.join(experiment_folder, excel_name + ".xlsx")
    experiment_metadata, run_data = load_one_experiment(experiment_folder)
    create_workbook(experiment_metadata, run_data, excel_path, metric_percentile, time_unit)


@click.command()
@click.option(
    "--experiments-folder",
    type=click.Path(exists=True),
    required=True,
    prompt=True,
    help="Path to folder with experiments. It could be a folder containing "
    "multiple experiments, or just one experiment.",
)
@click.option(
    "--group-key",
    type=str,
    required=True,
    prompt=True,
    help="Key to group the data by (e.g., 'traffic_scenario', 'server_version'). "
    "Use 'none' for single scenario analysis.",
)
@click.option(
    "--filter-criteria",
    type=str,
    default=None,
    callback=validate_filter_criteria,
    help="A dictionary containing filter criteria for the plot. Default: {}. "
    "Example: '{'model': 'meta-llama/Meta-Llama-3-70B-Instruct'}'",
)
@click.option(
    "--plot-config",
    type=click.Path(),
    default=None,
    help="Path to JSON plot configuration file. If not provided, uses default 2x4 "
    "layout.",
)
@click.option(
    "--preset",
    type=click.Choice(
        ["2x4_default", "simple_2x2", "multi_line_latency", "single_scenario_analysis"]
    ),
    default=None,
    help="Use a built-in plot preset. Overrides --plot-config if both are provided.",
)
@click.option(
    "--time-unit",
    type=click.Choice(["s", "ms"], case_sensitive=False),
    default="s",
    help="Time unit for latency metrics display and export. "
    "Options: 's' (seconds), 'ms' (milliseconds). Default: s",
)
@click.option(
    "--list-fields",
    is_flag=True,
    help="List all available fields with actual data from the experiment folder and "
    "exit.",
)
@click.option(
    "--validate-only",
    is_flag=True,
    help="Only validate the plot configuration without generating plots.",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable verbose logging for debugging.",
)
@click.pass_context
def plot(
    ctx,
    experiments_folder,
    group_key,
    filter_criteria,
    plot_config,
    preset,
    time_unit,
    list_fields,
    validate_only,
    verbose,
):
    """
    Plots the experiment(s) results based on filters and group.
    Supports flexible plot configurations via JSON files or built-in presets.
    """
    LoggingManager("plot")
    logger = init_logger("genai_bench.plot")

    # Handle --list-fields option
    if list_fields:
        from genai_bench.analysis.plot_config import PlotConfigManager

        # Load experiment data to get real field availability
        logger.info(f"Scanning experiment data in {experiments_folder}...")

        if is_single_experiment_folder(experiments_folder):
            experiment_metadata, run_data = load_one_experiment(
                experiments_folder, filter_criteria
            )
            if not experiment_metadata or not run_data:
                click.echo(f"❌ No experiment data found in {experiments_folder}")
                return
            run_data_list = [(experiment_metadata, run_data)]
        else:
            run_data_list = load_multiple_experiments(
                experiments_folder, filter_criteria
            )
            if not run_data_list:
                click.echo(
                    "❌ No valid experiment data found for multiple experiments."
                )
                return

        # Extract available fields from actual data
        try:
            _, experiment_metrics = run_data_list[0]
            first_scenario = list(experiment_metrics.keys())[0]
            first_concurrency = list(experiment_metrics[first_scenario].keys())[0]
            sample_metrics = experiment_metrics[first_scenario][first_concurrency][
                "aggregated_metrics"
            ]

            available_fields = PlotConfigManager.get_fields_from_data(sample_metrics)

            click.echo(
                f"Available fields from experiment data in {experiments_folder}:"
            )
            click.echo("=" * 70)
            click.echo(
                f"Found {len(available_fields)} available fields with actual data:"
            )
            click.echo()

            # Group fields by category for better readability
            direct_fields = []
            stats_fields = []

            for field_path, (value, field_type) in sorted(available_fields.items()):
                if field_path.startswith("stats."):
                    stats_fields.append((field_path, value, field_type))
                else:
                    direct_fields.append((field_path, value, field_type))

            if direct_fields:
                click.echo("📊 Direct Metrics:")
                for field_path, value, field_type in direct_fields:
                    click.echo(f"  {field_path:<35} = {value} ({field_type})")
                click.echo()

            if stats_fields:
                click.echo("📈 Statistical Metrics:")
                for field_path, value, field_type in stats_fields:
                    click.echo(f"  {field_path:<35} = {value} ({field_type})")

        except Exception as e:
            click.echo(f"❌ Error extracting fields from experiment data: {e}")
            # Fallback to static field list
            click.echo("\nFalling back to static field definitions...")
            fields = PlotConfigManager.get_available_fields()
            click.echo("Available fields from AggregatedMetrics schema:")
            click.echo("=" * 50)
            for field_path, description in sorted(fields.items()):
                click.echo(f"{field_path:<40} - {description}")

        return

    # Load plot configuration
    try:
        from genai_bench.analysis.flexible_plot_report import (
            plot_experiment_data_flexible,
            validate_plot_config_with_data,
        )
        from genai_bench.analysis.plot_config import PlotConfigManager

        if preset:
            config = PlotConfigManager.load_preset(preset)
            logger.info(f"Using preset configuration: {preset}")
        elif plot_config:
            config = PlotConfigManager.load_from_file(plot_config)
            logger.info(f"Using configuration from: {plot_config}")
        else:
            config = PlotConfigManager.load_preset("2x4_default")
            logger.info("Using default 2x4 configuration")

    except Exception as e:
        logger.error(f"Error loading plot configuration: {e}")
        return

    # Load experiment data
    logger.info(
        f"Plotting experiments in {experiments_folder} with filter: {filter_criteria}"
    )

    if is_single_experiment_folder(experiments_folder):
        experiment_metadata, run_data = load_one_experiment(
            experiments_folder, filter_criteria
        )
        if not experiment_metadata or not run_data:
            logger.info(
                f"No experiment_metadata or run_data found in {experiments_folder}"
            )
            return
        run_data_list = [(experiment_metadata, run_data)]
    else:
        run_data_list = load_multiple_experiments(experiments_folder, filter_criteria)
        if not run_data_list:
            logger.info("No valid experiment data found for multiple experiments.")
            return

    # Validate configuration with actual data
    validation_errors = validate_plot_config_with_data(config, run_data_list)
    if validation_errors:
        logger.error("Plot configuration validation failed:")
        for error in validation_errors:
            logger.error(f"  - {error}")
        if validate_only:
            return
        logger.warning("Proceeding with plot generation despite validation errors...")
    else:
        logger.info("Plot configuration validation passed")
        if validate_only:
            logger.info(
                "Configuration is valid. Use without --validate-only to generate plots."
            )
            return

    # Generate plots using flexible system
    try:
        plot_experiment_data_flexible(
            run_data_list=run_data_list,
            group_key=group_key,
            experiment_folder=experiments_folder,
            plot_config=config,
            time_unit=time_unit,
        )
        logger.info("Plot generation completed successfully")
    except Exception as e:
        logger.error(f"Error generating plots: {e}")
        # Fallback to originally plotting system
        logger.info("Falling back to original plotting system...")
        plot_experiment_data(
            run_data_list,
            group_key=group_key,
            experiment_folder=experiments_folder,
        )
