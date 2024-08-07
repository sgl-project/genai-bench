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
@click.pass_context
def excel(ctx, experiment_folder, excel_name, metric_percentile):
    """
    Exports the experiment results to an Excel file.
    """
    LoggingManager("excel")
    _ = init_logger("genai_bench.excel")
    excel_path = os.path.join(experiment_folder, excel_name + ".xlsx")
    experiment_metadata, run_data = load_one_experiment(experiment_folder)
    create_workbook(experiment_metadata, run_data, excel_path, metric_percentile)


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
    help="A dictionary containing filter criteria for the plot. Default: {}. "
    "Example: '{'model': 'vllm-model'}'",
)
@click.option(
    "--filter-criteria",
    type=str,
    default=None,
    callback=validate_filter_criteria,
    help="A dictionary containing filter criteria for the plot. Default: {}. "
    "Example: '{'model': 'vllm-model'}'",
)
@click.pass_context
def plot(ctx, experiments_folder, filter_criteria, group_key):
    """
    Plots the experiment(s) results based on filters and group.
    """
    LoggingManager("plot")
    logger = init_logger("genai_bench.plot")
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
        plot_experiment_data(
            [(experiment_metadata, run_data)],  # Single experiment is wrapped in a list
            group_key=group_key,
            experiment_folder=experiments_folder,
        )
    else:
        run_data_list = load_multiple_experiments(experiments_folder, filter_criteria)
        if run_data_list:
            plot_experiment_data(
                run_data_list,
                group_key=group_key,
                experiment_folder=experiments_folder,
            )
        else:
            logger.info("No valid experiment data found for multiple experiments.")
