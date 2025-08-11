"""Example to run plots filtering and generate plot from one experiment."""

import os

from genai_bench.analysis.experiment_loader import (
    load_multiple_experiments,
    load_one_experiment,
)
from genai_bench.analysis.flexible_plot_report import plot_experiment_data_flexible
from genai_bench.logging import LoggingManager

LoggingManager("plot")


# Example usage with filtering multiple experiments
folder_name = "<Path to the experiment folder>"
filter_criteria = {
    "model": "Llama-4-Scout-17B-16E-Instruct",
}

os.makedirs(folder_name, exist_ok=True)

run_data_list = load_multiple_experiments(folder_name, filter_criteria)

if not run_data_list:
    print("Empty data after filtering")
else:
    # Plot the data grouped by 'server_version'
    plot_experiment_data_flexible(
        run_data_list, group_key="server_version", experiment_folder=folder_name
    )

# Plot for one experiment
experiment_folder = os.path.join(
    folder_name,
    "openai_SGLang_v0.4.7.post1_text-to-text_Llama-4-Scout-17B-16E-Instruct_20250620_042005",
)
experiment_metadata, run_data = load_one_experiment(experiment_folder)
if not experiment_metadata or not run_data:
    print("Didn't find any experiment data")
else:
    plot_experiment_data_flexible(
        [
            [experiment_metadata, run_data],
        ],
        group_key="traffic_scenario",
        experiment_folder=experiment_folder,
    )
