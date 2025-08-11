"""Example to run plots filtering and generate plot from one experiment."""

import os

from genai_bench.analysis.experiment_loader import (
    load_multiple_experiments,
    load_one_experiment,
)
from genai_bench.analysis.plot_report import plot_experiment_data
from genai_bench.logging import LoggingManager

LoggingManager("plot")


# Example usage with filtering multiple experiments
folder_name = "/Users/changsu/experiment_plot"
filter_criteria = {
    "model": "vllm-model",
}

os.makedirs(folder_name, exist_ok=True)

run_data_list = load_multiple_experiments(folder_name, filter_criteria)

if not run_data_list:
    print("Empty data after filtering")
else:
    # Plot the data grouped by 'server_version'
    plot_experiment_data(
        run_data_list, group_key="server_version", experiment_folder=folder_name
    )

# Plot for one experiment
experiment_folder = os.path.join(
    folder_name,
    "openai_chat_vllm-model_tokenizer__mnt_data_models_Llama-3-70B-Instruct_20240904_003850",
)
experiment_metadata, run_data = load_one_experiment(experiment_folder)
if not experiment_metadata or not run_data:
    print("Didn't find any experiment data")
else:
    plot_experiment_data(
        [
            [experiment_metadata, run_data],
        ],
        group_key="traffic_scenario",
        experiment_folder=experiment_folder,
    )
