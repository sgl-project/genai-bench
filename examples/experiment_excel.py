"""Example to generate excel report from an experiment folder."""

import os

from genai_bench.analysis.excel_report import create_workbook
from genai_bench.analysis.experiment_loader import (
    load_one_experiment,
)
from genai_bench.logging import LoggingManager

LoggingManager("excel")


folder_name = "<Path to your experiment folder>"  # noqa: E501
os.makedirs(folder_name, exist_ok=True)
experiment_metadata, run_data = load_one_experiment(folder_name)
create_workbook(
    experiment_metadata,
    run_data,
    "experiment_summary_sheet_5.xlsx",
)
