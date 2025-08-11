"""Example to generate excel report from an experiment folder."""

import os

from genai_bench.analysis.excel_report import create_workbook
from genai_bench.analysis.experiment_loader import (
    load_one_experiment,
)
from genai_bench.logging import LoggingManager

LoggingManager("excel")


folder_name = "/Users/changsu/openai_chat_sglang-model_tokenizer__mnt_data_models_Llama-3-70B-Instruct_20240904_003850"  # noqa: E501
os.makedirs(folder_name, exist_ok=True)
experiment_metadata, run_data = load_one_experiment(folder_name)
create_workbook(
    experiment_metadata,
    run_data,
    "experiment_summary_sheet_5.xlsx",
)
