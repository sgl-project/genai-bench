import os
import tempfile

from openpyxl import load_workbook

from genai_bench.analysis.excel_report import create_workbook
from genai_bench.metrics.metrics import AggregatedMetrics, MetricStats, StatField
from genai_bench.protocol import ExperimentMetadata


def _make_metadata(scenarios: list[str]) -> ExperimentMetadata:
    return ExperimentMetadata(
        cmd="genai-bench benchmark",
        benchmark_version="test",
        api_backend="openai",
        auth_config={},
        api_model_name="gpt-3.5",
        model="gpt-3.5",
        task="text-to-text",
        num_concurrency=[1],
        batch_size=None,
        iteration_type="num_concurrency",
        traffic_scenario=scenarios,
        additional_request_params={},
        server_engine="engine",
        server_version="v1",
        server_gpu_type="A100",
        server_gpu_count="1",
        max_time_per_run_s=60,
        max_requests_per_run=10,
        experiment_folder_name="/tmp",
        dataset_path=None,
        character_token_ratio=1.0,
    )


def _agg_metrics(
    scenario: str, num_concurrency: int, output_infer_speed_mean: float
) -> AggregatedMetrics:
    stats = MetricStats(output_inference_speed=StatField(mean=output_infer_speed_mean))
    return AggregatedMetrics(
        scenario=scenario,
        num_concurrency=num_concurrency,
        iteration_type="num_concurrency",
        stats=stats,
    )


def test_summary_displays_na_when_threshold_not_met():
    scenario = "D(100,100)"
    metadata = _make_metadata([scenario])

    # Build run_data where no iteration exceeds threshold (10 tokens/s)
    # Provide a dummy individual_request_metrics entry to avoid empty merge ranges
    run_data = {
        scenario: {
            1: {
                "aggregated_metrics": _agg_metrics(
                    scenario, 1, output_infer_speed_mean=5.0
                ),
                "individual_request_metrics": [{}],
            }
        }
    }

    with tempfile.TemporaryDirectory() as td:
        out_path = os.path.join(td, "summary.xlsx")
        create_workbook(metadata, run_data, out_path, percentile="mean")

        wb = load_workbook(out_path)
        ws = wb["Summary"]

        # Find the row for our scenario
        display_scenario = "Scenario 2: Chatbot/Dialog D(100,100)"
        found = False
        for row in ws.iter_rows(min_row=2, values_only=True):
            # Columns: [GPU Type, Use Case, Summary Value, Total Chars/Hour]
            if row[1] == display_scenario:
                assert row[2] == "N/A"
                assert row[3] == "N/A"
                found = True
                break

        assert found, "Scenario row not found in Summary sheet"
