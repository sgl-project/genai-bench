import json
from unittest.mock import patch

import pytest
from click.testing import CliRunner
from rich.console import Console

from genai_bench.analysis.plot_config import PlotConfigManager
from genai_bench.cli.report import excel
from genai_bench.metrics.aggregated_metrics_collector import AggregatedMetricsCollector
from genai_bench.metrics.metrics import (
    AggregatedMetrics,
    RequestLevelMetrics,
    StatField,
)
from genai_bench.ui.layout import create_metric_panel


def add_rates(collector, rates):
    for rate in rates:
        collector.add_single_request_metrics(
            RequestLevelMetrics(
                ttft=0.2,
                tpot=0.1,
                e2e_latency=1.2,
                output_latency=1.0,
                num_input_tokens=10,
                num_output_tokens=10,
                num_reasoning_tokens=0,
                total_tokens=20,
                input_throughput=rate,
                output_throughput=rate,
                output_inference_speed=rate,
                audio_throughput=rate,
            )
        )


@pytest.mark.parametrize(
    "rates, expected",
    [([10, 20, 30, 40, 50], (10.4, 12, 14)), ([42], (42, 42, 42)), ([0, 0], (0, 0, 0))],
)
def test_lower_tail_percentiles_live_and_saved(tmp_path, rates, expected):
    collector = AggregatedMetricsCollector()
    add_rates(collector, rates)
    collector.add_single_request_metrics(
        RequestLevelMetrics(error_code=500, output_throughput=1000000)
    )
    live = collector.get_live_metrics()["stats"]
    for field in ("input_throughput", "output_throughput"):
        assert [live[field][p] for p in ("p1", "p5", "p10")] == pytest.approx(expected)
    collector.aggregate_metrics_data(0, 10, 0, 0)
    for field in (
        "input_throughput",
        "output_throughput",
        "output_inference_speed",
        "audio_throughput",
    ):
        stats = getattr(collector.aggregated_metrics.stats, field)
        assert [stats[p] for p in ("p1", "p5", "p10")] == pytest.approx(expected)
    destination = tmp_path / "metrics.json"
    collector.save(str(destination), metrics_time_unit="ms")
    saved = json.loads(destination.read_text())["aggregated_metrics"]
    assert saved["stats"]["ttft"]["p1"] == pytest.approx(200)
    restored = AggregatedMetrics.model_validate(saved)
    assert restored.stats.output_throughput.p5 == pytest.approx(expected[1])


def test_lower_tail_percentiles_respect_warmup_and_cooldown():
    collector = AggregatedMetricsCollector()
    add_rates(collector, [0, 10, 20, 30, 100])
    collector.aggregate_metrics_data(0, 10, 0.2, 0.2)
    stats = collector.aggregated_metrics.stats.output_throughput
    assert [stats[p] for p in ("p1", "p5", "p10")] == pytest.approx((10.2, 11, 12))
    assert stats.p99 == pytest.approx(29.8)


def test_lower_tail_percentiles_displayed_in_live_panel():
    collector = AggregatedMetricsCollector()
    add_rates(collector, [10, 20, 30, 40, 50])
    stats = collector.get_live_metrics()["stats"]
    _, panel = create_metric_panel("Output", stats["ttft"], stats["output_throughput"])
    console = Console(record=True, width=80)
    console.print(panel)
    text = console.export_text()
    for label, value in [("P1:", "10.40"), ("P5:", "12.00"), ("P10:", "14.00")]:
        assert label in text and value in text


@pytest.mark.parametrize("percentile", ["p1", "p5", "p10"])
def test_excel_accepts_lower_tail_percentiles(tmp_path, percentile):
    with (
        patch("genai_bench.cli.report.load_one_experiment", return_value=(None, {})),
        patch("genai_bench.cli.report.create_workbook") as create,
    ):
        result = CliRunner().invoke(
            excel,
            [
                "--experiment-folder",
                str(tmp_path),
                "--excel-name",
                "tail",
                "--metric-percentile",
                percentile,
            ],
        )
    assert result.exit_code == 0, result.output
    assert create.call_args.args[3] == percentile
    assert (
        f"stats.output_throughput.{percentile}"
        in PlotConfigManager.get_available_fields()
    )


def test_older_statistics_without_lower_percentiles_still_load():
    stats = StatField.model_validate({"mean": 10, "p99": 20})
    assert stats.p99 == 20
    assert stats.p1 is None and stats.p5 is None and stats.p10 is None
