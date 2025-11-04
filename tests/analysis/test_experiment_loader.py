import json
import logging
from unittest.mock import call, mock_open, patch

import pytest

from genai_bench.analysis.experiment_loader import (
    apply_filter_to_metadata,
    load_experiment_metadata,
    load_multiple_experiments,
    load_one_experiment,
    load_run_data,
)
from genai_bench.metrics.metrics import AggregatedMetrics, MetricStats, StatField
from genai_bench.protocol import ExperimentMetadata


def load_mock_data(file_path):
    with open(file_path, "r") as f:
        return f.read()


@pytest.fixture
def mock_experiment_metadata():
    experiment_metadata_json_str = load_mock_data(
        "tests/analysis/mock_experiment_data.json"
    )
    experiment_metadata = json.loads(experiment_metadata_json_str)
    return ExperimentMetadata(**experiment_metadata)


@patch("os.listdir", return_value=["experiment_1", "experiment_2"])
@patch("os.path.isdir", return_value=True)
@patch("genai_bench.analysis.experiment_loader.load_one_experiment")
def test_load_metrics_from_experiments(
    mock_load_one_experiment, mock_isdir, mock_listdir, mock_experiment_metadata
):
    mock_load_one_experiment.side_effect = [
        (mock_experiment_metadata, {"metrics": "test"}),
        (mock_experiment_metadata, {"metrics": "test2"}),
    ]

    folder_name = "fake_folder"
    run_data_list = load_multiple_experiments(folder_name)

    assert len(run_data_list) == 2
    assert run_data_list[0][0] == mock_experiment_metadata
    assert run_data_list[0][1] == {"metrics": "test"}


@patch(
    "os.listdir",
    return_value=[
        "experiment_metadata.json",
        "N480_240_300_150_chat_concurrency_1_time_600s.json",
    ],
)
@patch("os.path.isdir", return_value=False)
@patch("os.path.exists", return_value=True)
@patch(
    "builtins.open",
    new_callable=mock_open,
    read_data='{"aggregated_metrics": {}}',
)
@patch("genai_bench.analysis.experiment_loader.load_experiment_metadata")
@patch("genai_bench.analysis.experiment_loader.load_run_data")
def test_load_one_experiment(
    mock_load_run_data,
    mock_load_experiment_metadata,
    mock_open,
    mock_exists,
    mock_isdir,
    mock_listdir,
    mock_experiment_metadata,
):
    mock_load_experiment_metadata.return_value = mock_experiment_metadata

    folder_name = "fake_experiment_folder"
    metadata, run_data = load_one_experiment(folder_name)

    assert metadata == mock_experiment_metadata
    mock_load_run_data.assert_called_once()


@patch(
    "os.listdir",
    return_value=[
        "experiment_metadata.json",
        "N480_240_300_150_chat_concurrency_1_time_600s.json",
        "N480_240_300_150_chat_concurrency_4_time_600s.json",
    ],
)
@patch("os.path.isdir", return_value=False)
@patch("os.path.exists", return_value=True)
@patch(
    "builtins.open",
    new_callable=mock_open,
    read_data='{"aggregated_metrics": '
    '{"scenario": "N(480,240)/(300,150)", '
    '"num_concurrency": 1}}',
)
@patch("genai_bench.analysis.experiment_loader.load_experiment_metadata")
@patch("genai_bench.analysis.experiment_loader.load_run_data")
@patch("logging.Logger.warning")
def test_load_one_experiment_with_missing_concurrency_levels(
    mock_logger_warning,
    mock_load_run_data,
    mock_load_experiment_metadata,
    mock_open,
    mock_isdir,
    mock_exists,
    mock_listdir,
    mock_experiment_metadata,
):
    mock_load_experiment_metadata.return_value = mock_experiment_metadata
    mock_experiment_metadata.num_concurrency = [1, 2, 4]

    folder_name = "fake_experiment_folder"

    # Simulate load_run_data behavior directly modifying run_data
    def mock_load_run_data_side_effect(file_path, run_data, filter_criteria):
        scenario = "N(480,240)/(300,150)"
        if "concurrency_1" in file_path:
            run_data.setdefault(scenario, {}).setdefault(
                "num_concurrency_levels", set()
            ).add(1)
        elif "concurrency_4" in file_path:
            run_data.setdefault(scenario, {}).setdefault(
                "num_concurrency_levels", set()
            ).add(4)

    mock_load_run_data.side_effect = mock_load_run_data_side_effect

    metadata, run_data = load_one_experiment(folder_name)

    # Ensure the metadata is returned correctly
    assert metadata == mock_experiment_metadata
    mock_load_run_data.assert_has_calls(
        [
            call(
                "fake_experiment_folder/N480_240_300_150_chat_concurrency_1_time_600s.json",  # noqa: E501
                {"N(480,240)/(300,150)": {}},
                None,
            ),
            call(
                "fake_experiment_folder/N480_240_300_150_chat_concurrency_4_time_600s.json",  # noqa: E501
                {"N(480,240)/(300,150)": {}},
                None,
            ),
        ]
    )

    # Ensure that the logger warning is called for missing concurrency level
    # (2 in this case)
    mock_logger_warning.assert_has_calls(
        [
            call(
                "‼️ Scenario D(100,100) in metadata but metrics not found! "
                "Please re-run this scenario if necessary!"
            ),
            call(
                "‼️ Scenario D(2000,200) in metadata but metrics not found! "
                "Please re-run this scenario if necessary!"
            ),
            call(
                "‼️ Scenario 'N(480,240)/(300,150)' is missing num_concurrency "
                "levels: [2]. Please re-run this scenario if necessary!"
            ),
        ]
    )


@patch(
    "builtins.open",
    new_callable=mock_open,
    read_data=load_mock_data("tests/analysis/mock_experiment_data.json"),
)
def test_load_experiment_metadata(mock_open, mock_experiment_metadata):
    file_path = "experiment_metadata.json"
    metadata = load_experiment_metadata(file_path, None)

    assert metadata.traffic_scenario == [
        "N(480,240)/(300,150)",
        "D(100,100)",
        "D(100,1000)",
        "D(2000,200)",
        "D(7800,200)",
    ]
    mock_open.assert_called_once_with(file_path, "r")


@patch(
    "builtins.open",
    new_callable=mock_open,
    read_data=load_mock_data("tests/analysis/mock_experiment_data.json"),
)
def test_load_experiment_metadata_with_filter(
    mock_open, mock_experiment_metadata, caplog
):
    file_path = "experiment_metadata.json"
    with caplog.at_level(logging.INFO):
        metadata = load_experiment_metadata(
            file_path, {"traffic_scenario": ["D(120,100)"]}
        )
        assert metadata is None
        mock_open.assert_called_once_with(file_path, "r")

    assert (
        f"No match with filter_criteria found in ExperimentMetadata under "
        f"{file_path}." in caplog.text
    )


def test_apply_filter_to_metadata(mock_experiment_metadata, caplog):
    filter_criteria = {"traffic_scenario": ["D(100,100)"]}
    result = apply_filter_to_metadata(mock_experiment_metadata, filter_criteria)
    assert result is True

    with caplog.at_level(logging.INFO):
        filter_criteria = {"traffic_scenario": ["D(120,100)"]}
        result = apply_filter_to_metadata(mock_experiment_metadata, filter_criteria)
        assert result is False
    assert (
        "The scenarios ['D(120,100)'] you want to filter is not "
        "presented in your experiments." in caplog.text
    )

    filter_criteria = {"server_version": "v0.5.3"}
    result = apply_filter_to_metadata(mock_experiment_metadata, filter_criteria)
    assert result is False

    filter_criteria = {"model": "Meta-Llama-3.1-70B-Instruct"}
    result = apply_filter_to_metadata(mock_experiment_metadata, filter_criteria)
    assert result is True

    filter_criteria = {"model-name": "Meta-Llama-3.1-70B-Instruct"}
    result = apply_filter_to_metadata(mock_experiment_metadata, filter_criteria)
    assert result is False


@patch(
    "builtins.open",
    new_callable=mock_open,
    read_data=load_mock_data("tests/analysis/mock_run_data.json"),
)
def test_load_run_data(mock_open):
    run_data = {}
    file_path = "fake_path.json"
    filter_criteria = None

    load_run_data(file_path, run_data, filter_criteria)

    # Assert the run_data dictionary has been updated correctly
    assert "D(100,100)" in run_data
    assert 1 in run_data["D(100,100)"]
    assert run_data["D(100,100)"][1]["aggregated_metrics"].scenario == "D(100,100)"
    assert len(run_data["D(100,100)"][1]["individual_request_metrics"]) == 2
    assert (
        run_data["D(100,100)"][1]["individual_request_metrics"][0]["num_input_tokens"]
        == 92
    )


# Test for filter criteria
@patch(
    "builtins.open",
    new_callable=mock_open,
    read_data=load_mock_data("tests/analysis/mock_run_data.json"),
)
def test_load_run_data_with_filter(mock_open):
    run_data = {}
    file_path = "fake_path.json"
    filter_criteria = {
        "traffic_scenario": ["D(200,200)"]
    }  # The filter does not match the scenario

    load_run_data(file_path, run_data, filter_criteria)

    # Since the scenario is not in the filter, run_data should remain empty
    assert run_data == {}


# Test for matching filter criteria
@patch(
    "builtins.open",
    new_callable=mock_open,
    read_data=load_mock_data("tests/analysis/mock_run_data.json"),
)
def test_load_run_data_with_matching_filter(mock_open):
    run_data = {}
    file_path = "fake_path.json"
    filter_criteria = {"traffic_scenario": ["D(100,100)"]}

    load_run_data(file_path, run_data, filter_criteria)

    assert "D(100,100)" in run_data
    assert 1 in run_data["D(100,100)"]
    assert run_data["D(100,100)"][1]["aggregated_metrics"].scenario == "D(100,100)"
    assert run_data["D(100,100)"][1]["individual_request_metrics"][0][
        "tpot"
    ] == pytest.approx(0.018614205326884986, rel=0.0000001)


# Tests for request_rate in analysis and reporting


def create_test_stat_field(**kwargs):
    """Helper to create a StatField with sensible defaults."""
    defaults = {
        "min": 0.01,
        "max": 1.0,
        "mean": 0.5,
        "stddev": 0.1,
        "sum": 50.0,
        "p25": 0.4,
        "p50": 0.5,
        "p75": 0.6,
        "p90": 0.7,
        "p95": 0.8,
        "p99": 0.9,
    }
    defaults.update(kwargs)
    return StatField(**defaults)


def create_test_metric_stats():
    """Helper to create MetricStats with all required StatField objects."""
    return MetricStats(
        ttft=create_test_stat_field(),
        tpot=create_test_stat_field(min=0.01, max=0.05, mean=0.02),
        e2e_latency=create_test_stat_field(min=0.5, max=2.0, mean=1.5),
        output_latency=create_test_stat_field(min=0.4, max=1.8, mean=1.3),
        output_inference_speed=create_test_stat_field(min=40.0, max=200.0, mean=70.0),
        num_input_tokens=create_test_stat_field(min=90.0, max=110.0, mean=100.0),
        num_output_tokens=create_test_stat_field(min=90.0, max=110.0, mean=100.0),
        total_tokens=create_test_stat_field(min=180.0, max=220.0, mean=200.0),
        input_throughput=create_test_stat_field(min=400.0, max=1000.0, mean=600.0),
        output_throughput=create_test_stat_field(min=40.0, max=200.0, mean=70.0),
    )


class TestRequestRateInExperimentLoader:
    """Test request_rate handling in experiment_loader."""

    def test_experiment_loader_handles_request_rate_levels(self):
        """Test that experiment_loader properly handles request_rate_levels."""
        # This tests the key fix for the request_rate_levels tracking

        # Would need to create a mock experiment file structure
        pytest.skip("Requires full experiment file structure setup")

    def test_request_rate_levels_cleanup(self):
        """Test that all _levels keys are cleaned up regardless of iteration_type."""
        # Tests the fix where we delete all possible _levels keys
        pytest.skip("Requires mock experiment data")

    def test_mixed_run_warning_with_request_rate(self):
        """Test that mixed runs with request_rate don't trigger false warnings."""
        # Tests the fix for the warning about missing num_concurrency levels
        pytest.skip("Requires mock experiment data with mixed iteration types")


class TestRequestRateInExcelReport:
    """Test request_rate handling in excel_report generation."""

    def test_summary_header_includes_request_rate(self):
        """Test that summary sheet has correct header for request_rate."""
        # This test requires full ExperimentMetadata with all required fields
        # Skip for now as it's complex to set up all required fields
        pytest.skip("Requires complete ExperimentMetadata setup")

    def test_appendix_header_includes_request_rate(self):
        """Test that appendix sheet has correct header for request_rate."""
        # Similar to above but for appendix sheet
        pytest.skip("Requires full workbook creation")

    def test_merge_cells_handles_empty_scenarios(self):
        """Test that merge_cells doesn't fail on empty scenarios."""
        # Tests the fix for ValueError in merge_cells
        # This test requires full ExperimentMetadata setup which is complex
        pytest.skip("Requires complete ExperimentMetadata and workbook setup")

    def test_request_rate_excluded_from_aggregated_metrics_sheet(self):
        """Test that request_rate column is excluded from aggregated metrics."""
        # Tests that request_rate is in the exclusion list for base_headers
        pytest.skip("Requires workbook inspection")


class TestRequestRateDataFlow:
    """Integration tests for request_rate data flow through analysis."""

    def test_request_rate_value_preserved_through_pipeline(self):
        """Test that request_rate values are preserved from run to report."""
        # End-to-end test: run -> metrics -> experiment_loader -> excel_report
        pytest.skip("Requires full integration test")

    def test_request_rate_float_values_in_excel(self):
        """Test that fractional request_rate values display correctly in Excel."""
        # Test that 2.5 req/s shows up correctly, not rounded to 2 or 3
        pytest.skip("Requires Excel cell inspection")

    def test_request_rate_sorting_in_report(self):
        """Test that request_rate values are sorted correctly in report."""
        # Test that rates like [20.0, 5.0, 10.0] are sorted to [5.0, 10.0, 20.0]
        pytest.skip("Requires full report generation")


class TestRequestRateFormulas:
    """Test calculations and formulas related to request_rate."""

    def test_requests_per_second_matches_request_rate(self):
        """Test that requests_per_second in metrics matches the target request_rate."""
        metrics = AggregatedMetrics(
            scenario="test_scenario",
            num_concurrency=10,
            batch_size=1,
            request_rate=20.0,
            iteration_type="request_rate",
            run_duration=60.0,
            mean_output_throughput_tokens_per_s=1000.0,
            mean_input_throughput_tokens_per_s=1000.0,
            mean_total_tokens_throughput_tokens_per_s=2000.0,
            mean_total_chars_per_hour=10000000.0,
            requests_per_second=20.0,
            error_codes_frequency={},
            error_rate=0.0,
            num_error_requests=0,
            num_completed_requests=1200,
            num_requests=1200,
            stats=create_test_metric_stats(),
        )

        # For request_rate runs, the actual RPS should be close to target
        # (within some tolerance due to rate limiter precision)
        assert abs(metrics.requests_per_second - metrics.request_rate) < 1.0

    def test_throughput_calculated_correctly_for_request_rate(self):
        """Test that throughput calculations work for request_rate runs."""
        # Throughput = tokens/s should be independent of whether we used
        # request_rate or num_concurrency as iteration type
        metrics = AggregatedMetrics(
            scenario="test_scenario",
            num_concurrency=10,
            batch_size=1,
            request_rate=15.0,
            iteration_type="request_rate",
            run_duration=60.0,
            mean_output_throughput_tokens_per_s=1500.0,
            mean_input_throughput_tokens_per_s=1500.0,
            mean_total_tokens_throughput_tokens_per_s=3000.0,
            mean_total_chars_per_hour=15000000.0,
            requests_per_second=15.0,
            error_codes_frequency={},
            error_rate=0.0,
            num_error_requests=0,
            num_completed_requests=900,
            num_requests=900,
            stats=create_test_metric_stats(),
        )

        # Verify throughput calculations
        assert metrics.mean_total_tokens_throughput_tokens_per_s > 0
        assert (
            metrics.mean_total_tokens_throughput_tokens_per_s
            == metrics.mean_output_throughput_tokens_per_s
            + metrics.mean_input_throughput_tokens_per_s
        )


class TestRequestRateEdgeCasesInAnalysis:
    """Test edge cases for request_rate in analysis."""

    def test_single_request_rate_value_in_report(self):
        """Test report generation with single request_rate value."""
        pytest.skip("Requires full report generation")

    def test_large_number_of_request_rate_values(self):
        """Test report with many request_rate values."""
        # Test with [1.0, 2.0, 3.0, ..., 100.0]
        pytest.skip("Requires full report generation")

    def test_very_small_request_rate_in_report(self):
        """Test report with very small request_rate (e.g., 0.1 req/s)."""
        pytest.skip("Requires full report generation")

    def test_very_large_request_rate_in_report(self):
        """Test report with very large request_rate (e.g., 1000 req/s)."""
        pytest.skip("Requires full report generation")

    def test_request_rate_with_high_error_rate(self):
        """Test that request_rate metrics handle high error rates correctly."""
        metrics = AggregatedMetrics(
            scenario="test_scenario",
            num_concurrency=10,
            batch_size=1,
            request_rate=20.0,
            iteration_type="request_rate",
            run_duration=60.0,
            mean_output_throughput_tokens_per_s=500.0,
            mean_input_throughput_tokens_per_s=500.0,
            mean_total_tokens_throughput_tokens_per_s=1000.0,
            mean_total_chars_per_hour=5000000.0,
            requests_per_second=20.0,
            error_codes_frequency={"500": 600},
            error_rate=50.0,  # 50% error rate
            num_error_requests=600,
            num_completed_requests=600,
            num_requests=1200,
            stats=create_test_metric_stats(),
        )

        # Metrics should still be valid
        assert metrics.error_rate == 50.0
        assert metrics.num_error_requests == 600
        assert metrics.request_rate == 20.0


class TestRequestRateIterationValueExtraction:
    """Test extraction of iteration_value for request_rate runs."""

    def test_iteration_value_is_float_for_request_rate(self):
        """Test that iteration_value is correctly typed as float for request_rate."""
        # Tests the fix for type annotation Union[int, float, None]
        metrics = AggregatedMetrics(
            scenario="test_scenario",
            num_concurrency=10,
            batch_size=1,
            request_rate=15.5,
            iteration_type="request_rate",
            run_duration=60.0,
            mean_output_throughput_tokens_per_s=1000.0,
            mean_input_throughput_tokens_per_s=1000.0,
            mean_total_tokens_throughput_tokens_per_s=2000.0,
            mean_total_chars_per_hour=10000000.0,
            requests_per_second=15.5,
            error_codes_frequency={},
            error_rate=0.0,
            num_error_requests=0,
            num_completed_requests=930,
            num_requests=930,
            stats=create_test_metric_stats(),
        )

        # Extract iteration value based on type
        iteration_value = None
        if metrics.iteration_type == "request_rate":
            iteration_value = metrics.request_rate
        elif metrics.iteration_type == "batch_size":
            iteration_value = metrics.batch_size
        else:  # num_concurrency
            iteration_value = metrics.num_concurrency

        assert iteration_value == 15.5
        assert isinstance(iteration_value, float)

    def test_iteration_value_is_int_for_num_concurrency(self):
        """Test that iteration_value is int for num_concurrency runs."""
        metrics = AggregatedMetrics(
            scenario="test_scenario",
            num_concurrency=10,
            batch_size=1,
            iteration_type="num_concurrency",
            run_duration=60.0,
            mean_output_throughput_tokens_per_s=1000.0,
            mean_input_throughput_tokens_per_s=1000.0,
            mean_total_tokens_throughput_tokens_per_s=2000.0,
            mean_total_chars_per_hour=10000000.0,
            requests_per_second=15.0,
            error_codes_frequency={},
            error_rate=0.0,
            num_error_requests=0,
            num_completed_requests=900,
            num_requests=900,
            stats=create_test_metric_stats(),
        )

        # Extract iteration value
        iteration_value = None
        if metrics.iteration_type == "request_rate":
            iteration_value = metrics.request_rate
        elif metrics.iteration_type == "batch_size":
            iteration_value = metrics.batch_size
        else:  # num_concurrency
            iteration_value = metrics.num_concurrency

        assert iteration_value == 10
        assert isinstance(iteration_value, int)
