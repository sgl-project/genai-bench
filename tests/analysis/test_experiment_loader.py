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
