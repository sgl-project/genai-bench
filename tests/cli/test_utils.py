from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from genai_bench.cli.utils import get_experiment_path, manage_run_time


@pytest.fixture
def mock_environment():
    environment = MagicMock()
    return environment


@pytest.fixture
def mock_runner():
    runner = MagicMock()
    return runner


@patch("time.sleep", return_value=None)
def test_exit_based_on_max_run_time(mock_sleep, mock_environment):
    mock_requests = PropertyMock(side_effect=[10, 20, 30, 40, 50])
    type(mock_environment.runner.stats.total).num_requests = mock_requests

    total_run_time = manage_run_time(
        max_time_per_run=5,
        max_requests_per_run=1000,
        environment=mock_environment,
    )

    assert total_run_time == 5  # Should run for 5 secs since max run time is reached


@patch("time.sleep", return_value=None)
def test_exit_based_on_max_requests(mock_sleep, mock_environment):
    mock_requests = PropertyMock(side_effect=[50, 100, 150])
    type(mock_environment.runner.stats.total).num_requests = mock_requests

    total_run_time = manage_run_time(
        max_time_per_run=600,
        max_requests_per_run=150,
        environment=mock_environment,
    )

    assert total_run_time == 3  # Should exit after 3 secs when 150 requests are reached


@pytest.fixture
def mock_datetime():
    with patch("genai_bench.cli.utils.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2024, 3, 14, 15, 9, 26)
        yield mock_dt


def test_get_experiment_path_with_custom_name(tmp_path):
    """Test get_experiment_path with a custom folder name"""
    custom_name = "my_experiment"
    path = get_experiment_path(
        experiment_folder_name=custom_name,
        experiment_base_dir=str(tmp_path),
        api_backend="openai",
        server_engine="vLLM",
        server_version="1.0",
        task="text-to-text",
        model="gpt-4",
    )

    assert path == tmp_path / custom_name
    assert path.exists()


def test_get_experiment_path_default_name(mock_datetime, tmp_path):
    """Test get_experiment_path with default name generation"""
    path = get_experiment_path(
        experiment_folder_name=None,
        experiment_base_dir=str(tmp_path),
        api_backend="openai",
        server_engine="vLLM",
        server_version="1.0",
        task="text-to-text",
        model="gpt-4",
    )

    expected_name = "openai_vLLM_1.0_text-to-text_gpt-4_20240314_150926"
    assert path == tmp_path / expected_name
    assert path.exists()


def test_get_experiment_path_no_server_info(mock_datetime, tmp_path):
    """Test get_experiment_path without server engine and version"""
    path = get_experiment_path(
        experiment_folder_name=None,
        experiment_base_dir=str(tmp_path),
        api_backend="openai",
        server_engine=None,
        server_version=None,
        task="text-to-text",
        model="gpt-4",
    )

    expected_name = "openai_text-to-text_gpt-4_20240314_150926"
    assert path == tmp_path / expected_name
    assert path.exists()


def test_get_experiment_path_no_base_dir(mock_datetime):
    """Test get_experiment_path without base directory"""
    with patch("pathlib.Path.mkdir") as mock_mkdir:
        path = get_experiment_path(
            experiment_folder_name=None,
            experiment_base_dir=None,
            api_backend="openai",
            server_engine="vLLM",
            server_version="1.0",
            task="text-to-text",
            model="gpt-4",
        )

        expected_name = "openai_vLLM_1.0_text-to-text_gpt-4_20240314_150926"
        assert path == Path(expected_name)
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)


@patch("genai_bench.cli.utils.logger")
def test_get_experiment_path_existing_folder(mock_logger, tmp_path):
    """Test get_experiment_path with existing folder"""
    folder_name = "existing_experiment"
    folder_path = tmp_path / folder_name
    folder_path.mkdir(parents=True)

    path = get_experiment_path(
        experiment_folder_name=folder_name,
        experiment_base_dir=str(tmp_path),
        api_backend="openai",
        server_engine="vLLM",
        server_version="1.0",
        task="text-to-text",
        model="gpt-4",
    )

    assert path == folder_path
    mock_logger.warning.assert_called_once()
