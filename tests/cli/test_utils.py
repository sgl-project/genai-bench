from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from genai_bench.cli.utils import get_experiment_path, get_run_params, manage_run_time


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
        server_engine="SGLang",
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
        server_engine="SGLang",
        server_version="1.0",
        task="text-to-text",
        model="gpt-4",
    )

    expected_name = "openai_SGLang_1.0_text-to-text_gpt-4_20240314_150926"
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
            server_engine="SGLang",
            server_version="1.0",
            task="text-to-text",
            model="gpt-4",
        )

        expected_name = "openai_SGLang_1.0_text-to-text_gpt-4_20240314_150926"
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
        server_engine="SGLang",
        server_version="1.0",
        task="text-to-text",
        model="gpt-4",
    )

    assert path == folder_path
    mock_logger.warning.assert_called_once()


class TestGetRunParams:
    """Test get_run_params function with different iteration types."""

    def test_get_run_params_with_request_rate(self):
        """Test get_run_params() with iteration_type='request_rate'."""
        header, batch_size, num_concurrency = get_run_params(
            iteration_type="request_rate", iteration_value=10.0
        )

        assert header == "Request Rate"
        assert batch_size == 1
        # Concurrency should be rate * 10 = 100
        assert num_concurrency == 100

    def test_get_run_params_request_rate_calculates_concurrency(self):
        """Test that concurrency is calculated as rate * 10 for request_rate."""
        # Test with rate 5.0 -> concurrency should be 50
        header, batch_size, num_concurrency = get_run_params(
            iteration_type="request_rate", iteration_value=5.0
        )
        assert num_concurrency == 50

        # Test with rate 2.5 -> concurrency should be 25
        header, batch_size, num_concurrency = get_run_params(
            iteration_type="request_rate", iteration_value=2.5
        )
        assert num_concurrency == 25

    def test_get_run_params_request_rate_fractional_values(self):
        """Test get_run_params() handles fractional request_rate values."""
        # Test with fractional rate
        header, batch_size, num_concurrency = get_run_params(
            iteration_type="request_rate", iteration_value=2.5
        )

        assert header == "Request Rate"
        assert batch_size == 1
        # 2.5 * 10 = 25
        assert num_concurrency == 25

        # Test with very small fractional rate
        header, batch_size, num_concurrency = get_run_params(
            iteration_type="request_rate", iteration_value=0.1
        )
        # 0.1 * 10 = 1, but minimum is 10
        assert num_concurrency == 10

    def test_get_run_params_request_rate_minimum_concurrency(self):
        """Test get_run_params() ensures minimum concurrency of 10 for request_rate."""
        # Test with very low rate that would give < 10 concurrency
        header, batch_size, num_concurrency = get_run_params(
            iteration_type="request_rate", iteration_value=0.5
        )
        # 0.5 * 10 = 5, but should be clamped to minimum of 10
        assert num_concurrency == 10

        # Test with rate that gives exactly 10
        header, batch_size, num_concurrency = get_run_params(
            iteration_type="request_rate", iteration_value=1.0
        )
        assert num_concurrency == 10

    def test_get_run_params_with_batch_size(self):
        """Test get_run_params() with batch_size iteration type."""
        header, batch_size, num_concurrency = get_run_params(
            iteration_type="batch_size", iteration_value=32
        )

        assert header == "Batch Size"
        assert batch_size == 32
        assert num_concurrency == 1

    def test_get_run_params_with_num_concurrency(self):
        """Test get_run_params() with num_concurrency iteration type."""
        header, batch_size, num_concurrency = get_run_params(
            iteration_type="num_concurrency", iteration_value=16
        )

        assert header == "Concurrency"
        assert batch_size == 1
        assert num_concurrency == 16
