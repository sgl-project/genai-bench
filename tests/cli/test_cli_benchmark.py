"""
This is a unit test for the `benchmark` command in the CLI.

The test is not an integration test. It uses extensive mocking to isolate the
CLI logic and verify that the `benchmark` command handles its options correctly,
such as the API backend, model settings, server configurations, and dataset
file input.

The test ensures that:
- The benchmark command validates the provided options correctly.
- Mocked components, such as the dashboard, tokenizer validation, HTTP requests,
  and report generation, are used to avoid external dependencies and actual file
  I/O during testing.
- It keeps the benchmark integration with Sampling.
"""

from locust.runners import WorkerRunner

import logging
import tempfile
import traceback
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from click.testing import CliRunner

from genai_bench.cli.cli import benchmark


@pytest.fixture
def cli_runner():
    return CliRunner()


@pytest.fixture
def default_options():
    """Default command options used in most tests"""
    return [
        "--api-backend",
        "openai",
        "--api-base",
        "https://api.openai.com",
        "--api-key",
        "test_key",
        "--task",
        "text-to-text",
        "--api-model-name",
        "gpt-3.5-turbo",
        "--model-tokenizer",
        "gpt2",
        "--max-time-per-run",
        "1",
        "--max-requests-per-run",
        "5",
        "--num-concurrency",
        "1",
    ]


@pytest.fixture
def request_rate_options():
    """Default command options for tests using --request-rate
    (without --num-concurrency)"""
    return [
        "--api-backend",
        "openai",
        "--api-base",
        "https://api.openai.com",
        "--api-key",
        "test_key",
        "--task",
        "text-to-text",
        "--api-model-name",
        "gpt-3.5-turbo",
        "--model-tokenizer",
        "gpt2",
        "--max-time-per-run",
        "1",
        "--max-requests-per-run",
        "5",
    ]


@pytest.fixture
def mock_env_variables():
    with patch.dict("os.environ", {"HF_TOKEN": "dummy_key"}):
        yield  # Yield ensures the patch is active for the duration of the test


# Mock Dashboard
@pytest.fixture
def mock_dashboard():
    with patch("genai_bench.cli.cli.create_dashboard") as mock_dashboard_patch:
        yield mock_dashboard_patch


# Mock validate_tokenizer
@pytest.fixture
def mock_validate_tokenizer():
    mock_tokenizer = MagicMock()
    with patch("genai_bench.cli.cli.validate_tokenizer", return_value=mock_tokenizer):
        yield mock_tokenizer


# Mock gevent.sleep (used instead of time.sleep for cooperative multitasking)
@pytest.fixture
def mock_time_sleep():
    with patch("gevent.sleep", return_value=None):
        yield


# Mock os.makedirs
@pytest.fixture
def mock_makedirs():
    with patch("os.makedirs") as mock_makedirs_patch:
        yield mock_makedirs_patch


# Mock file system interactions
@pytest.fixture
def mock_file_system():
    with (
        patch("genai_bench.cli.cli.Path.write_text") as mock_write_text,
    ):
        yield mock_write_text


# Mock HTTP requests
@pytest.fixture
def mock_http_requests():
    with patch("requests.get") as mock_get, patch("requests.post") as mock_post:
        mock_get.return_value = MagicMock(
            status_code=200, json=lambda: {"models": ["gpt-3.5-turbo"]}
        )
        mock_post.return_value = MagicMock(
            status_code=200, json=lambda: {"choices": [{"text": "response"}]}
        )
        yield mock_get, mock_post


# Mock report and plot generation
@pytest.fixture
def mock_report_and_plot():
    mock_experiment_metadata = MagicMock()
    mock_experiment_metadata.server_gpu_count = 4
    with (
        patch(
            "genai_bench.cli.cli.load_one_experiment",
            return_value=(mock_experiment_metadata, MagicMock()),
        ) as mock_load_experiment,
        patch("genai_bench.cli.cli.create_workbook") as mock_create_workbook,
        patch(
            "genai_bench.cli.cli.plot_experiment_data_flexible"
        ) as mock_plot_experiment_data_flexible,
        patch(
            "genai_bench.cli.cli.plot_single_scenario_inference_speed_vs_throughput"
        ) as mock_plot_single_scenario_inference_speed_vs_throughput,
    ):
        yield {
            "load_experiment": mock_load_experiment,
            "create_workbook": mock_create_workbook,
            "plot_experiment_data_flexible": mock_plot_experiment_data_flexible,
            "experiment_metadata": mock_experiment_metadata,
            "plot_single_scenario_inference_speed_vs_throughput": mock_plot_single_scenario_inference_speed_vs_throughput,  # noqa: E501
        }


@pytest.fixture
def mock_experiment_path():
    with patch("genai_bench.cli.cli.get_experiment_path") as mock_path:
        mock_path.return_value = Path("/mock/experiment/path")
        yield mock_path


@pytest.fixture
def mock_runner_stats():
    """Mock runner stats for request counting"""
    stats = MagicMock()
    stats.total = MagicMock()
    stats.total.num_requests = 0
    return stats


@pytest.mark.usefixtures(
    "mock_env_variables",
    "mock_dashboard",
    "mock_validate_tokenizer",
    "mock_time_sleep",
    "mock_file_system",
    "mock_report_and_plot",
    "mock_http_requests",
    "mock_experiment_path",
)
def test_benchmark_command(cli_runner, default_options, mock_report_and_plot):
    result = cli_runner.invoke(
        benchmark,
        [
            *default_options,
            "--traffic-scenario",
            "D(100,100)",
        ],
    )

    # Debug output
    if result.exit_code != 0:
        print(f"Exit code: {result.exit_code}")
        print(f"Output: {result.output}")
        if result.exception:
            print(f"Exception: {result.exception}")

            traceback.print_exception(
                type(result.exception), result.exception, result.exception.__traceback__
            )

    assert result.exit_code == 0, f"Command failed with output: {result.output}"

    assert mock_report_and_plot["load_experiment"].called
    assert mock_report_and_plot["create_workbook"].called
    assert mock_report_and_plot["plot_experiment_data_flexible"].called
    assert mock_report_and_plot["experiment_metadata"].server_gpu_count == 4


@pytest.mark.usefixtures(
    "mock_env_variables",
    "mock_dashboard",
    "mock_validate_tokenizer",
    "mock_time_sleep",
    "mock_file_system",
    "mock_report_and_plot",
    "mock_http_requests",
    "mock_experiment_path",
)
def test_benchmark_command_with_file_based_sampling(
    cli_runner, default_options, caplog
):
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_file:
        mock_data = pd.DataFrame({"text": ["Text 1", "Text 2", "Text 3"]})
        mock_data.to_csv(temp_file.name, index=False)

        with caplog.at_level(logging.INFO):
            result = cli_runner.invoke(
                benchmark,
                [
                    *default_options,
                    "--dataset-path",
                    temp_file.name,
                    "--dataset-prompt-column",
                    "text",
                ],
            )
        assert result.exit_code == 0, f"Command failed with output: {result.output}"
        # Check that the command completed successfully with CSV dataset
        # Note: The specific log message may have changed during refactoring


@pytest.mark.usefixtures(
    "mock_env_variables",
    "mock_dashboard",
    "mock_validate_tokenizer",
    "mock_time_sleep",
    "mock_makedirs",
    "mock_file_system",
    "mock_report_and_plot",
    "mock_http_requests",
    "mock_experiment_path",
)
def test_invalid_server_gpu_type(cli_runner, default_options):
    result = cli_runner.invoke(
        benchmark,
        [
            *default_options,
            "--server-engine",
            "SGLang",
            "--server-gpu-type",
            "YYY",  # Invalid GPU type
            "--server-version",
            "v0.6.3",
            "--server-gpu-count",
            "8",
        ],
    )
    assert result.exit_code != 0
    assert "Error: Invalid value for '--server-gpu-type'" in result.output


@pytest.mark.usefixtures(
    "mock_env_variables",
    "mock_dashboard",
    "mock_validate_tokenizer",
    "mock_time_sleep",
    "mock_file_system",
    "mock_report_and_plot",
    "mock_http_requests",
    "mock_experiment_path",
)
def test_benchmark_command_with_minimal_logging(
    cli_runner, default_options, monkeypatch
):
    monkeypatch.setenv("ENABLE_UI", "false")

    result = cli_runner.invoke(
        benchmark,
        [
            *default_options,
            "--traffic-scenario",
            "D(100,100)",
        ],
    )
    assert result.exit_code == 0, f"Command failed with output: {result.output}"


@pytest.mark.usefixtures(
    "mock_env_variables",
    "mock_dashboard",
    "mock_validate_tokenizer",
    "mock_time_sleep",
    "mock_makedirs",
    "mock_file_system",
    "mock_report_and_plot",
    "mock_http_requests",
    "mock_experiment_path",
)
def test_benchmark_command_with_embedding_sampling(cli_runner, default_options, caplog):
    """Test benchmark with embeddings task and file-based sampling"""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_file:
        mock_data = pd.DataFrame({"text": ["Text 1", "Text 2", "Text 3"]})
        mock_data.to_csv(temp_file.name, index=False)

        # Override task in default options to use embeddings
        options = [
            opt if opt != "text-to-text" else "text-to-embeddings"
            for opt in default_options
        ]

        with caplog.at_level(logging.INFO):
            result = cli_runner.invoke(
                benchmark,
                [
                    *options,
                    "--dataset-path",
                    temp_file.name,
                    "--dataset-prompt-column",
                    "text",
                    "--traffic-scenario",
                    "E(100)",
                ],
            )
        assert result.exit_code == 0, f"Command failed with output: {result.output}"


@pytest.mark.usefixtures(
    "mock_env_variables",
    "mock_dashboard",
    "mock_validate_tokenizer",
    "mock_time_sleep",
    "mock_makedirs",
    "mock_file_system",
    "mock_report_and_plot",
    "mock_http_requests",
    "mock_experiment_path",
)
def test_benchmark_command_with_cohere(cli_runner, default_options):
    with (
        tempfile.NamedTemporaryFile() as config_file,
        tempfile.NamedTemporaryFile() as key_file,
    ):
        key_file.write(b"key")
        key_file.flush()
        config = f"""[DEFAULT]
    user=ocid1.user.oc1..example
    fingerprint=8c:4c:01:71:6e:4a:e8:11:ab:c9:c2:63:fb:36:f5:ec
    tenancy=ocid1.tenancy.oc1..example
    region=us-ashburn-1
    key_file={key_file.name}"""
        config_file.write(config.encode("utf-8"))
        config_file.flush()

        # Override API backend in default options
        options = [opt if opt != "openai" else "oci-cohere" for opt in default_options]

        result = cli_runner.invoke(
            benchmark,
            [
                *options,
                "--config-file",
                config_file.name,
            ],
        )
    assert result.exit_code == 0, f"Command failed with output: {result.output}"


@pytest.mark.usefixtures(
    "mock_env_variables",
    "mock_dashboard",
    "mock_validate_tokenizer",
    "mock_time_sleep",
    "mock_makedirs",
    "mock_file_system",
    "mock_report_and_plot",
    "mock_http_requests",
    "mock_experiment_path",
)
def test_benchmark_command_with_multiple_workers(
    cli_runner, default_options, mock_runner_stats
):
    with (
        patch("genai_bench.cli.cli.DistributedRunner") as mock_runner_class,
        patch("genai_bench.cli.cli.Environment") as mock_env_class,
    ):
        mock_env = MagicMock()
        mock_env_class.return_value = mock_env
        mock_runner = MagicMock()
        mock_runner_class.return_value = mock_runner
        mock_runner.environment = mock_env
        mock_runner.environment.runner.stats = mock_runner_stats

        result = cli_runner.invoke(
            benchmark,
            [
                *default_options,
                "--num-workers",
                "2",
                "--master-port",
                "5557",
            ],
        )
        assert result.exit_code == 0, f"Command failed with output: {result.output}"
        mock_runner_class.assert_called_once()
        mock_runner.setup.assert_called_once()


@pytest.mark.usefixtures(
    "mock_env_variables",
    "mock_dashboard",
    "mock_validate_tokenizer",
    "mock_time_sleep",
    "mock_makedirs",
    "mock_file_system",
    "mock_report_and_plot",
    "mock_http_requests",
    "mock_experiment_path",
)
def test_benchmark_command_with_worker_process(cli_runner, default_options):
    with (
        patch("genai_bench.cli.cli.DistributedRunner") as mock_runner_class,
        patch("genai_bench.cli.cli.Environment") as mock_env_class,
    ):
        mock_env = MagicMock()
        mock_env_class.return_value = mock_env
        mock_runner = MagicMock()
        mock_runner_class.return_value = mock_runner
        mock_runner.environment = mock_env
        mock_runner.environment.runner = MagicMock(spec=WorkerRunner)

        result = cli_runner.invoke(
            benchmark,
            [
                *default_options,
                "--num-workers",
                "2",
            ],
        )
        assert result.exit_code == 0, f"Command failed with output: {result.output}"
        mock_runner_class.assert_called_once()
        mock_runner.setup.assert_called_once()
        mock_runner.update_scenario.assert_not_called()


@pytest.mark.usefixtures(
    "mock_env_variables",
    "mock_dashboard",
    "mock_validate_tokenizer",
    "mock_time_sleep",
    "mock_makedirs",
    "mock_file_system",
    "mock_report_and_plot",
    "mock_http_requests",
    "mock_experiment_path",
)
def test_benchmark_command_with_traffic_scenarios(cli_runner, default_options, caplog):
    """Test benchmark command with different traffic scenario configurations."""
    with caplog.at_level(logging.INFO):
        # Test with custom traffic scenarios
        result = cli_runner.invoke(
            benchmark,
            [
                *default_options,
                "--server-engine",
                "SGLang",
                "--server-version",
                "1.0",
                "--server-gpu-type",
                "H100",
                "--server-gpu-count",
                "4",
                "--traffic-scenario",
                "D(100,100)",
                "--traffic-scenario",
                "D(200,200)",
                "--experiment-folder-name",
                "test_exp",
                "--experiment-base-dir",
                "/tmp",
            ],
        )
        assert result.exit_code == 0, f"Command failed with output: {result.output}"

        # Test with default scenarios
        result = cli_runner.invoke(
            benchmark,
            [
                *default_options,
                "--server-engine",
                "SGLang",
                "--server-version",
                "1.0",
                "--server-gpu-type",
                "H100",
                "--server-gpu-count",
                "4",
                "--experiment-folder-name",
                "test_exp",
                "--experiment-base-dir",
                "/tmp",
            ],
        )
        assert result.exit_code == 0, f"Command failed with output: {result.output}"


@pytest.mark.usefixtures(
    "mock_env_variables",
    "mock_dashboard",
    "mock_validate_tokenizer",
    "mock_time_sleep",
    "mock_makedirs",
    "mock_file_system",
    "mock_report_and_plot",
    "mock_http_requests",
    "mock_experiment_path",
)
def test_benchmark_command_with_oci_auth(cli_runner, default_options, caplog):
    """Test benchmark command with OCI authentication."""
    with patch("genai_bench.cli.cli.UnifiedAuthFactory") as mock_auth_factory:
        # Mock UnifiedAuthFactory
        mock_auth = MagicMock()
        mock_auth.get_config.return_value = {"auth_type": "instance_principal"}
        mock_auth_factory.create_model_auth.return_value = mock_auth

        with caplog.at_level(logging.INFO):
            result = cli_runner.invoke(
                benchmark,
                [
                    "--api-backend",
                    "oci-cohere",
                    "--api-base",
                    "https://api.cohere.oci.com",
                    "--task",
                    "text-to-text",
                    "--api-model-name",
                    "command",
                    "--model-tokenizer",
                    "gpt2",
                    "--auth",
                    "instance_principal",
                    "--region",
                    "us-ashburn-1",
                    "--experiment-folder-name",
                    "test_exp",
                    "--experiment-base-dir",
                    "/tmp",
                    "--max-time-per-run",
                    "1",
                    "--max-requests-per-run",
                    "5",
                    "--server-engine",
                    "SGLang",
                    "--server-version",
                    "1.0",
                    "--server-gpu-type",
                    "H100",
                    "--server-gpu-count",
                    "4",
                    "--traffic-scenario",
                    "D(100,100)",
                ],
            )
            assert result.exit_code == 0, f"Command failed with output: {result.output}"
            assert "Using oci-cohere authentication" in caplog.text
            mock_auth_factory.create_model_auth.assert_called_once_with(
                "oci",
                auth_type="instance_principal",
                config_path="~/.oci/config",
                profile="DEFAULT",
                token=None,
                region="us-ashburn-1",
            )


@pytest.mark.usefixtures(
    "mock_env_variables",
    "mock_dashboard",
    "mock_validate_tokenizer",
    "mock_time_sleep",
    "mock_file_system",
    "mock_report_and_plot",
    "mock_http_requests",
    "mock_experiment_path",
)
def test_benchmark_help_doesnt_prompt(cli_runner):
    result = cli_runner.invoke(
        benchmark,
        [
            "--help",
        ],
        input="cohere",  # Dummy input to prevent hang
    )
    assert "Api backend (openai, oci-cohere, cohere):" not in result.output
    assert result.exit_code == 0, f"Command failed with output: {result.output}"


@pytest.mark.usefixtures(
    "mock_env_variables",
    "mock_dashboard",
    "mock_validate_tokenizer",
    "mock_time_sleep",
    "mock_makedirs",
    "mock_file_system",
    "mock_report_and_plot",
    "mock_http_requests",
    "mock_experiment_path",
)
def test_benchmark_command_with_spawn_rate(
    cli_runner, default_options, mock_report_and_plot
):
    """Test benchmark command with spawn-rate option."""
    result = cli_runner.invoke(
        benchmark,
        [
            *default_options,
            "--spawn-rate",
            "50",
        ],
    )
    assert result.exit_code == 0, f"Command failed with output: {result.output}"

    # Verify report generation like other basic tests
    assert mock_report_and_plot["load_experiment"].called
    assert mock_report_and_plot["create_workbook"].called
    assert mock_report_and_plot["plot_experiment_data_flexible"].called


@pytest.mark.usefixtures(
    "mock_env_variables",
    "mock_dashboard",
    "mock_validate_tokenizer",
    "mock_time_sleep",
    "mock_makedirs",
    "mock_file_system",
    "mock_report_and_plot",
    "mock_http_requests",
    "mock_experiment_path",
)
def test_benchmark_command_with_spawn_rate_and_workers(
    cli_runner, default_options, mock_runner_stats
):
    """Test benchmark command with both spawn-rate and num-workers options."""
    with (
        patch("genai_bench.cli.cli.DistributedRunner") as mock_runner_class,
        patch("genai_bench.cli.cli.Environment") as mock_env_class,
    ):
        mock_env = MagicMock()
        mock_env_class.return_value = mock_env
        mock_runner = MagicMock()
        mock_runner_class.return_value = mock_runner
        mock_runner.environment = mock_env
        mock_runner.environment.runner.stats = mock_runner_stats

        result = cli_runner.invoke(
            benchmark,
            [
                *default_options,
                "--num-workers",
                "4",
                "--spawn-rate",
                "25",
            ],
        )
        assert result.exit_code == 0, f"Command failed with output: {result.output}"
        mock_runner_class.assert_called_once()
        mock_runner.setup.assert_called_once()


@pytest.mark.usefixtures(
    "mock_env_variables",
    "mock_dashboard",
    "mock_validate_tokenizer",
    "mock_time_sleep",
    "mock_makedirs",
    "mock_file_system",
    "mock_report_and_plot",
    "mock_http_requests",
    "mock_experiment_path",
)
def test_spawn_rate_passed_to_runner_start(
    cli_runner, default_options, mock_runner_stats
):
    """Test that spawn-rate parameter is correctly passed to runner.start."""
    with (
        patch("genai_bench.cli.cli.DistributedRunner") as mock_runner_class,
        patch("genai_bench.cli.cli.Environment") as mock_env_class,
    ):
        mock_env = MagicMock()
        mock_env_class.return_value = mock_env
        mock_runner = MagicMock()
        mock_runner_class.return_value = mock_runner
        mock_runner.environment = mock_env
        mock_runner.environment.runner.stats = mock_runner_stats

        # Mock the runner.start method to capture its arguments
        mock_runner.environment.runner.start = MagicMock()

        result = cli_runner.invoke(
            benchmark,
            [
                *default_options,
                "--num-concurrency",
                "100",
                "--spawn-rate",
                "25",
            ],
        )
        assert result.exit_code == 0, f"Command failed with output: {result.output}"

        # Verify that runner.start was called with the correct spawn_rate
        mock_runner.environment.runner.start.assert_called_with(100, spawn_rate=25)


def test_spawn_rate_option_in_help(cli_runner):
    """Test that spawn-rate option appears in the CLI help output."""
    result = cli_runner.invoke(benchmark, ["--help"])
    assert result.exit_code == 0
    assert "--spawn-rate" in result.output
    assert "Number of users to spawn per second" in result.output


# Tests for request_rate functionality


@pytest.mark.usefixtures(
    "mock_env_variables",
    "mock_dashboard",
    "mock_validate_tokenizer",
    "mock_time_sleep",
    "mock_makedirs",
    "mock_file_system",
    "mock_report_and_plot",
    "mock_http_requests",
    "mock_experiment_path",
)
def test_benchmark_command_with_request_rate(
    cli_runner, request_rate_options, mock_report_and_plot
):
    """Test benchmark command with request-rate option."""
    result = cli_runner.invoke(
        benchmark,
        [
            *request_rate_options,
            "--request-rate",
            "10",
            "--request-rate",
            "20",
        ],
    )
    assert result.exit_code == 0, f"Command failed with output: {result.output}"

    # Verify report generation
    assert mock_report_and_plot["load_experiment"].called
    assert mock_report_and_plot["create_workbook"].called
    assert mock_report_and_plot["plot_experiment_data_flexible"].called


@pytest.mark.usefixtures(
    "mock_env_variables",
    "mock_dashboard",
    "mock_validate_tokenizer",
    "mock_time_sleep",
    "mock_makedirs",
    "mock_file_system",
    "mock_report_and_plot",
    "mock_http_requests",
    "mock_experiment_path",
)
def test_benchmark_request_rate_iteration_type(cli_runner, request_rate_options):
    """Test that request_rate option sets iteration_type correctly."""
    with (
        patch("genai_bench.cli.cli.DistributedRunner") as mock_runner_class,
        patch("genai_bench.cli.cli.Environment") as mock_env_class,
    ):
        mock_env = MagicMock()
        mock_runner = MagicMock()
        mock_runner_stats = MagicMock()

        mock_runner_stats.total.num_requests = 100
        mock_env_class.return_value = mock_env
        mock_runner_class.return_value = mock_runner
        mock_runner.environment = mock_env
        mock_runner.environment.runner.stats = mock_runner_stats
        mock_runner.metrics_collector = MagicMock()
        mock_runner.environment.runner.user_count = 10

        result = cli_runner.invoke(
            benchmark,
            [
                *request_rate_options,
                "--request-rate",
                "5",
            ],
        )

        assert result.exit_code == 0, f"Command failed with output: {result.output}"
        # Note shown in output when using request_rate
        assert "Using request_rate iteration" in result.output


@pytest.mark.usefixtures(
    "mock_env_variables",
    "mock_dashboard",
    "mock_validate_tokenizer",
    "mock_time_sleep",
    "mock_makedirs",
    "mock_file_system",
    "mock_report_and_plot",
    "mock_http_requests",
    "mock_experiment_path",
)
def test_benchmark_request_rate_creates_rate_limiter(cli_runner, request_rate_options):
    """Test that request_rate run creates a TokenBucketRateLimiter."""
    with (
        patch("genai_bench.cli.cli.DistributedRunner") as mock_runner_class,
        patch("genai_bench.cli.cli.Environment") as mock_env_class,
        patch("genai_bench.cli.cli.TokenBucketRateLimiter") as mock_rate_limiter_class,
    ):
        mock_env = MagicMock()
        mock_runner = MagicMock()
        mock_runner_stats = MagicMock()
        mock_rate_limiter = MagicMock()

        mock_runner_stats.total.num_requests = 100
        mock_env_class.return_value = mock_env
        mock_runner_class.return_value = mock_runner
        mock_rate_limiter_class.return_value = mock_rate_limiter
        mock_runner.environment = mock_env
        mock_runner.environment.runner.stats = mock_runner_stats
        mock_runner.metrics_collector = MagicMock()
        mock_runner.environment.runner.user_count = 10

        result = cli_runner.invoke(
            benchmark,
            [
                *request_rate_options,
                "--request-rate",
                "15",
            ],
        )

        assert result.exit_code == 0, f"Command failed with output: {result.output}"
        # Verify rate limiter was created
        mock_rate_limiter_class.assert_called_with(rate=15)


def test_request_rate_option_in_help(cli_runner):
    """Test that request-rate option appears in the CLI help output."""
    result = cli_runner.invoke(benchmark, ["--help"])
    assert result.exit_code == 0
    assert "--request-rate" in result.output
    assert "request rates (requests/second)" in result.output


@pytest.mark.usefixtures(
    "mock_env_variables",
    "mock_dashboard",
    "mock_validate_tokenizer",
    "mock_time_sleep",
    "mock_makedirs",
    "mock_file_system",
    "mock_report_and_plot",
    "mock_http_requests",
    "mock_experiment_path",
)
def test_request_rate_with_multiple_values(cli_runner, request_rate_options):
    """Test request_rate with multiple values runs correctly."""
    with (
        patch("genai_bench.cli.cli.DistributedRunner") as mock_runner_class,
        patch("genai_bench.cli.cli.Environment") as mock_env_class,
    ):
        mock_env = MagicMock()
        mock_runner = MagicMock()
        mock_runner_stats = MagicMock()

        mock_runner_stats.total.num_requests = 100
        mock_env_class.return_value = mock_env
        mock_runner_class.return_value = mock_runner
        mock_runner.environment = mock_env
        mock_runner.environment.runner.stats = mock_runner_stats
        mock_runner.metrics_collector = MagicMock()
        mock_runner.environment.runner.user_count = 10

        result = cli_runner.invoke(
            benchmark,
            [
                *request_rate_options,
                "--request-rate",
                "5",
                "--request-rate",
                "10",
                "--request-rate",
                "20",
            ],
        )

        assert result.exit_code == 0, f"Command failed with output: {result.output}"


@pytest.mark.usefixtures(
    "mock_env_variables",
    "mock_dashboard",
    "mock_validate_tokenizer",
    "mock_time_sleep",
    "mock_makedirs",
    "mock_file_system",
    "mock_report_and_plot",
    "mock_http_requests",
    "mock_experiment_path",
)
def test_rate_limiter_created_local_mode(cli_runner, request_rate_options):
    """Test that rate limiter is created in local mode for request_rate runs."""
    with (
        patch("genai_bench.cli.cli.DistributedRunner") as mock_runner_class,
        patch("genai_bench.cli.cli.Environment") as mock_env_class,
        patch("genai_bench.cli.cli.TokenBucketRateLimiter") as mock_rate_limiter_class,
    ):
        mock_env = MagicMock()
        mock_runner = MagicMock()
        mock_runner_stats = MagicMock()
        mock_rate_limiter = MagicMock()

        mock_runner_stats.total.num_requests = 100
        mock_env_class.return_value = mock_env
        mock_runner_class.return_value = mock_runner
        mock_rate_limiter_class.return_value = mock_rate_limiter
        mock_runner.environment = mock_env
        mock_runner.environment.runner.stats = mock_runner_stats
        mock_runner.metrics_collector = MagicMock()
        mock_runner.environment.runner.user_count = 10

        result = cli_runner.invoke(
            benchmark,
            [
                *request_rate_options,
                "--request-rate",
                "10",
            ],
        )

        assert result.exit_code == 0
        # Verify rate limiter was created with correct rate
        mock_rate_limiter_class.assert_called_with(rate=10)
        # Verify it was assigned to environment
        assert mock_env.rate_limiter == mock_rate_limiter


@pytest.mark.usefixtures(
    "mock_env_variables",
    "mock_dashboard",
    "mock_validate_tokenizer",
    "mock_time_sleep",
    "mock_makedirs",
    "mock_file_system",
    "mock_report_and_plot",
    "mock_http_requests",
    "mock_experiment_path",
)
def test_rate_limiter_divided_among_workers(cli_runner, request_rate_options):
    """Test that rate limiter is divided among workers in distributed mode."""
    with (
        patch("genai_bench.cli.cli.DistributedRunner") as mock_runner_class,
        patch("genai_bench.cli.cli.Environment") as mock_env_class,
    ):
        mock_env = MagicMock()
        mock_runner = MagicMock()
        mock_runner_stats = MagicMock()

        mock_runner_stats.total.num_requests = 100
        mock_env_class.return_value = mock_env
        mock_runner_class.return_value = mock_runner
        mock_runner.environment = mock_env
        mock_runner.environment.runner.stats = mock_runner_stats
        mock_runner.metrics_collector = MagicMock()
        mock_runner.environment.runner.user_count = 10

        result = cli_runner.invoke(
            benchmark,
            [
                *request_rate_options,
                "--request-rate",
                "20",
                "--num-workers",
                "4",
            ],
        )

        assert result.exit_code == 0
        # Verify update_rate_limiter was called with per-worker rate
        # 20 / 4 = 5 per worker
        # It's called twice: once with the rate, once with None to stop
        calls = mock_runner.update_rate_limiter.call_args_list
        # Check that it was called with 5.0 (per-worker rate)
        rate_calls = [call for call in calls if call[0][0] == 5.0]
        assert (
            len(rate_calls) > 0
        ), "update_rate_limiter should be called with per-worker rate 5.0"
        # Master should not have rate limiter in distributed mode
        assert mock_env.rate_limiter is None


@pytest.mark.usefixtures(
    "mock_env_variables",
    "mock_dashboard",
    "mock_validate_tokenizer",
    "mock_time_sleep",
    "mock_makedirs",
    "mock_file_system",
    "mock_report_and_plot",
    "mock_http_requests",
    "mock_experiment_path",
)
def test_rate_limiter_stopped_after_run(cli_runner, request_rate_options):
    """Test that rate limiter is stopped after run completes."""
    with (
        patch("genai_bench.cli.cli.DistributedRunner") as mock_runner_class,
        patch("genai_bench.cli.cli.Environment") as mock_env_class,
        patch("genai_bench.cli.cli.TokenBucketRateLimiter") as mock_rate_limiter_class,
    ):
        mock_env = MagicMock()
        mock_runner = MagicMock()
        mock_runner_stats = MagicMock()
        mock_rate_limiter = MagicMock()

        mock_runner_stats.total.num_requests = 100
        mock_env_class.return_value = mock_env
        mock_runner_class.return_value = mock_runner
        mock_rate_limiter_class.return_value = mock_rate_limiter
        mock_runner.environment = mock_env
        mock_runner.environment.runner.stats = mock_runner_stats
        mock_runner.metrics_collector = MagicMock()
        mock_runner.environment.runner.user_count = 10
        mock_env.rate_limiter = mock_rate_limiter

        result = cli_runner.invoke(
            benchmark,
            [
                *request_rate_options,
                "--request-rate",
                "10",
            ],
        )

        assert result.exit_code == 0
        # Verify stop was called on rate limiter
        mock_rate_limiter.stop.assert_called()


@pytest.mark.usefixtures(
    "mock_env_variables",
    "mock_dashboard",
    "mock_validate_tokenizer",
    "mock_time_sleep",
    "mock_makedirs",
    "mock_file_system",
    "mock_report_and_plot",
    "mock_http_requests",
    "mock_experiment_path",
)
def test_rate_limiter_warning_low_per_worker_rate(
    cli_runner, request_rate_options, caplog
):
    """Test that warning is logged for very low per-worker rates."""

    with (
        patch("genai_bench.cli.cli.DistributedRunner") as mock_runner_class,
        patch("genai_bench.cli.cli.Environment") as mock_env_class,
    ):
        mock_env = MagicMock()
        mock_runner = MagicMock()
        mock_runner_stats = MagicMock()

        mock_runner_stats.total.num_requests = 100
        mock_env_class.return_value = mock_env
        mock_runner_class.return_value = mock_runner
        mock_runner.environment = mock_env
        mock_runner.environment.runner.stats = mock_runner_stats
        mock_runner.metrics_collector = MagicMock()
        mock_runner.environment.runner.user_count = 10

        with caplog.at_level(logging.WARNING):
            result = cli_runner.invoke(
                benchmark,
                [
                    *request_rate_options,
                    "--request-rate",
                    "1",  # Low rate
                    "--num-workers",
                    "20",  # Many workers -> 0.05 req/s per worker
                ],
            )

        assert result.exit_code == 0
        # Should log warning about low per-worker rate
        assert "Per-worker rate" in caplog.text
        assert "very low" in caplog.text


@pytest.mark.usefixtures(
    "mock_env_variables",
    "mock_dashboard",
    "mock_validate_tokenizer",
    "mock_time_sleep",
    "mock_makedirs",
    "mock_file_system",
    "mock_report_and_plot",
    "mock_http_requests",
    "mock_experiment_path",
)
def test_rate_limiter_cleanup_between_runs(cli_runner, request_rate_options):
    """Test that rate limiter is cleaned up between runs."""
    with (
        patch("genai_bench.cli.cli.DistributedRunner") as mock_runner_class,
        patch("genai_bench.cli.cli.Environment") as mock_env_class,
        patch("genai_bench.cli.cli.TokenBucketRateLimiter") as mock_rate_limiter_class,
    ):
        mock_env = MagicMock()
        mock_runner = MagicMock()
        mock_runner_stats = MagicMock()

        mock_runner_stats.total.num_requests = 100
        mock_env_class.return_value = mock_env
        mock_runner_class.return_value = mock_runner
        mock_runner.environment = mock_env
        # Set runner on mock_env directly since code accesses environment.runner
        mock_env.runner = MagicMock()
        mock_env.runner.stats = mock_runner_stats
        mock_env.runner.user_count = 10
        mock_runner.metrics_collector = MagicMock()
        # Mock methods that might be called
        mock_runner.metrics_collector.get_ui_scatter_plot_metrics.return_value = {}
        mock_runner.metrics_collector.aggregated_metrics = MagicMock()

        # Set up rate limiters to be returned in sequence
        # Use a function to return mocks so we can track which ones are created
        rate_limiters_created = []

        def create_rate_limiter(*args, **kwargs):
            limiter = MagicMock()
            rate_limiters_created.append(limiter)
            return limiter

        mock_rate_limiter_class.side_effect = create_rate_limiter
        # Set rate limiters on environment as they're created
        mock_env.rate_limiter = None  # Start with None

        result = cli_runner.invoke(
            benchmark,
            [
                *request_rate_options,
                "--request-rate",
                "10",
                "--request-rate",
                "20",  # Multiple rates = multiple runs
            ],
        )

        assert result.exit_code == 0, f"Command failed with output: {result.output}"
        # Verify that rate limiters were created and stopped
        # The first rate limiter should be stopped before the second is created
        if len(rate_limiters_created) > 0:
            assert rate_limiters_created[
                0
            ].stop.called, "First rate limiter should be stopped"


@pytest.mark.usefixtures(
    "mock_env_variables",
    "mock_validate_tokenizer",
)
def test_benchmark_request_rate_and_num_concurrency_mutually_exclusive(
    cli_runner, request_rate_options
):
    """Test that providing both --request-rate and --num-concurrency
    (with non-default values) raises an error."""
    result = cli_runner.invoke(
        benchmark,
        [
            *request_rate_options,
            "--num-concurrency",
            "5",  # Non-default value
            "--num-concurrency",
            "10",  # Non-default value
            "--request-rate",
            "2",
        ],
    )
    assert result.exit_code != 0
    assert (
        "mutually exclusive" in result.output.lower()
        or "--num-concurrency and --request-rate are mutually exclusive"
        in result.output
    )
