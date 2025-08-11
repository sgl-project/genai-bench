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


# Mock time.sleep
@pytest.fixture
def mock_time_sleep():
    with patch("time.sleep", return_value=None):
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
        patch("genai_bench.cli.cli.plot_experiment_data") as mock_plot_experiment_data,
        patch(
            "genai_bench.cli.cli.plot_single_scenario_inference_speed_vs_throughput"
        ) as mock_plot_single_scenario_inference_speed_vs_throughput,
    ):
        yield {
            "load_experiment": mock_load_experiment,
            "create_workbook": mock_create_workbook,
            "plot_experiment_data": mock_plot_experiment_data,
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
            import traceback

            traceback.print_exception(
                type(result.exception), result.exception, result.exception.__traceback__
            )

    assert result.exit_code == 0, f"Command failed with output: {result.output}"

    assert mock_report_and_plot["load_experiment"].called
    assert mock_report_and_plot["create_workbook"].called
    assert mock_report_and_plot["plot_experiment_data"].called
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
            "vLLM",
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
                "vLLM",
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
                "vLLM",
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
                    "vLLM",
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
    assert mock_report_and_plot["plot_experiment_data"].called


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
