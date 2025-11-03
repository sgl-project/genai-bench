"""Tests for verifying request rate accuracy with rate limiting.

These tests verify that the rate limiter accurately controls request rates
by using a mock user class that records timestamps instead of making real
HTTP requests.

- Local mode tests use the actual CLI to ensure configuration matches real usage
- Distributed mode tests use direct setup (patches don't cross process boundaries)
"""

from locust import task
from locust.env import Environment
from locust.runners import MasterRunner, WorkerRunner

import contextlib
import gc
import multiprocessing
import os
import threading
import time
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from genai_bench.cli.cli import benchmark
from genai_bench.cli.utils import get_run_params
from genai_bench.distributed.runner import DistributedConfig, DistributedRunner
from genai_bench.scenarios.base import Scenario
from genai_bench.user.base_user import BaseUser

# ============================================================================
# Mock User Class for Local Mode (CLI-based tests)
# ============================================================================


class MockRateTestUser(BaseUser):
    """Mock user that records timestamps instead of sending HTTP requests."""

    host = "http://localhost:8000"
    BACKEND_NAME = "openai"
    supported_tasks = {"text-to-text": "chat"}

    _timestamps: List[float] = []
    _timestamps_lock = threading.Lock()

    @task
    def chat(self):
        """Record timestamp when we would send a request."""
        if not self.acquire_rate_limit_token():
            return

        timestamp = time.monotonic()

        with MockRateTestUser._timestamps_lock:
            MockRateTestUser._timestamps.append(timestamp)

        # Fire request event to increment Locust's request counter
        self.environment.events.request.fire(
            request_type="POST",
            name="/mock",
            response_time=0,
            response_length=0,
        )

    @classmethod
    def clear_timestamps(cls):
        with cls._timestamps_lock:
            cls._timestamps.clear()

    @classmethod
    def get_timestamps(cls) -> List[float]:
        with cls._timestamps_lock:
            return list(cls._timestamps)


# ============================================================================
# Mock User Class for Distributed Mode (direct setup tests)
# ============================================================================


class MockDistributedUser(BaseUser):
    """Mock user for distributed mode tests."""

    host = "http://localhost:8000"
    _timestamps: List[float] = []
    _timestamps_lock = threading.Lock()

    @task
    def mock_request(self):
        if not self.acquire_rate_limit_token():
            return

        timestamp = time.monotonic()

        if hasattr(self.environment, "runner") and self.environment.runner:
            if isinstance(self.environment.runner, WorkerRunner):
                self.environment.runner.send_message("mock_timestamp", timestamp)
            else:
                with MockDistributedUser._timestamps_lock:
                    MockDistributedUser._timestamps.append(timestamp)
        else:
            with MockDistributedUser._timestamps_lock:
                MockDistributedUser._timestamps.append(timestamp)

    @classmethod
    def clear_timestamps(cls):
        with cls._timestamps_lock:
            cls._timestamps.clear()

    @classmethod
    def get_timestamps(cls) -> List[float]:
        with cls._timestamps_lock:
            return list(cls._timestamps)


# ============================================================================
# Helper Functions
# ============================================================================


def analyze_timestamp_spacing(timestamps: List[float]) -> List[float]:
    """Analyze timestamp spacing to verify rate limiting."""
    if len(timestamps) < 2:
        return []

    spacings = []
    for i in range(1, len(timestamps)):
        spacing = timestamps[i] - timestamps[i - 1]
        spacings.append(spacing)

    return spacings


# ============================================================================
# Fixtures for CLI-based Local Mode Tests
# ============================================================================


@pytest.fixture
def cli_runner():
    return CliRunner()


@pytest.fixture
def mock_env_variables():
    with patch.dict("os.environ", {"HF_TOKEN": "dummy_key"}):
        yield


@pytest.fixture
def mock_dashboard():
    with patch("genai_bench.cli.cli.create_dashboard"):
        yield


@pytest.fixture
def mock_validate_tokenizer():
    with patch("genai_bench.cli.cli.validate_tokenizer", return_value=MagicMock()):
        yield


@pytest.fixture
def mock_file_system():
    with patch("genai_bench.cli.cli.Path.write_text"):
        yield


@pytest.fixture
def mock_report_and_plot():
    mock_metadata = MagicMock()
    mock_metadata.server_gpu_count = 1
    with (
        patch(
            "genai_bench.cli.cli.load_one_experiment",
            return_value=(mock_metadata, MagicMock()),
        ),
        patch("genai_bench.cli.cli.create_workbook"),
        patch("genai_bench.cli.cli.plot_experiment_data_flexible"),
        patch("genai_bench.cli.cli.plot_single_scenario_inference_speed_vs_throughput"),
    ):
        yield


@pytest.fixture
def mock_metrics_collector():
    mock_collector = MagicMock()
    mock_collector.aggregated_metrics = MagicMock()
    mock_collector.get_ui_scatter_plot_metrics.return_value = {}
    with patch(
        "genai_bench.distributed.runner.AggregatedMetricsCollector",
        return_value=mock_collector,
    ):
        yield


@pytest.fixture
def mock_experiment_path():
    with patch("genai_bench.cli.cli.get_experiment_path") as mock_path:
        mock_path.return_value = Path("/mock/experiment/path")
        yield


@pytest.fixture
def mock_user_class():
    """Patch API backend map to use MockRateTestUser."""
    MockRateTestUser.clear_timestamps()

    from genai_bench.cli.validation import API_BACKEND_USER_MAP

    patched_map = API_BACKEND_USER_MAP.copy()
    patched_map["openai"] = MockRateTestUser

    with patch("genai_bench.cli.validation.API_BACKEND_USER_MAP", patched_map):
        yield


# ============================================================================
# CLI-based Local Mode Test Helper
# ============================================================================


def run_benchmark_with_rate(
    cli_runner,
    target_rate: int,
    max_requests: int,
    max_time: int = 60,
) -> List[float]:
    """Run the benchmark CLI with a specific request rate."""
    result = cli_runner.invoke(
        benchmark,
        [
            "--api-backend",
            "openai",
            "--api-base",
            "http://localhost:8000",
            "--api-key",
            "test_key",
            "--task",
            "text-to-text",
            "--api-model-name",
            "test-model",
            "--model-tokenizer",
            "gpt2",
            "--request-rate",
            str(target_rate),
            "--max-requests-per-run",
            str(max_requests),
            "--max-time-per-run",
            str(max_time),
            "--traffic-scenario",
            "D(100,100)",
        ],
    )

    if result.exit_code != 0:
        raise RuntimeError(
            f"Benchmark failed with exit code {result.exit_code}: {result.output}"
        )

    return MockRateTestUser.get_timestamps()


# ============================================================================
# Direct Setup Distributed Mode Test Helper
# ============================================================================


def run_distributed_benchmark(
    target_rate: int,
    num_workers: int,
    num_requests: int,
    master_port: int = 5557,
) -> List[float]:
    """Run a distributed benchmark using direct setup."""
    import gevent

    MockDistributedUser.clear_timestamps()

    # Use same concurrency as CLI would use
    _, _, concurrency = get_run_params("request_rate", target_rate)

    environment = Environment(user_classes=[MockDistributedUser])
    environment.user_classes[0].tasks = [MockDistributedUser.mock_request]
    environment.scenario = Scenario.from_string("D(100,100)")
    environment.sampler = MagicMock()

    environment._mock_timestamps_list = []

    def handle_mock_timestamp(environment, msg, **kwargs):
        if hasattr(environment, "_mock_timestamps_list"):
            environment._mock_timestamps_list.append(msg.data)

    config = DistributedConfig(
        num_workers=num_workers,
        master_port=master_port,
        wait_time=3,
    )
    runner = DistributedRunner(environment=environment, config=config)
    runner.setup()

    if isinstance(environment.runner, MasterRunner):
        environment.runner.register_message("mock_timestamp", handle_mock_timestamp)

    if isinstance(environment.runner, WorkerRunner):
        return []

    per_worker_rate = target_rate / num_workers
    runner.update_rate_limiter(per_worker_rate)
    environment.rate_limiter = None

    environment.runner.start(concurrency, spawn_rate=concurrency)

    spawn_wait = max(1.5, concurrency / concurrency + 0.5)
    gevent.sleep(spawn_wait)

    start_time = time.monotonic()
    timeout = 60.0
    last_count = 0
    no_progress_timeout = 10.0
    last_progress_time = time.monotonic()

    while True:
        gevent.sleep(0.1)
        count = len(environment._mock_timestamps_list)

        if count > last_count:
            last_progress_time = time.monotonic()
            last_count = count

        if count >= num_requests:
            break

        elapsed = time.monotonic() - start_time
        if elapsed > timeout:
            break

        if time.monotonic() - last_progress_time > no_progress_timeout:
            if count > 0:
                break
            raise RuntimeError(f"No timestamps collected after {elapsed:.1f}s")

    environment.runner.stop()
    runner.update_rate_limiter(None)
    runner.wait_for_rate_limiter_stop(timeout=2.0)

    timestamps = list(environment._mock_timestamps_list)
    timestamps.sort()

    with contextlib.suppress(Exception):
        runner.cleanup()

    # Kill all remaining greenlets, otherwise pytest doesn't exit
    current = gevent.getcurrent()
    live_greenlets = [
        obj
        for obj in gc.get_objects()
        if isinstance(obj, gevent.Greenlet) and obj is not current and not obj.dead
    ]
    if live_greenlets:
        gevent.killall(live_greenlets, block=True, timeout=2.0)

    # Give time for final cleanup
    gevent.sleep(0.5)

    return timestamps


# ============================================================================
# Local Mode Tests (CLI-based)
# ============================================================================


class TestRequestRateAccuracy:
    """Test local mode rate accuracy using the actual CLI."""

    @pytest.mark.usefixtures(
        "mock_env_variables",
        "mock_dashboard",
        "mock_validate_tokenizer",
        "mock_file_system",
        "mock_report_and_plot",
        "mock_experiment_path",
        "mock_user_class",
        "mock_metrics_collector",
    )
    @pytest.mark.parametrize(
        "target_rate,num_requests",
        [
            (1, 10),
            (100, 2000),
            (400, 4000),
        ],
    )
    def test_local_mode_rate_accuracy(self, cli_runner, target_rate, num_requests):
        """Test local mode achieves correct request rate."""
        tolerance = 0.05
        expected_spacing = 1.0 / target_rate

        timestamps = run_benchmark_with_rate(
            cli_runner, target_rate=target_rate, max_requests=num_requests, max_time=20
        )

        assert (
            len(timestamps) >= num_requests
        ), f"Expected >= {num_requests} timestamps, got {len(timestamps)}"

        spacings = analyze_timestamp_spacing(timestamps)
        skip_count = min(2, len(spacings) - 1) if len(spacings) > 1 else 0
        avg_spacing = (
            sum(spacings[skip_count:]) / len(spacings[skip_count:])
            if len(spacings) > skip_count
            else sum(spacings) / len(spacings)
        )

        assert abs(avg_spacing - expected_spacing) / expected_spacing < tolerance, (
            f"Actual spacing {avg_spacing:.4f}s != expected {expected_spacing:.4f}s "
            f"(tolerance: {tolerance * 100}%)"
        )


# ============================================================================
# Distributed Mode Tests (Direct Setup)
# ============================================================================

MIN_CPUS_FOR_DISTRIBUTED = 4


@pytest.mark.skipif(
    multiprocessing.cpu_count() < MIN_CPUS_FOR_DISTRIBUTED,
    reason=f"Need {MIN_CPUS_FOR_DISTRIBUTED}+ CPUs, have {multiprocessing.cpu_count()}",
)
class TestDistributedModeRateAccuracy:
    """Test distributed mode rate accuracy using direct setup."""

    @pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
    @pytest.mark.parametrize(
        "num_workers,target_rate,port_offset",
        [
            (2, 400, 0),
            (4, 800, 1),
            (4, 1600, 2),
        ],
    )
    def test_distributed_mode_rate_accuracy(
        self, num_workers, target_rate, port_offset
    ):
        """Test distributed mode achieves correct total rate."""
        expected_total_rate = target_rate
        num_requests_per_worker = 5000
        total_requests = num_requests_per_worker * num_workers
        tolerance = 0.05  # 5% for distributed mode

        base_port = 5557 + (os.getpid() % 1000) + (port_offset * 100)

        timestamps = run_distributed_benchmark(
            target_rate=target_rate,
            num_workers=num_workers,
            num_requests=total_requests,
            master_port=base_port,
        )

        assert (
            len(timestamps) >= total_requests * 0.7
        ), f"Expected >= {total_requests * 0.7} timestamps, got {len(timestamps)}"

        if len(timestamps) >= 2:
            # Skip initial timestamps for synchronization warmup
            skip_count = max(10, int(len(timestamps) * 0.05))
            steady_state_timestamps = timestamps[skip_count:]

            if len(steady_state_timestamps) >= 2:
                total_time = steady_state_timestamps[-1] - steady_state_timestamps[0]
                actual_rate = (
                    (len(steady_state_timestamps) - 1) / total_time
                    if total_time > 0
                    else 0
                )

                assert (
                    abs(actual_rate - expected_total_rate) / expected_total_rate
                    < tolerance
                ), (
                    f"Actual rate {actual_rate:.2f} req/s != target "
                    f"{expected_total_rate:.2f} req/s (tolerance: {tolerance * 100}%)"
                )

    @pytest.mark.skip(
        reason="High-intensity test - skip in CI due to resource constraints"
    )
    @pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
    @pytest.mark.parametrize(
        "num_workers,target_rate,port_offset",
        [(2, 5000, 3), (4, 9000, 4), (8, 10000, 5), (16, 10000, 6)],
    )
    def test_distributed_mode_rate_accuracy_high_rate(
        self, num_workers, target_rate, port_offset
    ):
        """Test distributed mode achieves correct total rate for high rates."""
        expected_total_rate = target_rate
        num_requests_per_worker = 10000
        total_requests = num_requests_per_worker * num_workers
        tolerance = 0.05  # 5% for high rate distributed mode (CI variability)

        base_port = 5557 + (os.getpid() % 1000) + (port_offset * 100)

        timestamps = run_distributed_benchmark(
            target_rate=target_rate,
            num_workers=num_workers,
            num_requests=total_requests,
            master_port=base_port,
        )

        assert (
            len(timestamps) >= total_requests * 0.7
        ), f"Expected >= {total_requests * 0.7} timestamps, got {len(timestamps)}"

        if len(timestamps) >= 2:
            # Skip initial timestamps for synchronization warmup
            skip_count = max(10, int(len(timestamps) * 0.05))
            steady_state_timestamps = timestamps[skip_count:]

            if len(steady_state_timestamps) >= 2:
                total_time = steady_state_timestamps[-1] - steady_state_timestamps[0]
                actual_rate = (
                    (len(steady_state_timestamps) - 1) / total_time
                    if total_time > 0
                    else 0
                )

                assert (
                    abs(actual_rate - expected_total_rate) / expected_total_rate
                    < tolerance
                ), (
                    f"Actual rate {actual_rate:.2f} req/s != target "
                    f"{expected_total_rate:.2f} req/s (tolerance: {tolerance * 100}%)"
                )
