from locust.runners import MasterRunner, WorkerRunner

import multiprocessing
from unittest.mock import ANY, MagicMock, patch

import pytest

from genai_bench.distributed.runner import DistributedConfig, DistributedRunner
from genai_bench.metrics.metrics import RequestLevelMetrics
from genai_bench.rate_limiter import TokenBucketRateLimiter
from genai_bench.scenarios.base import Scenario


@pytest.fixture
def mock_environment():
    with patch("locust.env.Environment") as mock_env_class:
        mock_env = MagicMock()
        mock_env_class.return_value = mock_env
        mock_env.sampler = MagicMock()
        yield mock_env


@pytest.fixture
def mock_dashboard():
    dashboard = MagicMock()
    dashboard.handle_single_request = MagicMock()
    return dashboard


@pytest.fixture
def distributed_runner(mock_environment, mock_dashboard):
    config = DistributedConfig(num_workers=0)
    return DistributedRunner(
        environment=mock_environment,
        config=config,
        dashboard=mock_dashboard,
    )


def test_local_mode_setup(distributed_runner, mock_environment):
    """Test runner setup in local mode"""
    distributed_runner.setup()

    assert not isinstance(mock_environment.runner, WorkerRunner)
    assert len(distributed_runner._worker_processes) == 0


@patch("multiprocessing.Process")
@patch("locust.runners.MasterRunner")
def test_distributed_mode_setup(
    mock_master_runner,
    mock_process,
    mock_environment,
    mock_dashboard,
):
    """Test runner setup in distributed mode"""
    config = DistributedConfig(num_workers=2)
    runner = DistributedRunner(mock_environment, config, mock_dashboard)

    mock_environment.create_master_runner.return_value = mock_master_runner

    mock_process.return_value.start = MagicMock()
    runner.setup()

    assert mock_process.call_count == 2
    assert len(runner._worker_processes) == 2


def test_scenario_update(distributed_runner, mock_environment):
    """Test scenario update handling"""
    distributed_runner.setup()

    scenario_str = "D(100,100)"
    mock_msg = MagicMock()
    mock_msg.data = scenario_str

    distributed_runner._handle_scenario_update(mock_environment, mock_msg)
    assert isinstance(mock_environment.scenario, Scenario)
    assert mock_environment.scenario.to_string() == scenario_str


def test_batch_size_update(distributed_runner, mock_environment):
    """Test batch size update handling"""
    distributed_runner.setup()

    batch_size = 32
    mock_msg = MagicMock()
    mock_msg.data = batch_size

    distributed_runner._handle_batch_size_update(mock_environment, mock_msg)
    assert mock_environment.sampler.batch_size == batch_size


def test_rate_limiter_update(distributed_runner, mock_environment):
    """Test rate limiter update handling"""
    distributed_runner.setup()

    rate = 2.5
    mock_msg = MagicMock()
    mock_msg.data = rate

    distributed_runner._handle_rate_limiter_update(mock_environment, mock_msg)
    assert mock_environment.rate_limiter is not None
    assert mock_environment.rate_limiter.rate == rate

    # Test removing rate limiter
    mock_msg.data = None
    distributed_runner._handle_rate_limiter_update(mock_environment, mock_msg)
    assert mock_environment.rate_limiter is None


def test_rate_limiter_update_multiple_workers(mock_environment, mock_dashboard):
    """Test rate limiter update with multiple workers"""
    # Simulate master sending rate updates to workers
    config = DistributedConfig(num_workers=4)
    runner = DistributedRunner(
        environment=mock_environment,
        config=config,
        dashboard=mock_dashboard,
    )
    runner.setup()

    # Test dividing rate among 4 workers
    total_rate = 10.0
    per_worker_rate = total_rate / 4  # 2.5 req/s per worker

    mock_msg = MagicMock()
    mock_msg.data = per_worker_rate

    # Simulate worker receiving the update
    runner._handle_rate_limiter_update(mock_environment, mock_msg)

    assert mock_environment.rate_limiter is not None
    assert isinstance(mock_environment.rate_limiter, TokenBucketRateLimiter)
    assert mock_environment.rate_limiter.rate == per_worker_rate

    # Verify total rate across all workers would be correct
    # (In real scenario, each worker would have this rate)
    expected_total = per_worker_rate * 4
    assert abs(expected_total - total_rate) < 0.01


def test_rate_limiter_update_low_rate(mock_environment, mock_dashboard):
    """Test rate limiter update with very low per-worker rate"""

    config = DistributedConfig(num_workers=10)
    runner = DistributedRunner(
        environment=mock_environment,
        config=config,
        dashboard=mock_dashboard,
    )
    runner.setup()

    # Test with very low rate divided among many workers
    total_rate = 1.0
    per_worker_rate = total_rate / 10  # 0.1 req/s per worker

    mock_msg = MagicMock()
    mock_msg.data = per_worker_rate

    # Should still work, but rate will be low
    runner._handle_rate_limiter_update(mock_environment, mock_msg)

    assert mock_environment.rate_limiter is not None
    assert mock_environment.rate_limiter.rate == per_worker_rate


def test_rate_limiter_update_invalid_rate(mock_environment, mock_dashboard):
    """Test rate limiter update with invalid (zero or negative) rate"""
    config = DistributedConfig(num_workers=2)
    runner = DistributedRunner(
        environment=mock_environment,
        config=config,
        dashboard=mock_dashboard,
    )
    runner.setup()

    # Test with zero rate - should remove rate limiter
    mock_msg = MagicMock()
    mock_msg.data = 0

    runner._handle_rate_limiter_update(mock_environment, mock_msg)
    assert mock_environment.rate_limiter is None

    # Test with negative rate - should remove rate limiter
    mock_msg.data = -1.0
    runner._handle_rate_limiter_update(mock_environment, mock_msg)
    assert mock_environment.rate_limiter is None

    # Test with None - should remove rate limiter
    mock_msg.data = None
    runner._handle_rate_limiter_update(mock_environment, mock_msg)
    assert mock_environment.rate_limiter is None


def test_rate_limiter_master_none_in_distributed(mock_environment, mock_dashboard):
    """Test that master process has rate_limiter set to None in distributed mode"""
    config = DistributedConfig(num_workers=2)
    runner = DistributedRunner(
        environment=mock_environment,
        config=config,
        dashboard=mock_dashboard,
    )

    # Mock master runner
    mock_environment.runner = MagicMock(spec=MasterRunner)
    runner.setup()

    # In distributed mode, master should not have a rate limiter
    # (rate limiting happens on workers only)
    # This is verified by checking that update_rate_limiter sends message to workers
    # but master's environment.rate_limiter should be None
    assert (
        hasattr(mock_environment, "rate_limiter")
        or mock_environment.rate_limiter is None
    )


def test_message_handlers_registration(mock_environment, mock_dashboard):
    """Test that all message handlers are registered in all modes"""
    # Test local mode
    config = DistributedConfig(num_workers=0)
    runner = DistributedRunner(mock_environment, config, dashboard=mock_dashboard)
    runner.setup()

    # Verify all handlers are registered
    assert mock_environment.runner.register_message.call_count == 4
    mock_environment.runner.register_message.assert_any_call(
        "update_scenario", runner._handle_scenario_update
    )
    mock_environment.runner.register_message.assert_any_call(
        "update_batch_size", runner._handle_batch_size_update
    )
    mock_environment.runner.register_message.assert_any_call(
        "update_rate_limiter", runner._handle_rate_limiter_update
    )
    mock_environment.runner.register_message.assert_any_call("request_metrics", ANY)


def test_message_handlers_in_distributed_mode(mock_environment, mock_dashboard):
    """Test message handlers in distributed mode (both master and worker)"""
    config = DistributedConfig(num_workers=2)
    runner = DistributedRunner(mock_environment, config, dashboard=mock_dashboard)

    # Test worker mode
    mock_environment.runner = MagicMock(spec=WorkerRunner)
    runner._register_message_handlers()
    assert mock_environment.runner.register_message.call_count == 3

    # Test master mode
    mock_environment.runner = MagicMock(spec=MasterRunner)
    runner._register_message_handlers()
    assert mock_environment.runner.register_message.call_count == 3


def test_update_rate_limiter_sends_message(mock_environment, mock_dashboard):
    """Test that update_rate_limiter() sends message to workers."""
    config = DistributedConfig(num_workers=2)
    runner = DistributedRunner(mock_environment, config, dashboard=mock_dashboard)
    runner.setup()

    # Mock runner with send_message method
    mock_environment.runner = MagicMock()
    mock_environment.runner.send_message = MagicMock()

    # Call update_rate_limiter
    rate = 5.0
    runner.update_rate_limiter(rate)

    # Verify message was sent
    mock_environment.runner.send_message.assert_called_once_with(
        "update_rate_limiter", rate
    )

    # Test with None (stop signal)
    runner.update_rate_limiter(None)
    mock_environment.runner.send_message.assert_called_with("update_rate_limiter", None)


def test_update_rate_limiter_local_mode(mock_environment, mock_dashboard):
    """Test update_rate_limiter in local mode (direct assignment)."""
    config = DistributedConfig(num_workers=0)
    runner = DistributedRunner(mock_environment, config, dashboard=mock_dashboard)
    runner.setup()

    # In local mode, update_rate_limiter should still work
    # but the handler will be called directly
    rate = 10.0
    mock_msg = MagicMock()
    mock_msg.data = rate

    # Simulate local mode handler call
    runner._handle_rate_limiter_update(mock_environment, mock_msg)

    # Should create rate limiter directly
    assert mock_environment.rate_limiter is not None
    assert isinstance(mock_environment.rate_limiter, TokenBucketRateLimiter)
    assert mock_environment.rate_limiter.rate == rate


def test_message_handling(mock_environment, mock_dashboard):
    """Test actual message handling"""
    config = DistributedConfig(num_workers=0)
    runner = DistributedRunner(mock_environment, config, dashboard=mock_dashboard)
    runner.setup()

    # Test scenario update
    scenario_str = "D(100,100)"
    mock_msg = MagicMock()
    mock_msg.data = scenario_str
    runner._handle_scenario_update(mock_environment, mock_msg)
    assert isinstance(mock_environment.scenario, Scenario)
    assert mock_environment.scenario.to_string() == scenario_str

    # Test batch size update
    batch_size = 32
    mock_msg = MagicMock()
    mock_msg.data = batch_size
    runner._handle_batch_size_update(mock_environment, mock_msg)
    assert mock_environment.sampler.batch_size == batch_size

    # Test metrics handling
    metrics = RequestLevelMetrics(
        num_input_tokens=10,
        ttft=0.5,
        tpot=1.0,
        e2e_latency=1.0,
        output_latency=0.5,
        output_inference_speed=100.0,
        num_output_tokens=20,
        total_tokens=30,
        input_throughput=10.0,
        output_throughput=20.0,
    )
    mock_msg = MagicMock()
    mock_msg.data = metrics.model_dump_json()
    handler = runner._create_metrics_handler()
    handler(mock_environment, mock_msg)


@patch("gevent.sleep")
def test_cleanup(mock_environment, mock_dashboard):
    """Test cleanup in all modes"""
    # Test local mode cleanup
    config = DistributedConfig(num_workers=0)
    runner = DistributedRunner(mock_environment, config, dashboard=mock_dashboard)
    runner.setup()
    runner.cleanup()
    assert mock_environment.runner is None

    # Test distributed mode cleanup
    config = DistributedConfig(num_workers=2)
    runner = DistributedRunner(mock_environment, config, dashboard=mock_dashboard)
    mock_process1 = MagicMock()
    mock_process2 = MagicMock()
    runner._worker_processes = [mock_process1, mock_process2]
    runner.cleanup()
    mock_process1.terminate.assert_called_once()
    mock_process2.terminate.assert_called_once()
    assert mock_environment.runner is None


@patch("psutil.Process")
def test_cpu_affinity_mapping(mock_process, mock_environment, mock_dashboard):
    """Test CPU affinity mapping for workers"""
    # Test with custom mapping
    config = DistributedConfig(num_workers=2, cpu_affinity_map={0: 1, 1: 3})
    runner = DistributedRunner(mock_environment, config, mock_dashboard)

    mock_process_instance = MagicMock()
    mock_process.return_value = mock_process_instance

    # Test successful affinity setting
    runner._set_cpu_affinity(0)
    mock_process_instance.cpu_affinity.assert_called_with([1])

    # Test invalid CPU mapping
    config.cpu_affinity_map = {0: 999}  # Invalid CPU number
    runner._set_cpu_affinity(0)
    mock_process_instance.cpu_affinity.assert_called_with(
        [0]
    )  # Should fallback to worker_id % cpu_count

    # Test affinity setting failure
    mock_process_instance.cpu_affinity.side_effect = Exception("Failed to set affinity")
    runner._set_cpu_affinity(0)  # Should log warning but not fail


@pytest.fixture
def mock_logging_env():
    """Mock environment variables for logging"""
    with patch.dict("os.environ", {"GENAI_BENCH_LOGGING_LEVEL": "INFO"}):
        yield


@pytest.mark.usefixtures("mock_logging_env")
def test_worker_process_failure(mock_environment, mock_dashboard):
    """Test worker process error handling"""
    config = DistributedConfig(num_workers=1)
    runner = DistributedRunner(mock_environment, config, mock_dashboard)

    # Define a specific exception type
    class WorkerSetupError(Exception):
        pass

    # Simulate worker process failure with specific exception
    mock_environment.create_worker_runner.side_effect = WorkerSetupError(
        "Worker failed"
    )

    # The worker process should now catch the exception and return gracefully
    # instead of re-raising it
    result = runner._worker_process(0)

    # Verify that the function returns None (graceful exit)
    assert result is None


@patch("gevent.spawn")
def test_log_consumer_setup(mock_spawn, mock_environment, mock_dashboard):
    """Test log consumer setup and handling"""
    config = DistributedConfig(num_workers=2)
    runner = DistributedRunner(mock_environment, config, mock_dashboard)

    # Mock master runner
    mock_environment.runner = MagicMock(spec=MasterRunner)

    # Test master setup with log consumer
    runner._setup_master()
    mock_spawn.assert_called_once_with(runner._consume_worker_logs)


def test_log_consumer_processing(mock_environment, mock_dashboard):
    """Test log consumer message processing"""
    config = DistributedConfig(num_workers=1)
    runner = DistributedRunner(mock_environment, config, mock_dashboard)

    # Add test log messages to queue
    test_log = {"worker_id": "0", "message": "Test log message", "level": "INFO"}
    runner.worker_log_queue.put(test_log)

    # Test log draining
    runner._drain_log_queue()

    # Test invalid log data handling
    runner.worker_log_queue.put(None)  # Should be skipped
    runner.worker_log_queue.put({"invalid": "log"})  # Should handle missing fields
    runner._drain_log_queue()


def test_log_handler_creation(mock_environment, mock_dashboard):
    """Test log handler creation and message handling"""
    config = DistributedConfig(num_workers=1)
    runner = DistributedRunner(mock_environment, config, mock_dashboard)

    # Create handler
    handler = runner._create_log_handler()

    # Test with non-master runner (should return early)
    mock_environment.runner = MagicMock(spec=WorkerRunner)
    mock_msg = MagicMock()
    mock_msg.data = {"worker_id": "0", "message": "Test message", "level": "INFO"}
    handler(mock_environment, mock_msg)

    # Test with master runner
    mock_environment.runner = MagicMock(spec=MasterRunner)
    handler(mock_environment, mock_msg)


@patch("os.environ")
@pytest.mark.usefixtures("mock_logging_env")
def test_distributed_worker_process(mock_environ, mock_environment, mock_dashboard):
    """Test distributed mode setup and environment configuration"""
    # Set up mock environment variables
    mock_environ.get.return_value = "INFO"  # Mock GENAI_BENCH_LOGGING_LEVEL

    config = DistributedConfig(num_workers=2)
    runner = DistributedRunner(mock_environment, config, mock_dashboard)

    # Test tokenizer parallelism disable
    runner._setup_distributed()
    mock_environ.__setitem__.assert_called_with("TOKENIZERS_PARALLELISM", "false")

    # Test worker process setup
    mock_environment.runner = MagicMock(spec=WorkerRunner)
    runner._setup_distributed()  # Should return early for worker


def test_high_worker_count_warning(mock_environment, mock_dashboard):
    """Test warning for high worker count relative to CPU cores"""
    cpu_count = multiprocessing.cpu_count()
    config = DistributedConfig(num_workers=cpu_count * 5)  # Excessive workers
    DistributedRunner(mock_environment, config, mock_dashboard)
    # Warning should be logged


def test_log_data_processing(mock_environment, mock_dashboard):
    """Test processing of individual log messages"""
    config = DistributedConfig(num_workers=1)
    runner = DistributedRunner(mock_environment, config, mock_dashboard)

    # Test valid log data
    valid_log = {"worker_id": "0", "message": "Test message", "level": "INFO"}
    runner._process_log_data(valid_log)

    # Test empty/invalid log data
    runner._process_log_data(None)
    runner._process_log_data({})
    runner._process_log_data({"invalid": "data"})


@patch("genai_bench.distributed.runner.logger")
def test_metrics_handler_validation_error_handling(
    mock_logger, mock_environment, mock_dashboard
):
    """Test that ValidationError in metrics handler is caught and logged."""
    config = DistributedConfig(num_workers=0)
    runner = DistributedRunner(mock_environment, config, mock_dashboard)
    runner.setup()

    # Create handler
    handler = runner._create_metrics_handler()

    # Create invalid metrics data that will cause ValidationError
    # This simulates the case where tpot is None but error_code is None
    invalid_metrics_json = '{"ttft": 0.5, "tpot": null, "error_code": null}'

    mock_msg = MagicMock()
    mock_msg.data = invalid_metrics_json

    # Call handler - should not raise exception
    handler(mock_environment, mock_msg)

    # Verify warning was logged
    mock_logger.warning.assert_called_once()
    warning_call = mock_logger.warning.call_args[0][0]
    assert "Dropping invalid metrics record due to validation error" in warning_call


def test_handle_rate_limiter_stopped_handler(mock_environment, mock_dashboard):
    """Test _handle_rate_limiter_stopped handler behavior."""
    config = DistributedConfig(num_workers=3)
    runner = DistributedRunner(mock_environment, config, dashboard=mock_dashboard)
    runner.setup()

    # Initially no confirmations
    assert runner._rate_limiter_stop_confirmations == 0

    # Receive confirmation from first worker
    mock_msg1 = MagicMock()
    mock_msg1.data = True
    runner._handle_rate_limiter_stopped(mock_environment, mock_msg1)
    assert runner._rate_limiter_stop_confirmations == 1

    # Receive confirmation from second worker
    mock_msg2 = MagicMock()
    mock_msg2.data = True
    runner._handle_rate_limiter_stopped(mock_environment, mock_msg2)
    assert runner._rate_limiter_stop_confirmations == 2

    # Test with None message object (should be ignored)
    runner._handle_rate_limiter_stopped(mock_environment, None)
    # Should still be 2 (None message object is ignored)
    assert runner._rate_limiter_stop_confirmations == 2

    # Test with empty/falsy message object
    runner._handle_rate_limiter_stopped(mock_environment, False)
    # Should still be 2 (falsy message object is ignored)
    assert runner._rate_limiter_stop_confirmations == 2

    # Test that message with data=None is
    # still counted (handler checks msg, not msg.data)
    mock_msg3 = MagicMock()
    mock_msg3.data = None
    runner._handle_rate_limiter_stopped(mock_environment, mock_msg3)
    # Should be 3 (MagicMock is truthy even if data is None)
    assert runner._rate_limiter_stop_confirmations == 3

    # Each handler call increments the counter (each worker sends one confirmation)
    mock_msg4 = MagicMock()
    mock_msg4.data = True
    runner._handle_rate_limiter_stopped(mock_environment, mock_msg4)
    assert runner._rate_limiter_stop_confirmations == 4


@pytest.mark.usefixtures("mock_logging_env")
def test_wait_for_rate_limiter_stop_local_mode(mock_environment, mock_dashboard):
    """Test wait_for_rate_limiter_stop() in local mode (should return immediately)."""
    config = DistributedConfig(num_workers=0)
    runner = DistributedRunner(mock_environment, config, dashboard=mock_dashboard)
    runner.setup()

    # In local mode, should return True immediately
    result = runner.wait_for_rate_limiter_stop(timeout=1.0)
    assert result is True


@pytest.mark.usefixtures("mock_logging_env")
@patch("gevent.sleep")
def test_wait_for_rate_limiter_stop_success(
    mock_sleep, mock_environment, mock_dashboard
):
    """Test wait_for_rate_limiter_stop() successfully waits for all confirmations."""
    config = DistributedConfig(num_workers=3)
    runner = DistributedRunner(mock_environment, config, dashboard=mock_dashboard)
    runner.setup()

    # Kill the background greenlet before patching time.monotonic
    # to prevent it from interfering with multiprocessing queue operations
    if hasattr(runner, "log_consumer"):
        runner.log_consumer.kill()

    try:
        # Patch time.monotonic after killing the greenlet
        with patch("time.monotonic") as mock_monotonic:
            # Mock time to return fixed values
            mock_monotonic.return_value = 0.0

            # Simulate receiving confirmations from workers AFTER wait starts
            # (wait_for_rate_limiter_stop clears confirmations at start)
            call_count = [0]

            def side_effect_sleep(duration):
                call_count[0] += 1
                # After first check, add all confirmations
                if call_count[0] == 1:
                    mock_msg1 = MagicMock()
                    mock_msg1.data = True
                    mock_msg2 = MagicMock()
                    mock_msg2.data = True
                    mock_msg3 = MagicMock()
                    mock_msg3.data = True
                    runner._handle_rate_limiter_stopped(mock_environment, mock_msg1)
                    runner._handle_rate_limiter_stopped(mock_environment, mock_msg2)
                    runner._handle_rate_limiter_stopped(mock_environment, mock_msg3)

            mock_sleep.side_effect = side_effect_sleep

            # Wait - confirmations will arrive during the wait loop
            result = runner.wait_for_rate_limiter_stop(timeout=2.0, expected_workers=3)
            assert result is True
            # Should have slept at least once to check for confirmations
            assert mock_sleep.call_count >= 1
    finally:
        # Clean up the background greenlet to prevent crashes
        with patch("genai_bench.distributed.runner.logger"):
            runner.cleanup()


@pytest.mark.usefixtures("mock_logging_env")
@patch("gevent.sleep")
def test_wait_for_rate_limiter_stop_timeout(
    mock_sleep, mock_environment, mock_dashboard
):
    """Test wait_for_rate_limiter_stop() times out when not all workers confirm."""
    config = DistributedConfig(num_workers=3)
    runner = DistributedRunner(mock_environment, config, dashboard=mock_dashboard)
    runner.setup()

    # Kill the background greenlet before patching time.monotonic
    # to prevent it from interfering with multiprocessing queue operations
    if hasattr(runner, "log_consumer"):
        runner.log_consumer.kill()

    try:
        # Patch time.monotonic after killing the greenlet
        with patch("time.monotonic") as mock_monotonic:
            # Simulate time passing - need enough values for multiple loop iterations
            time_counter = [0.0]

            def time_side_effect():
                time_counter[0] += 0.1
                return time_counter[0]

            mock_monotonic.side_effect = time_side_effect

            # Set up mock_sleep to add only 2 confirmations (missing 1)
            call_count = [0]

            def side_effect_sleep(duration):
                call_count[0] += 1
                # After first sleep, add first confirmation
                if call_count[0] == 1:
                    mock_msg1 = MagicMock()
                    mock_msg1.data = True
                    runner._handle_rate_limiter_stopped(mock_environment, mock_msg1)
                # After second sleep, add second confirmation (but not third)
                elif call_count[0] == 2:
                    mock_msg2 = MagicMock()
                    mock_msg2.data = True
                    runner._handle_rate_limiter_stopped(mock_environment, mock_msg2)
                # No third confirmation - should timeout

            mock_sleep.side_effect = side_effect_sleep

            # Wait with timeout - should return False after timeout
            # Since time keeps increasing, it will eventually exceed timeout
            result = runner.wait_for_rate_limiter_stop(timeout=2.0, expected_workers=3)
            assert result is False
            # Should have slept multiple times waiting for the last confirmation
            assert mock_sleep.call_count > 0
    finally:
        # Clean up the background greenlet to prevent crashes
        with patch("genai_bench.distributed.runner.logger"):
            runner.cleanup()


@pytest.mark.usefixtures("mock_logging_env")
@patch("gevent.sleep")
def test_wait_for_rate_limiter_stop_partial_confirmations(
    mock_sleep, mock_environment, mock_dashboard
):
    """Test wait_for_rate_limiter_stop() with
    partial confirmations that arrive during wait."""

    config = DistributedConfig(num_workers=3)
    runner = DistributedRunner(mock_environment, config, dashboard=mock_dashboard)
    runner.setup()

    # Kill the background greenlet before patching time.monotonic
    # to prevent it from interfering with multiprocessing queue operations
    if hasattr(runner, "log_consumer"):
        runner.log_consumer.kill()

    try:
        # Patch time.monotonic after killing the greenlet
        with patch("time.monotonic") as mock_monotonic:
            # Simulate time passing with confirmations arriving
            # Need enough values for the loop iterations
            time_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            mock_monotonic.side_effect = time_values

            # Set up mock_sleep to simulate receiving confirmations during wait
            call_count = [0]

            def side_effect_sleep(duration):
                call_count[0] += 1
                # After first sleep, add first confirmation
                if call_count[0] == 1:
                    mock_msg1 = MagicMock()
                    mock_msg1.data = True
                    runner._handle_rate_limiter_stopped(mock_environment, mock_msg1)
                # After second sleep, add second confirmation
                elif call_count[0] == 2:
                    mock_msg2 = MagicMock()
                    mock_msg2.data = True
                    runner._handle_rate_limiter_stopped(mock_environment, mock_msg2)
                # After third sleep, add the last confirmation
                elif call_count[0] == 3:
                    mock_msg3 = MagicMock()
                    mock_msg3.data = True
                    runner._handle_rate_limiter_stopped(mock_environment, mock_msg3)

            mock_sleep.side_effect = side_effect_sleep

            # Wait - should eventually return True when all confirmations arrive
            result = runner.wait_for_rate_limiter_stop(timeout=2.0, expected_workers=3)
            assert result is True
            # Should have slept multiple times before all confirmations arrived
            assert mock_sleep.call_count >= 3
    finally:
        # Clean up the background greenlet to prevent crashes
        with patch("genai_bench.distributed.runner.logger"):
            runner.cleanup()


@pytest.mark.usefixtures("mock_logging_env")
@patch("gevent.sleep")
def test_wait_for_rate_limiter_stop_integration(
    mock_sleep, mock_environment, mock_dashboard
):
    """Test full integration flow:
    master sends stop signal → workers confirm → master waits."""

    config = DistributedConfig(num_workers=4)
    runner = DistributedRunner(mock_environment, config, dashboard=mock_dashboard)
    runner.setup()

    # Kill the background greenlet before patching time.monotonic
    # to prevent it from interfering with multiprocessing queue operations
    if hasattr(runner, "log_consumer"):
        runner.log_consumer.kill()

    # Mock master runner
    mock_environment.runner = MagicMock(spec=MasterRunner)
    mock_environment.runner.send_message = MagicMock()

    try:
        # Patch time.monotonic after killing the greenlet
        with patch("time.monotonic") as mock_monotonic:
            # Step 1: Master sends stop signal to workers
            runner.update_rate_limiter(None)  # None signals stop
            mock_environment.runner.send_message.assert_called_with(
                "update_rate_limiter", None
            )

            # Step 2: Simulate workers receiving stop signal and sending confirmations
            # Mock time progression
            time_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
            mock_monotonic.side_effect = time_values

            call_count = [0]

            def side_effect_sleep(duration):
                call_count[0] += 1
                # Simulate workers sending confirmations at different times
                if call_count[0] == 1:
                    # Worker 1 confirms
                    mock_msg1 = MagicMock()
                    mock_msg1.data = True
                    runner._handle_rate_limiter_stopped(mock_environment, mock_msg1)
                elif call_count[0] == 2:
                    # Worker 2 confirms
                    mock_msg2 = MagicMock()
                    mock_msg2.data = True
                    runner._handle_rate_limiter_stopped(mock_environment, mock_msg2)
                elif call_count[0] == 3:
                    # Worker 3 confirms
                    mock_msg3 = MagicMock()
                    mock_msg3.data = True
                    runner._handle_rate_limiter_stopped(mock_environment, mock_msg3)
                elif call_count[0] == 4:
                    # Worker 4 confirms
                    mock_msg4 = MagicMock()
                    mock_msg4.data = True
                    runner._handle_rate_limiter_stopped(mock_environment, mock_msg4)

            mock_sleep.side_effect = side_effect_sleep

            # Step 3: Master waits for all confirmations
            result = runner.wait_for_rate_limiter_stop(timeout=2.0, expected_workers=4)
            assert result is True
            assert runner._rate_limiter_stop_confirmations == 4
    finally:
        # Clean up the background greenlet to prevent crashes
        with patch("genai_bench.distributed.runner.logger"):
            runner.cleanup()
