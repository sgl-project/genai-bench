from locust.runners import MasterRunner, WorkerRunner

import multiprocessing
from unittest.mock import ANY, MagicMock, patch

import pytest

from genai_bench.distributed.runner import DistributedConfig, DistributedRunner
from genai_bench.metrics.metrics import RequestLevelMetrics
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


def test_message_handlers_registration(mock_environment, mock_dashboard):
    """Test that all message handlers are registered in all modes"""
    # Test local mode
    config = DistributedConfig(num_workers=0)
    runner = DistributedRunner(mock_environment, config, dashboard=mock_dashboard)
    runner.setup()

    # Verify all handlers are registered
    assert mock_environment.runner.register_message.call_count == 3
    mock_environment.runner.register_message.assert_any_call(
        "update_scenario", runner._handle_scenario_update
    )
    mock_environment.runner.register_message.assert_any_call(
        "update_batch_size", runner._handle_batch_size_update
    )
    mock_environment.runner.register_message.assert_any_call("request_metrics", ANY)


def test_message_handlers_in_distributed_mode(mock_environment, mock_dashboard):
    """Test message handlers in distributed mode (both master and worker)"""
    config = DistributedConfig(num_workers=2)
    runner = DistributedRunner(mock_environment, config, dashboard=mock_dashboard)

    # Test worker mode
    mock_environment.runner = MagicMock(spec=WorkerRunner)
    runner._register_message_handlers()
    assert mock_environment.runner.register_message.call_count == 2

    # Test master mode
    mock_environment.runner = MagicMock(spec=MasterRunner)
    runner._register_message_handlers()
    assert mock_environment.runner.register_message.call_count == 2


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


@patch("time.sleep")
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
