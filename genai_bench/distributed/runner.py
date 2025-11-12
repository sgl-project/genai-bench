from locust.env import Environment
from locust.runners import MasterRunner, WorkerRunner

import atexit
import logging
import multiprocessing
import os
import time
from dataclasses import dataclass
from multiprocessing.queues import Queue
from typing import Any, Dict, List, Optional, Protocol

import gevent
import psutil
from pydantic import ValidationError

from genai_bench.logging import WorkerLoggingManager, init_logger
from genai_bench.metrics.aggregated_metrics_collector import AggregatedMetricsCollector
from genai_bench.metrics.metrics import RequestLevelMetrics
from genai_bench.scenarios.base import Scenario
from genai_bench.ui.dashboard import Dashboard

logger = init_logger(__name__)


@dataclass
class DistributedConfig:
    """Configuration for distributed runners"""

    num_workers: int
    master_host: str = "127.0.0.1"
    master_port: int = 5557
    wait_time: int = 2
    log_dir: Optional[str] = None

    # Experimental:
    # CPU pinning is not supported on all platforms, so we disable it by default
    # If you want to enable it, you may need to set the cpu_affinity_map
    pin_to_cores: bool = False
    cpu_affinity_map: Optional[Dict[int, int]] = None  # Custom worker->CPU mapping

    def __post_init__(self):
        cpu_count = multiprocessing.cpu_count()
        if self.num_workers > cpu_count * 4:
            logger.warning(
                f"Number of workers ({self.num_workers}) is much higher than "
                f"available CPU cores ({cpu_count}). This might impact performance."
            )


class MessageHandler(Protocol):
    """Protocol for message handlers"""

    def __call__(self, environment: Environment, msg: Any, **kwargs) -> None: ...


class DistributedRunner:
    """Manages distributed load test execution with master and worker processes.

    This class handles the distributed architecture of the benchmark system:

    1. Process Model:
       - Master process: Controls test execution and aggregates metrics
       - Worker processes: Execute actual API requests and send metrics to master
       - Local mode: Single process handles both execution and aggregation

    2. Message Flow:
       - Master → Workers:
           * "update_scenario": Updates test scenario configuration
           * "update_batch_size": Updates batch size for requests
       - Workers → Master:
           * "request_metrics": Sends metrics from each request for aggregation
           * "worker_log": Sends worker logs to master

    3. Execution Flow:
       - Master process:
           * Sets up worker processes
           * Controls test scenarios and batch sizes
           * Aggregates metrics from workers
           * Runs the main benchmark loop
           * Updates dashboard with live metrics
       - Worker processes:
           * Receive test configurations from master
           * Execute API requests
           * Send metrics back to master
           * Do NOT execute the main benchmark loop

    4. Message Registration:
       - Each process registers only the handlers it needs:
           * Master: registers "request_metrics" handler
           * Workers: register "update_scenario", "update_batch_size" handlers
           * Local mode: registers all handlers

    5. Metrics Collection:
       - Only master/local maintains AggregatedMetricsCollector
       - Workers collect individual request metrics and send to master
       - Master aggregates metrics and updates dashboard

    Args:
        environment: Locust Environment instance for test execution
        config: Configuration for distributed setup
        dashboard: Optional dashboard for live metrics display

    Example:
        ```python
        config = DistributedConfig(num_workers=2)
        runner = DistributedRunner(environment, config, dashboard)
        runner.setup()

        # If worker, returns after setup
        if isinstance(environment.runner, WorkerRunner):
            return

        # Master continues with test execution
        runner.update_scenario("D(100,100)")
        runner.update_batch_size(32)
        ```
    """

    def __init__(
        self,
        environment: Environment,
        config: DistributedConfig,
        dashboard: Optional[Dashboard] = None,
    ):
        self.environment = environment
        self.config = config
        self.dashboard = dashboard
        self._worker_processes: List[multiprocessing.Process] = []
        self.metrics_collector: Optional[AggregatedMetricsCollector] = None
        self.worker_log_queue: Queue = multiprocessing.Queue()

    def setup(self) -> None:
        """Set up distributed or local test environment"""
        if self.config.num_workers > 0:
            self._setup_distributed()
        else:
            self._setup_local()

    def _setup_distributed(self) -> None:
        """Set up distributed mode with master and workers"""
        # Disable HuggingFace Transformers' tokenizer parallelism in Rust
        # to prevent conflicts with Python multiprocessing.
        # For more details, see:
        # https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning  # noqa: E501
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Register cleanup on program exit
        atexit.register(self.cleanup)

        self._worker_processes = self._create_workers()

        # If this is a worker process, exit after setup
        if isinstance(self.environment.runner, WorkerRunner):
            return

        self._setup_master()

    def _setup_master(self) -> None:
        """Set up master node"""
        self.environment.create_master_runner(
            master_bind_host=self.config.master_host,
            master_bind_port=self.config.master_port,
        )
        # Create collector only for master in distributed mode
        self.metrics_collector = AggregatedMetricsCollector()

        time.sleep(self.config.wait_time)
        self._register_message_handlers()

        # Start log consumer greenlet
        self.log_consumer = gevent.spawn(self._consume_worker_logs)

    def _process_log_data(self, log_data: dict) -> None:
        """Process a single log message from the queue"""
        if not log_data:
            return

        worker_id = log_data.get("worker_id")
        log_message = log_data.get("message")
        log_level = log_data.get("level", "INFO")

        # Log through master's logger
        logger.log(
            getattr(logging, log_level),
            f"[Worker {worker_id}] {log_message}",
        )

    def _consume_worker_logs(self):
        """Consume logs from worker processes"""
        try:
            while True:
                if self.worker_log_queue.empty():
                    gevent.sleep(0.1)
                    continue

                log_data = self.worker_log_queue.get_nowait()
                self._process_log_data(log_data)

        except KeyboardInterrupt:
            logger.info("Log consumer shutting down")
            self._drain_log_queue()

    def _drain_log_queue(self):
        """Drain remaining logs from the queue"""
        while not self.worker_log_queue.empty():
            log_data = self.worker_log_queue.get_nowait()
            self._process_log_data(log_data)

    def _setup_local(self) -> None:
        """Set up local mode"""
        self.environment.create_local_runner()
        # Create collector for local
        self.metrics_collector = AggregatedMetricsCollector()
        self._register_message_handlers()

    def _create_workers(self) -> List[multiprocessing.Process]:
        """Create worker processes"""
        workers = []
        for i in range(self.config.num_workers):
            process = multiprocessing.Process(target=self._worker_process, args=(i,))
            workers.append(process)
            process.start()
        return workers

    def _worker_process(self, worker_id: int) -> None:
        """Worker process function with CPU affinity"""
        try:
            WorkerLoggingManager(str(worker_id), self.worker_log_queue, self.config.log_dir)

            if self.config.pin_to_cores:
                self._set_cpu_affinity(worker_id)

            runner = self.environment.create_worker_runner(
                master_host=self.config.master_host, master_port=self.config.master_port
            )
            self._register_message_handlers()

            # Add periodic health check logging
            logger.info(
                f"Worker {worker_id} started successfully and connected to master"
            )

            runner.greenlet.join()
        except Exception as e:
            logger.error(f"Worker {worker_id} failed: {str(e)}")
            # Don't raise here to prevent worker restart loops
            return

    def _set_cpu_affinity(self, worker_id: int) -> None:
        """Set CPU affinity for worker process"""
        # NOTE: only works on Linux
        process = psutil.Process()
        cpu_count = multiprocessing.cpu_count()

        if self.config.cpu_affinity_map:
            # Use custom mapping if provided
            target_cpu = self.config.cpu_affinity_map.get(worker_id)
            if target_cpu is None or target_cpu >= cpu_count:
                logger.warning(f"Invalid CPU mapping for worker {worker_id}")
                target_cpu = worker_id % cpu_count
        else:
            # Round-robin assignment
            target_cpu = worker_id % cpu_count

        try:
            process.cpu_affinity([target_cpu])
            logger.info(
                f"Worker {worker_id} pinned to CPU {target_cpu} (PID: {process.pid})"
            )
        except Exception as e:
            logger.warning(
                f"Failed to set CPU affinity for worker {worker_id}: {str(e)}"
            )

    def _create_metrics_handler(self) -> MessageHandler:
        """Create handler for metrics messages"""

        def handler(environment: Environment, msg: Any, **kwargs) -> None:
            # Master receives and aggregates metrics
            try:
                metrics = RequestLevelMetrics.model_validate_json(msg.data)
            except ValidationError as e:
                logger.warning(
                    f"Dropping invalid metrics record due to validation error: {e}"
                )
                return

            if not self.metrics_collector:
                return

            self.metrics_collector.add_single_request_metrics(metrics)

            # Update dashboard if needed
            if self.dashboard and environment.runner and environment.runner.stats:
                self.dashboard.handle_single_request(
                    self.metrics_collector.get_live_metrics(),
                    environment.runner.stats.total.num_requests,
                    metrics.error_code,
                )

        return handler

    def _create_log_handler(self) -> MessageHandler:
        """Create handler for log messages from workers"""

        def handler(environment: Environment, msg: Any, **kwargs) -> None:
            if not isinstance(environment.runner, MasterRunner):
                return

            # Parse the log message
            log_data = msg.data
            worker_id = log_data.get("worker_id")
            log_message = log_data.get("message")
            log_level = log_data.get("level", "INFO")

            # Log through master's logger
            logger.log(
                getattr(logging, log_level), f"[Worker {worker_id}] {log_message}"
            )

        return handler

    def _handle_scenario_update(self, environment: Environment, msg: Any) -> None:
        """Handle scenario update messages"""
        if not msg:
            raise RuntimeError("Received empty scenario message")
        environment.scenario = Scenario.from_string(msg.data)  # type: ignore[attr-defined]

    def cleanup(self) -> None:
        """Clean up resources"""
        if hasattr(self, "log_consumer"):
            self.log_consumer.kill()  # Kill the log consumer greenlet

        if self.environment.runner:
            self.environment.runner.quit()
            self.environment.runner = None

        # Gracefully terminate worker processes
        for i, process in enumerate(self._worker_processes):
            try:
                if process.is_alive():
                    logger.info(f"Terminating worker {i}")
                    process.terminate()
                    process.join(timeout=10)  # Wait up to 10 seconds

                    # Force kill if still alive
                    if process.is_alive():
                        logger.warning(f"Force killing worker {i}")
                        process.kill()
                        process.join()
            except Exception as e:
                logger.error(f"Error terminating worker {i}: {e}")

    def _register_message_handlers(self) -> None:
        """Register message handlers based on runner type.

        Message Flow:
        1. Master Process:
           - SENDS: "update_scenario", "update_batch_size" to workers
           - RECEIVES: "request_metrics", "worker_log" from workers
           - REGISTERS: "request_metrics", "worker_log" handler

        2. Worker Process:
           - SENDS: "request_metrics" to master
           - RECEIVES: "update_scenario", "update_batch_size" from master
           - REGISTERS: "update_scenario", "update_batch_size" handlers

        3. Local Mode:
           - SENDS/RECEIVES: all messages (single process)
           - REGISTERS: all handlers

        Note:
            While it's safe to register all handlers in all modes (Locust handles
            message routing), we explicitly register only the handlers needed for each
            mode to be clear about the message flow.

            Message concurrency is handled by Locust's message queue - all messages
            are processed serially in a greenlet, so no additional synchronization is
            needed.

            Reference: https://docs.locust.io/en/stable/running-distributed.html
        """
        if not self.environment.runner:
            return

        if self.config.num_workers > 0:
            # Distributed mode
            if isinstance(self.environment.runner, WorkerRunner):
                # Workers receive scenario and batch updates from master
                self.environment.runner.register_message(
                    "update_scenario", self._handle_scenario_update
                )
                self.environment.runner.register_message(
                    "update_batch_size", self._handle_batch_size_update
                )
            if isinstance(self.environment.runner, MasterRunner):
                # Master receives metrics and logs from workers
                self.environment.runner.register_message(
                    "request_metrics", self._create_metrics_handler()
                )
                self.environment.runner.register_message(
                    "worker_log", self._create_log_handler()
                )
        else:
            # Local mode needs all handlers since it's both sender and receiver
            self.environment.runner.register_message(
                "update_scenario", self._handle_scenario_update
            )
            self.environment.runner.register_message(
                "update_batch_size", self._handle_batch_size_update
            )
            self.environment.runner.register_message(
                "request_metrics", self._create_metrics_handler()
            )

    def update_scenario(self, scenario_str: str) -> None:
        """Update scenario on all nodes"""
        if self.environment.runner:
            self.environment.runner.send_message("update_scenario", scenario_str)

    def update_batch_size(self, batch_size: int) -> None:
        """Update batch size on all nodes"""
        if self.environment.runner:
            self.environment.runner.send_message("update_batch_size", batch_size)

    def _handle_batch_size_update(self, environment: Environment, msg: Any) -> None:
        """Handle batch size update messages"""
        if not msg:
            raise RuntimeError("Received empty batch size message")
        if hasattr(environment, "sampler"):
            environment.sampler.batch_size = msg.data
