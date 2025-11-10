"""Open-loop QPS-based runner for fixed arrival rate execution."""

import asyncio
import random
import time
from typing import List, Optional

from genai_bench.async_runner.base import BaseAsyncRunner
from genai_bench.logging import init_logger

logger = init_logger(__name__)


class OpenLoopRunner(BaseAsyncRunner):
    """
    Open-loop QPS runner that schedules global inter-arrivals (tore-speed style)
    and emits RequestLevelMetrics via AggregatedMetricsCollector.

    This runner maintains a fixed arrival rate (QPS) regardless of system response time,
    making it suitable for open-loop load testing.
    """

    def _wait_intervals(
        self, qps_level: float, duration_s: int, random_seed: int, distribution: str
    ) -> List[float]:
        """
        Generate inter-arrival intervals for QPS-based scheduling.

        Args:
            qps_level: Target queries per second
            duration_s: Duration in seconds
            random_seed: Random seed for reproducibility
            distribution: Distribution type ("exponential", "uniform", "constant")

        Returns:
            List of wait intervals in seconds
        """
        mean = 1.0 / qps_level
        random.seed(random_seed)
        out: List[float] = []
        for _ in range(int(qps_level * duration_s)):
            if distribution == "exponential":
                out.append(random.expovariate(1.0 / mean))
            elif distribution == "uniform":
                out.append(random.uniform(0, 2 * mean))
            elif distribution == "constant":
                out.append(mean)
            else:
                raise ValueError(f"Invalid distribution: {distribution}")
        return out

    def run(
        self,
        *,
        qps_level: Optional[float] = None,
        target_concurrency: Optional[
            int
        ] = None,  # Not used in open-loop, but kept for API compatibility
        duration_s: int,
        distribution: str,
        random_seed: int,
        max_requests: Optional[int],
        max_time_s: Optional[int],
        scenario: str,
    ) -> float:
        """
        Run open-loop execution with fixed QPS arrival rate.

        Args:
            qps_level: Required. Target queries per second
            target_concurrency: Not used in open-loop mode (for API compatibility)
            duration_s: Planned duration in seconds
            distribution: Inter-arrival distribution ("exponential", "uniform", "constant")
            random_seed: Random seed for reproducibility
            max_requests: Optional maximum number of requests
            max_time_s: Optional maximum time in seconds
            scenario: Scenario string

        Returns:
            Actual duration in seconds
        """
        if qps_level is None:
            raise ValueError(
                "qps_level is required for OpenLoopRunner (open-loop mode)"
            )

        # Validate QPS value is reasonable
        if qps_level <= 0:
            raise ValueError(
                f"qps_level must be positive, got {qps_level}. "
                "QPS (queries per second) must be greater than 0."
            )

        if qps_level > 10000:
            raise ValueError(
                f"qps_level is too high: {qps_level}. "
                "Maximum allowed QPS is 10000. If you need higher throughput, "
                "consider using multiple runner instances or distributed execution."
            )

        start = time.monotonic()

        async def produce():
            # Initialize dashboard if available
            if self.dashboard is not None:
                self.dashboard.start_run(
                    run_time=duration_s,
                    start_time=start,
                    max_requests_per_run=max_requests if max_requests else 0,
                )

            # Periodic UI tick to advance time-based progress even before first completion
            done_flag = {"done": False}

            async def tick_progress():
                if self.dashboard is None:
                    return
                while not done_flag["done"]:
                    try:
                        # Only update time-based progress if no requests have completed yet
                        # Once requests start completing, handle_single_request will handle progress
                        # This prevents flashing backwards when request-based progress > time-based
                        total_completed = (
                            self.aggregated.aggregated_metrics.num_completed_requests
                            + self.aggregated.aggregated_metrics.num_error_requests
                        )
                        if total_completed == 0:
                            # No requests completed yet, safe to update time-based progress
                            progress = self.dashboard.calculate_time_based_progress()
                            self.dashboard.update_benchmark_progress_bars(progress)
                    except Exception as e:
                        # Log error but don't crash - progress updates are non-critical
                        logger.debug(f"Progress update error (non-critical): {e}")
                    await asyncio.sleep(0.5)

            tick_task = None
            if self.dashboard is not None:
                tick_task = asyncio.create_task(tick_progress())

            # Open-loop mode: fixed QPS arrival rate
            intervals = self._wait_intervals(
                qps_level, duration_s, random_seed, distribution
            )
            n = len(intervals)
            if max_requests is not None:
                n = min(n, max_requests)
                intervals = intervals[:n]

            # Generate requests on-demand to match Locust's behavior and avoid memory issues
            # This matches Locust (base_user.py:44, openai_user.py:53) and closed-loop runner (closed_loop.py:68)
            tasks = []
            actual_arrivals = 0
            for wait_s in intervals:
                # Check timeout before sleeping
                if max_time_s is not None and max_time_s > 0:
                    elapsed = time.monotonic() - start
                    if elapsed >= max_time_s:
                        logger.info(
                            f"Open-loop run reached max_time_s ({max_time_s}s), "
                            f"stopping request generation. Elapsed: {elapsed:.2f}s"
                        )
                        break
                await asyncio.sleep(wait_s)
                req = self._prepare_request(scenario)
                tasks.append(asyncio.create_task(self._send_one(req)))
                actual_arrivals += 1
            if tasks:
                # Handle CancelledError and other exceptions gracefully
                try:
                    await asyncio.gather(*tasks, return_exceptions=True)
                except Exception as e:
                    self._handle_error(e, "open-loop task execution")

            # Record total arrivals for open-loop mode (use actual count if timeout occurred)
            self.aggregated.aggregated_metrics.total_arrivals = actual_arrivals if actual_arrivals < n else n

            if tick_task is not None:
                done_flag["done"] = True
                # Give one last update chance
                await asyncio.sleep(0)
                tick_task.cancel()

            # Clean up session after all requests complete
            await self.cleanup()

        # Handle event loop edge cases: check if loop exists before creating new one
        # This prevents errors when called from async context
        # asyncio.get_running_loop() raises RuntimeError if no loop is running
        try:
            asyncio.get_running_loop()
            # If we get here, we're in an async context and can't use asyncio.run()
            # This is a limitation - run() must be called from sync context
            raise RuntimeError(
                "Async runner run() cannot be called from an async context. "
                "Please call it from a synchronous function."
            )
        except RuntimeError as e:
            # Check if this is our error or the "no running loop" error
            if "cannot be called from an async context" in str(e):
                raise  # Re-raise our custom error
            # No running loop, safe to use asyncio.run()
            # Note: max_time_s is now handled internally in the produce loop for graceful shutdown
            asyncio.run(produce())
        end = time.monotonic()
        actual_duration = end - start
        # Record arrivals as an arrival rate metric for this run
        # Use actual duration, not planned duration, for accurate arrival rate
        total_arrivals = self.aggregated.aggregated_metrics.total_arrivals or 0
        arrival_rate = (
            (total_arrivals / actual_duration) if actual_duration > 0 else 0.0
        )
        self.aggregated.aggregated_metrics.arrival_requests_per_second = arrival_rate
        return actual_duration
