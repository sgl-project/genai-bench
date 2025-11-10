"""Closed-loop concurrency-based runner for maintaining fixed concurrency."""

import asyncio
import time
from typing import Optional

from genai_bench.async_runner.base import BaseAsyncRunner
from genai_bench.logging import init_logger

logger = init_logger(__name__)


class ClosedLoopRunner(BaseAsyncRunner):
    """
    Closed-loop concurrency runner that maintains a fixed number of concurrent requests.

    When a request completes, a new request is immediately started to maintain the target
    concurrency level. This provides better streaming metrics than Locust while maintaining
    the same concurrency-based load pattern.
    """

    async def _run_closed_loop(
        self,
        target_concurrency: int,
        scenario: str,
        done_flag: dict,
        max_requests: Optional[int],
        max_time_s: Optional[int],
    ) -> None:
        """
        Run in closed-loop mode: maintain target_concurrency concurrent requests.
        When a request completes, immediately start a new one to maintain concurrency.

        Args:
            target_concurrency: Number of concurrent requests to maintain
            scenario: Scenario string
            done_flag: Dict with "done" key to signal completion
            max_requests: Optional maximum number of requests
            max_time_s: Optional maximum time in seconds (handled by timeout in run())
        """
        # Track total requests sent
        request_counter = {"count": 0}  # Use dict for mutability in nested function
        active_tasks = set()

        # Use semaphore to atomically enforce max_requests limit
        # This prevents race conditions where multiple tasks check and increment simultaneously
        semaphore = None
        if max_requests is not None:
            semaphore = asyncio.Semaphore(max_requests)

        async def send_one_and_replace():
            """Send a request, and when it completes, start a new one if we haven't hit limits."""
            # Check if we should stop before starting
            if done_flag["done"]:
                return

            # Acquire semaphore before sending request to atomically enforce max_requests limit
            # This blocks if we've already reached max_requests, preventing overshoot
            if semaphore is not None:
                try:
                    await semaphore.acquire()
                except asyncio.CancelledError:
                    return

            # Check done_flag again after acquiring semaphore (may have changed while waiting)
            if done_flag["done"]:
                if semaphore is not None:
                    semaphore.release()
                return

            # Increment counter after acquiring semaphore (ensures accurate counting)
            request_counter["count"] += 1

            # Check if we've hit max_requests after incrementing
            if max_requests is not None and request_counter["count"] >= max_requests:
                done_flag["done"] = True
                # Don't release semaphore here - we'll release after request completes
                # This ensures we don't exceed max_requests

            try:
                # Send request
                req = self._prepare_request(scenario)
                await self._send_one(req)
            finally:
                # Always release semaphore after request completes (success or failure)
                # This allows other tasks to proceed if we haven't hit max_requests
                if semaphore is not None:
                    semaphore.release()

            # Check if we've hit max_requests after completing this request
            if max_requests is not None and request_counter["count"] >= max_requests:
                done_flag["done"] = True
                return

            # Only create new task if we haven't hit limits and done_flag is still False
            # Double-check both conditions to prevent race conditions
            if done_flag["done"]:
                return  # Early exit if done_flag was set while we were sending

            if max_requests is None or request_counter["count"] < max_requests:
                # Create new task to maintain concurrency
                task = asyncio.create_task(send_one_and_replace())
                active_tasks.add(task)

                # Add done callback to remove from active_tasks
                def remove_task(task):
                    active_tasks.discard(task)

                task.add_done_callback(remove_task)

        # Start initial batch of requests to reach target concurrency
        for _ in range(target_concurrency):
            if done_flag["done"]:
                break
            if max_requests is not None and request_counter["count"] >= max_requests:
                done_flag["done"] = True
                break
            task = asyncio.create_task(send_one_and_replace())
            active_tasks.add(task)
            task.add_done_callback(active_tasks.discard)

        # Wait until done_flag is set (by max_requests or max_time_s timeout)
        # Note: max_time_s timeout is handled at the run() level via asyncio.wait_for()
        consecutive_empty_checks = 0
        max_empty_checks = 10  # If we check 10 times with no tasks, something is wrong

        # Continue while we haven't hit the limit
        # Exit condition: done_flag is set OR (no active tasks AND max_requests reached)
        while not done_flag["done"]:
            # Primary check: have we hit max_requests?
            if max_requests is not None and request_counter["count"] >= max_requests:
                done_flag["done"] = True
                # Cancel all active tasks to ensure clean shutdown
                for task in active_tasks:
                    if not task.done():
                        task.cancel()
                break

            # Check active tasks
            if not active_tasks:
                consecutive_empty_checks += 1
                # If we've hit max_requests and have no tasks, we're definitely done
                if (
                    max_requests is not None
                    and request_counter["count"] >= max_requests
                ):
                    done_flag["done"] = True
                    break
                # If we've checked many times with no tasks, something is wrong
                if consecutive_empty_checks >= max_empty_checks:
                    logger.warning(
                        f"No active tasks after {max_empty_checks} checks, "
                        f"request_counter={request_counter['count']}, max_requests={max_requests}, exiting"
                    )
                    done_flag["done"] = True
                    break
                # Brief wait to see if new tasks start - use yield to let other tasks run
                await asyncio.sleep(0.01)
            else:
                consecutive_empty_checks = 0  # Reset counter when we have tasks
                # Small sleep to avoid busy-waiting, but also yield to let tasks complete
                await asyncio.sleep(0.1)

        # Wait for any remaining tasks to complete (after cancellation)
        if active_tasks:
            # Gather with return_exceptions=True to handle cancelled tasks gracefully
            await asyncio.gather(*active_tasks, return_exceptions=True)

        # Record total arrivals for metrics
        total_sent = request_counter["count"]
        self.aggregated.aggregated_metrics.total_arrivals = total_sent

    def run(
        self,
        *,
        qps_level: Optional[
            float
        ] = None,  # Not used in closed-loop, but kept for API compatibility
        target_concurrency: Optional[int] = None,
        duration_s: int,
        distribution: str,  # Not used in closed-loop, but kept for API compatibility
        random_seed: int,  # Not used in closed-loop, but kept for API compatibility
        max_requests: Optional[int],
        max_time_s: Optional[int],
        scenario: str,
    ) -> float:
        """
        Run closed-loop execution maintaining target concurrency.

        Args:
            qps_level: Not used in closed-loop mode (for API compatibility)
            target_concurrency: Required. Target number of concurrent requests
            duration_s: Planned duration in seconds
            distribution: Not used in closed-loop mode (for API compatibility)
            random_seed: Not used in closed-loop mode (for API compatibility)
            max_requests: Optional maximum number of requests
            max_time_s: Optional maximum time in seconds
            scenario: Scenario string

        Returns:
            Actual duration in seconds
        """
        if target_concurrency is None:
            raise ValueError(
                "target_concurrency is required for ClosedLoopRunner (closed-loop mode)"
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

            # Closed-loop mode: maintain target_concurrency concurrent requests
            await self._run_closed_loop(
                target_concurrency=target_concurrency,
                scenario=scenario,
                done_flag=done_flag,
                max_requests=max_requests,
                max_time_s=max_time_s,
            )

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
                "ClosedLoopRunner.run() cannot be called from an async context. "
                "Please call it from a synchronous function."
            )
        except RuntimeError as e:
            # Check if this is our error or the "no running loop" error
            if "cannot be called from an async context" in str(e):
                raise  # Re-raise our custom error
            # No running loop, safe to use asyncio.run()
            try:
                if max_time_s is not None and max_time_s > 0:
                    asyncio.run(asyncio.wait_for(produce(), timeout=max_time_s))
                else:
                    asyncio.run(produce())
            except asyncio.TimeoutError:
                logger.info("Closed-loop run timed out per max_time_s")
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
