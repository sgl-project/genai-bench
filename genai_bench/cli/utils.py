from locust.env import Environment

import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from genai_bench.logging import init_logger

logger = init_logger(__name__)


def manage_run_time(
    max_time_per_run: int,
    max_requests_per_run: int,
    environment: Environment,
) -> int:
    """
    Manages the run time of the benchmarking process by tracking elapsed time
    and ensuring enough requests are completed before the test ends. The
    function will exit when one of the two conditions is met:
    1. The maximum allowed run time (`max_time_per_run`) is reached.
    2. The total number of requests exceeds the maximum requests per run
       (`max_requests_per_run`).

    Args:
        max_time_per_run (int): The maximum allowed run time in seconds.
        max_requests_per_run (int): The maximum number of requests per
            run.
        environment: The environment object with runner stats.

    Returns:
        int: The actual run time in seconds.
    """

    total_run_time = 0

    while total_run_time < max_time_per_run:
        time.sleep(1)
        total_run_time += 1

        assert environment.runner is not None, "environment.runner should not be None"
        total_completed_requests = environment.runner.stats.total.num_requests

        if total_completed_requests >= max_requests_per_run:
            logger.info(
                f"⏩ Exit the run as {total_completed_requests} requests have "
                "been completed."
            )
            break

    return int(total_run_time)


def get_experiment_path(
    experiment_folder_name: Optional[str],
    experiment_base_dir: Optional[str],
    api_backend: str,
    server_engine: Optional[str],
    server_version: Optional[str],
    task: str,
    model: str,
) -> Path:
    """
    Generate experiment path based on provided options and configuration.

    Args:
        experiment_folder_name: Optional custom folder name
        experiment_base_dir: Optional base directory for experiments
            (relative or absolute)
        api_backend: API backend name
        server_engine: Optional server engine name
        server_version: Optional server version
        task: Task name
        model: Model name

    Returns:
        Path: Full path to experiment directory
    """
    # Generate default name if not provided
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_name = (
        f"{api_backend}_"
        f"{server_engine + '_' if server_engine else ''}"
        f"{server_version + '_' if server_version else ''}"
        f"{task}_{model}_{timestamp}"
    )

    # Use provided name or default
    folder_name = experiment_folder_name or default_name

    # Determine full path
    if experiment_base_dir:
        # Convert to absolute path if relative
        base_dir = Path(experiment_base_dir).resolve()
        base_dir.mkdir(parents=True, exist_ok=True)
        experiment_path = base_dir / folder_name
    else:
        experiment_path = Path(folder_name)

    if experiment_path.exists():
        logger.warning(
            f"‼️ The folder {experiment_path} already exists. Data might be overridden."
        )

    experiment_path.mkdir(parents=True, exist_ok=True)
    return experiment_path
