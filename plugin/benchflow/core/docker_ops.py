import asyncio
import os
from pathlib import Path
from typing import Optional

import docker
from docker.models.containers import Container
from docker.types import Ulimit

from plugin.benchflow.core.protocol import BenchConfig, InferenceServiceConfig
from plugin.benchflow.core.utils import (
    cleanup_container,
    create_docker_device_request,
    ensure_docker_image,
    show_container_logs,
    stream_container_logs,
)
from plugin.benchflow.logging import get_logger

logger = get_logger(__name__)


async def start_inference_service(
    config: InferenceServiceConfig, network_name: str, gpu: str
) -> Container:
    """Start inference service in Docker asynchronously."""
    container: Optional[Container] = None
    try:
        # Ensure image is available
        await ensure_docker_image(config.image, config.version)

        client = docker.from_env()

        # Cleanup any existing container
        await asyncio.to_thread(cleanup_container, config.container_name)

        # Build device request
        device_request = create_docker_device_request(gpu)

        # Build command as a list of strings
        #vllm
        #cmd = ["--port", str(config.port)]
        #cmd.extend(config.extra_args)

        #sglang
        #cmd = []
        #cmd.extend(config.extra_args)
        #cmd.extend(["--port", str(config.port)])

        # support both sglang and vllm 
        # Build command as a list of strings
        start_mode = getattr(config, "start_mode", "flags")

        if start_mode == "flags":
            # vLLM style: only pass flags, letting container entrypoint run the binary
            cmd = ["--port", str(config.port)]
            cmd.extend(config.extra_args)

        else:
            # SGLang style: full command override
            # config.extra_args must contain: ["python3", "-m", "sglang.launch_server", ...]
            cmd = config.extra_args


        # Start the container (offload to thread)
        container = await asyncio.to_thread(
            client.containers.run,
            config.image_with_tag,
            cmd,
            device_requests=[device_request],
            environment=config.env_vars,
            name=config.container_name,
            network_mode=network_name,
            shm_size=config.shm_size,
            ulimits=[Ulimit(name="nofile", soft=65535, hard=65535)],
            volumes=config.volumes,
            detach=True,
            tty=True,
        )

        if container is None:
            raise RuntimeError(f"Failed to start container {config.container_name}")

        logger.info(
            f"Started inference service container {config.container_name}: "
            f"{container.id}"
        )
        return container

    except Exception as e:
        logger.error(f"Failed to start inference service: {e}")
        if container:
            await asyncio.to_thread(show_container_logs, container)
        raise


async def run_benchmark(
    config: BenchConfig, api_base: str, network_name: str, debug: bool = False
) -> None:
    """Run genai-bench benchmark using Docker asynchronously."""
    log_path = f"{os.getenv('HOME')}/{config.container_name}.log"
    log_file = Path(log_path)
    host_experiment_dir = f"{os.getenv('HOME')}/benchmark_results"
    container_experiment_dir = "/genai-bench/benchmark_results"
    container: Optional[Container] = None

    try:
        # Ensure the Docker image is available
        await ensure_docker_image(config.image, config.version)

        client = docker.from_env()

        # Cleanup any existing container (offloaded to a thread)
        await asyncio.to_thread(cleanup_container, config.container_name)

        # Prepare environment variables
        environment = config.env_vars.copy()  # Start with user-provided env vars
        environment["TRANSFORMERS_VERBOSITY"] = "error"
        environment["GENAI_BENCH_LOGGING_LEVEL"] = (
            "DEBUG" if debug else environment.get("GENAI_BENCH_LOGGING_LEVEL", "INFO")
        )

        # Build command arguments
        cmd = [
            "benchmark",
            "--api-backend",
            "openai",
            "--api-base",
            api_base,
            "--experiment-base-dir",
            container_experiment_dir,
        ]
        cmd.extend(config.extra_args)

        logger.info(f"Starting benchmark container with command: {' '.join(cmd)}")

        # Ensure log file exists (use asyncio file I/O if possible)
        if not log_file.exists():
            await asyncio.to_thread(log_file.touch)

        volumes = [
            f"{log_path}:/genai-bench/genai_bench.log:rw",
            f"{host_experiment_dir}:{container_experiment_dir}",
        ]
        volumes.extend(config.volumes)

        # Start the container (offloaded to a thread)
        container = await asyncio.to_thread(
            client.containers.run,
            config.image_with_tag,
            cmd,
            environment=environment,
            name=config.container_name,
            network_mode=network_name,
            volumes=volumes,
            shm_size="5g",
            ulimits=[Ulimit(name="nofile", soft=65535, hard=65535)],
            detach=True,
            tty=True,
        )

        if container is None:
            raise RuntimeError(f"Failed to start container {config.container_name}")

        # Stream logs from benchmark container
        await asyncio.to_thread(
            stream_container_logs,
            log_path,
            container,
            prefix=f"[Workflow {config.container_name}]",
        )

        # Check container exit code (offloaded to a thread)
        result = await asyncio.to_thread(container.wait)
        if result["StatusCode"] != 0:
            await asyncio.to_thread(
                show_container_logs,
                container,
                "Benchmark container failed. Container logs:",
            )
            raise RuntimeError(
                f"Benchmark container exited with code {result['StatusCode']}"
            )

    except Exception as e:
        logger.error(f"Failed to run benchmark container: {e}")
        if container:
            await asyncio.to_thread(show_container_logs, container)
        raise
    finally:
        logger.info(f"Experiment results are saved in {host_experiment_dir}.")
