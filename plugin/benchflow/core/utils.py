import asyncio
import time
from typing import Optional

import docker
from docker.errors import APIError, ImageNotFound, NotFound
from docker.models.containers import Container
from docker.types import DeviceRequest

from plugin.benchflow.logging import get_logger, get_raw_logger

logger = get_logger(__name__)


async def wait_for_server(
    service_container_name: str,
    url: str,
    timeout: int = 60 * 3,
    interval: int = 5,
) -> bool:
    """
    Wait for the server to be ready by executing a health check inside the
    service container asynchronously.

    Args:
        service_container_name (str): Name of the service container.
        url (str): Health check URL inside the container.
        timeout (int): Maximum time to wait for the server to become ready (in seconds).
        interval (int): Time to wait between retries (in seconds).

    Returns:
        bool: True if the server becomes ready, False otherwise.
    """
    try:
        client = docker.from_env()
        service_container = client.containers.get(service_container_name)

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Execute health check inside the container asynchronously
                exec_result = await asyncio.to_thread(
                    service_container.exec_run,
                    f"curl -s -f {url}/health",
                )
                if exec_result.exit_code == 0:
                    logger.info(f"Server {url} is ready.")
                    return True
                else:
                    logger.debug(
                        f"Health check failed for {url} with exit code "
                        f"{exec_result.exit_code}: "
                        f"{exec_result.output.decode().strip()}"
                    )
            except Exception as e:
                logger.debug(f"Health check error for {url}: {e}")

            await asyncio.sleep(interval)

        logger.error(f"Server {url} failed to become ready after {timeout} seconds.")
        return False

    except docker.errors.NotFound:
        logger.error(f"Service container {service_container_name} not found.")
        return False
    except Exception as e:
        logger.error(f"Failed to execute health check: {e}")
        return False


async def ensure_docker_image(image: str, version: str) -> None:
    """Ensure Docker image is available, pull if not present"""
    try:
        client = docker.from_env()
        image_with_tag = f"{image}:{version}"

        try:
            client.images.get(image_with_tag)
            logger.debug(f"Image {image_with_tag} found locally")
            return
        except ImageNotFound:
            logger.info(
                f"Image {image_with_tag} not found locally, attempting to pull..."
            )
            try:
                client.images.pull(image, tag=version)
                logger.info(f"Successfully pulled {image_with_tag}")
            except NotFound as err:
                raise RuntimeError(
                    f"Image {image_with_tag} not found in registry"
                ) from err
            except APIError as err:
                raise RuntimeError(
                    f"Failed to pull image {image_with_tag}: {err}"
                ) from err
    except Exception as e:
        logger.error(f"Docker image check failed: {e}")
        raise


def create_docker_device_request(gpu_devices: str) -> DeviceRequest:
    """Create Docker device request for GPU access"""
    # If no GPUs specified, don't request any
    if not gpu_devices:
        return None

    # Create device request for specific GPUs
    return DeviceRequest(
        capabilities=[["gpu"]],
        device_ids=gpu_devices.split(","),
    )


def cleanup_container(container_name: str, force: bool = True) -> None:
    """
    Forcefully remove a container if it exists.

    Args:
        container_name: Name of the container to remove
        force: Whether to force remove the container.
    """
    try:
        client = docker.from_env()
        existing = client.containers.get(container_name)
        logger.warning(f"Found existing container {container_name}, removing...")
        existing.remove(force=force)
        logger.info(f"Removed existing container {container_name}")
    except NotFound:
        pass  # Container doesn't exist, which is fine
    except Exception as e:
        logger.warning(f"Failed to cleanup container {container_name}: {e}")


def stream_container_logs(
    log_path: str, container: "Container", prefix: Optional[str] = None
) -> None:
    """
    Stream logs from a container log file with optional prefix.

    This is useful when the container logs in a rich format, and it is hard to
    stream it directly using container.logs() without a terminal.

    Args:
        log_path: Path to the log file
        container: Docker container instance
        prefix: Optional prefix for each log line (defaults to container name if None)
    """
    raw_logger = get_raw_logger()  # logs from container is already formatted
    log_prefix = prefix if prefix is not None else f"[{container.name}]"

    with open(log_path) as f:
        while True:
            line = f.readline()
            if line:
                raw_message = line.rstrip()
                raw_logger.info(f"{log_prefix} {raw_message}")
            else:
                # Check if the container is still running
                container.reload()
                if container.status != "running":
                    break
                time.sleep(1)


def show_container_logs(
    container: "Container",
    error_msg: str = "Container logs:",
    prefix: Optional[str] = None,
) -> None:
    """
    Display container logs with a custom error message and optional prefix.

    Args:
        container: Docker container instance
        error_msg: Custom error message to display before logs
        prefix: Optional prefix for each log line (defaults to container name if None)
    """
    if container:
        logger.error(error_msg)
        # Use container name as prefix if none provided
        log_prefix = prefix if prefix is not None else f"[{container.name}]"

        logs = container.logs(stream=False, follow=False).decode()
        for line in logs.splitlines():
            logger.error(f"{log_prefix} {line}")
