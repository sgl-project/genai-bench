import asyncio
import json
from typing import List, Optional

import docker
from docker.models.containers import Container

from plugin.benchflow.core.docker_ops import run_benchmark, start_inference_service
from plugin.benchflow.core.gpu_queue import GPUQueue
from plugin.benchflow.core.protocol import ExecutionMode, WorkflowConfig, WorkflowPair
from plugin.benchflow.core.utils import show_container_logs, wait_for_server
from plugin.benchflow.logging import get_logger

logger = get_logger(__name__)
# Add GPU queue as module-level singleton
gpu_queue = GPUQueue()


async def run_workflow_pair(workflow: WorkflowPair, debug: bool = False) -> None:
    """Run a single workflow pair asynchronously."""
    container: Optional[Container] = None
    try:
        # Acquire GPU asynchronously
        gpu = await gpu_queue.acquire_gpu(
            workflow.service.num_gpu_devices, workflow.service.container_name
        )
        if not gpu:
            raise RuntimeError(
                f"Failed to acquire {workflow.service.num_gpu_devices} GPUs for "
                f"{workflow.service.container_name}. "
                f"Workflow {workflow.name} cannot proceed."
            )

        logger.info(f"Starting workflow pair: {workflow.name}")

        # Create Docker network
        network_name = f"benchmark-network-{workflow.name}"
        client = docker.from_env()
        await asyncio.to_thread(client.networks.create, network_name, driver="bridge")
        logger.info(f"Created Docker network: {network_name}")

        # Start inference service
        container = await start_inference_service(workflow.service, network_name, gpu)

        # Wait for server to be ready
        server_url = f"http://{workflow.service.container_name}:{workflow.service.port}"
       
        if not await wait_for_server(workflow.service.container_name, server_url):
            show_container_logs(
                container,
                error_msg=f"Service {workflow.service.container_name} failed to start. "
                f"Container logs:",
                prefix=f"[Workflow {workflow.name}]",
            )
            raise RuntimeError(
                f"Inference service {workflow.service.container_name} failed to start "
                f"for workflow {workflow.name}. Check logs above for details."
            )

        # Run benchmark
        await run_benchmark(workflow.bench, server_url, network_name, debug)

    except Exception as e:
        logger.error(f"Workflow pair {workflow.name} failed: {e}")
        raise
    finally:
        # Release GPU
        await gpu_queue.release_gpu(gpu)

        # Cleanup resources
        try:
            client = docker.from_env()
            # Remove containers
            for container_name in [
                workflow.service.container_name,
                workflow.bench.container_name,
            ]:
                try:
                    container = client.containers.get(container_name)
                    container.stop()
                except Exception as e:
                    logger.warning(f"Failed to cleanup container {container_name}: {e}")

            # Remove network
            try:
                network = client.networks.get(network_name)
                network.remove()
                logger.info(f"Removed Docker network: {network_name}")
            except Exception as e:
                logger.warning(f"Failed to remove network {network_name}: {e}")

        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")


async def run_workflows_parallel(
    workflows: List[WorkflowPair], max_parallel: int, debug: bool = False
) -> None:
    """Run workflows in parallel with a maximum concurrency limit"""
    failed_workflows = []
    sem = asyncio.Semaphore(max_parallel)

    async def run_with_semaphore(workflow: WorkflowPair):
        try:
            async with sem:
                logger.info(f"Starting parallel workflow: {workflow.name}")
                await run_workflow_pair(workflow, debug)
                logger.info(f"Completed parallel workflow: {workflow.name}")
        except Exception as e:
            logger.error(
                f"Workflow {workflow.name} failed: {e}. Continuing with others."
            )
            failed_workflows.append((workflow.name, str(e)))

    # Create tasks
    tasks = []
    for workflow in workflows:
        task = asyncio.create_task(run_with_semaphore(workflow))
        tasks.append(task)

    # Wait for all tasks to complete
    await asyncio.gather(*tasks, return_exceptions=True)

    if failed_workflows:
        logger.error("Summary of failed workflows:")
        for name, error in failed_workflows:
            logger.error(f"- {name}: {error}")


async def run_workflows_sequential(
    workflows: List[WorkflowPair], debug: bool = False
) -> None:
    """Run workflows sequentially"""
    failed_workflows = []

    for workflow in workflows:
        try:
            await run_workflow_pair(workflow, debug)
        except Exception as e:
            logger.error(
                f"Workflow {workflow.name} failed: {e}. Continuing with next workflow."
            )
            failed_workflows.append((workflow.name, str(e)))
            continue

    if failed_workflows:
        logger.error("Summary of failed workflows:")
        for name, error in failed_workflows:
            logger.error(f"- {name}: {error}")


async def run_workflows(config_path: str, debug: bool = False) -> None:
    """Run workflow pairs based on configuration"""
    try:
        # Load and validate config with auto-naming
        with open(config_path) as f:
            config_dict = json.load(f)
        config = WorkflowConfig.from_dict(config_dict)

        logger.info(f"Running workflows in {config.execution_mode} mode")

        if config.execution_mode == ExecutionMode.PARALLEL:
            await run_workflows_parallel(config.workflows, config.max_parallel, debug)
        else:
            await run_workflows_sequential(config.workflows, debug)

    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
