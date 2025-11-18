import asyncio
from typing import Set

import nvidia_smi

from plugin.benchflow.logging import get_logger

logger = get_logger(__name__)


class GPUQueue:
    """Manages GPU allocation and queuing based on real-time GPU usage"""

    def __init__(self):
        try:
            nvidia_smi.nvmlInit()
            self.device_count = nvidia_smi.nvmlDeviceGetCount()
            # Add lock for GPU allocation
            self.allocation_lock = asyncio.Lock()
            # Track allocated GPUs
            self.allocated_gpus = set()
        except Exception as e:
            logger.warning(f"Failed to initialize nvidia-smi: {e}")
            self.device_count = 1

    def _get_unused_gpus(
        self, max_utilization: float = 10.0 , max_memory_usage: float = 10.0
    ) -> Set[str]:
        """
        Get list of available GPUs with low utilization and memory usage.
        A GPU is considered available if both utilization and memory usage are below
        thresholds. Excludes already allocated GPUs.
        """
        unused_gpus = set()

        try:
            for i in range(self.device_count):
                gpu_id = str(i)
                # Skip if already allocated
                if gpu_id in self.allocated_gpus:
                    logger.debug(f"GPU {i} is already allocated")
                    continue

                try:
                    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)

                    # Get GPU utilization
                    utilization = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
                    gpu_utilization = utilization.gpu

                    # Get memory utilization
                    memory = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                    memory_usage = (memory.used / memory.total) * 100.0

                    logger.info(
                        f"GPU {i} - Utilization: {gpu_utilization}%, "
                        f"Memory Usage: {memory_usage:.1f}%"
                    )

                    if (
                        gpu_utilization <= max_utilization
                        and memory_usage <= max_memory_usage
                    ):
                        unused_gpus.add(gpu_id)

                except Exception as e:
                    logger.warning(f"Failed to check GPU {i} status: {e}")
                    continue

            if not unused_gpus:
                logger.warning(
                    f"No GPUs found with utilization ≤{max_utilization}% and memory "
                    f"usage ≤{max_memory_usage}%"
                )
            else:
                logger.info(
                    f"Found {len(unused_gpus)} available GPUs: {sorted(unused_gpus)}"
                )

            return unused_gpus

        except Exception as e:
            logger.warning(f"Failed to get GPU status: {e}")
            raise

    async def acquire_gpu(self, request_num_gpus: int, container_name: str) -> str:
        """
        Request specified number of GPUs, checking real-time availability.
        Returns comma-separated string of GPU IDs or empty string if not available.

        Args:
            request_num_gpus: Number of GPUs to request
            container_name: Name of the container requesting GPUs
        """
        async with self.allocation_lock:  # Protect GPU allocation
            logger.info(
                f"Container {container_name} requesting {request_num_gpus} GPUs"
            )

            # Get current unused GPUs
            unused_gpus = self._get_unused_gpus()

            # Check if we have enough GPUs
            if len(unused_gpus) >= request_num_gpus:
                # Take the first n GPUs
                allocated_gpus = sorted(unused_gpus)[:request_num_gpus]
                # Mark GPUs as allocated
                for gpu in allocated_gpus:
                    self.allocated_gpus.add(gpu)
                allocated_str = ",".join(allocated_gpus)
                logger.info(
                    f"Container {container_name} acquired GPUs: {allocated_str}"
                )
                return allocated_str

            logger.error(
                f"Failed to allocate {request_num_gpus} GPUs for container "
                f"{container_name}. Available: {len(unused_gpus)}"
            )
            return ""

    async def release_gpu(self, gpus: str) -> None:
        """Release GPUs back to the pool"""
        if not gpus:
            return

        async with self.allocation_lock:
            for gpu in gpus.split(","):
                if gpu in self.allocated_gpus:
                    self.allocated_gpus.remove(gpu)
                    logger.info(f"Released GPU {gpu} back to pool")

    def __del__(self):
        """Cleanup NVIDIA management library"""
        try:
            nvidia_smi.nvmlShutdown()
        except Exception as e:
            logger.error(f"Failed to shutdown NVIDIA management library: {e}")
