from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class ExecutionMode(str, Enum):
    """Enum to define how workflows should be executed"""

    PARALLEL = "parallel"  # Execute workflows concurrently up to max_parallel limit
    SEQUENTIAL = "sequential"  # Execute workflows one at a time in order


class InferenceServiceConfig(BaseModel):
    """Configuration settings for running an inference service instance"""

    # Docker settings
    container_name: Optional[str] = None  # e.g. "vllm-server"
    image: str  # e.g. "vllm/vllm-openai"
    version: str  # e.g. "latest"

    # Determines how the Docker command is constructed:
    # - "flags": (default) used by vLLM-style images, which rely on the image's ENTRYPOINT
    #            and expect only flags like ["--port", "8080", "--model=..."].
    # - "command": used by SGLang images, which require overriding ENTRYPOINT completely
    #              with a full command such as ["python3", "-m", "sglang.launch_server", ...].
    start_mode: str = "flags"

    # Resource settings
    shm_size: str = "15g"
    num_gpu_devices: int = 1  # Number of GPUs to request

    # Server settings
    port: int = 8080  # Network port for API access

    # Environment variables
    env_vars: dict = Field(default_factory=dict)  # e.g. {"HUGGINGFACE_API_KEY": "xxx"}

    # Volume mappings
    volumes: List[str] = Field(
        default_factory=list
    )  # e.g. ["/host/path:/container/path:mode"]

    # Command line arguments for the service
    extra_args: List[str] = Field(
        default_factory=list
    )  # e.g. ["--model=/models/model-name", "--served-model-name=my-model"]

    @property
    def image_with_tag(self) -> str:
        """Get full image name with tag"""
        return f"{self.image}:{self.version}"


class BenchConfig(BaseModel):
    """Configuration settings for running benchmarks"""

    # Docker settings
    container_name: Optional[str] = None  # e.g. "bench-server"
    image: str = "phx.ocir.io/idqj093njucb/genai-bench"
    version: str

    # Volume mappings
    volumes: List[str] = Field(
        default_factory=lambda: [
            "/mnt/data/models:/models:ro",
            "~/images:/genai-bench/images:rw",
        ]
    )  # e.g. ["/host/path:/container/path:mode"]

    # Environment variables
    env_vars: dict = Field(default_factory=dict)  # e.g. {"HUGGINGFACE_API_KEY": "xxx"}

    # Command line arguments for genai-bench
    extra_args: List[str] = Field(
        default_factory=list
    )  # e.g. ["--task=text-to-text", "--model-tokenizer=/path/to/model"]

    @property
    def image_with_tag(self) -> str:
        """Get full image name with tag"""
        return f"{self.image}:{self.version}"


class WorkflowPair(BaseModel):
    """
    Links an inference service configuration with its corresponding
    benchmark settings
    """

    name: str  # Unique workflow identifier

    service: InferenceServiceConfig  # Service configuration
    bench: BenchConfig  # Benchmark configuration


class WorkflowConfig(BaseModel):
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    max_parallel: int = 1
    workflows: List[WorkflowPair]

    @classmethod
    def from_dict(cls, data: dict) -> "WorkflowConfig":
        """Create config with auto-generated container names"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for i, workflow in enumerate(data["workflows"]):
            # Generate service container name
            if workflow["service"].get("container_name", None) is None:
                workflow["service"]["container_name"] = generate_container_name(
                    workflow["name"], "service", i, timestamp
                )

            # Generate bench container name
            if workflow["bench"].get("container_name", None) is None:
                workflow["bench"]["container_name"] = generate_container_name(
                    workflow["name"], "bench", i, timestamp
                )

        return cls(**data)


def generate_container_name(
    workflow_name: str, service_type: str, index: int, timestamp: str
) -> str:
    """Generate unique container names"""
    return f"{service_type}-{workflow_name}-{index}-{timestamp}"
