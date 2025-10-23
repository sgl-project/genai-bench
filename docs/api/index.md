# API Reference

This section provides comprehensive API documentation for GenAI Bench components, including CLI commands, core classes, and usage examples.

> **Quick Start**: New to GenAI Bench? Check out our [Getting Started Guide](../getting-started/index.md) for installation and basic concepts, or jump to the [User Guide](../user-guide/index.md) for practical examples.

## Getting Started

Before diving into the API reference, we recommend familiarizing yourself with these foundational concepts:

- **[Installation Guide](../getting-started/installation.md)** - Set up GenAI Bench in your environment
- **[Task Definition](../getting-started/task-definition.md)** - Understand supported task types and their requirements
- **[Metrics Definition](../getting-started/metrics-definition.md)** - Learn about performance metrics and measurements
- **[Command Guidelines](../getting-started/command-guidelines.md)** - Best practices for CLI usage

## Table of Contents

- [CLI Commands](#cli-commands)
- [Core Protocol Classes](#core-protocol-classes)
- [Scenario System](#scenario-system)
- [Data Loading System](#data-loading-system)
- [Authentication System](#authentication-system)
- [Storage System](#storage-system)
- [UI and Dashboard System](#ui-and-dashboard-system)
- [Distributed System](#distributed-system)
- [Metrics and Analysis](#metrics-and-analysis)
- [Configuration Classes](#configuration-classes)
- [Comprehensive Examples](#comprehensive-examples)

## CLI Commands

### `genai-bench benchmark`

The main command for running benchmarks against LLM endpoints.

```bash
genai-bench benchmark [OPTIONS]
```

**Key Options:**

- `--api-backend` - API backend (openai, aws-bedrock, azure-openai, gcp-vertex, oci-cohere, oci-genai)
- `--api-key` - API key for authentication
- `--model` - Model name to benchmark
- `--task` - Task type (text-to-text, text-to-embeddings, image-text-to-text, etc.)
- `--traffic-scenario` - Traffic scenario specification
- `--num-concurrency` - Number of concurrent requests
- `--max-time-per-run` - Maximum time per run in seconds
- `--upload-results` - Upload results to cloud storage

> **📖 Learn More**: For detailed usage examples and multi-cloud configurations, see the [Run Benchmark Guide](../user-guide/run-benchmark.md) and [Multi-Cloud Authentication Guide](../user-guide/multi-cloud-auth-storage.md).

**Example:**
```bash
genai-bench benchmark \
  --api-backend openai \
  --api-key $OPENAI_KEY \
  --model gpt-4 \
  --task text-to-text \
  --traffic-scenario "N(100,50)" \
  --num-concurrency 1,2,4,8 \
  --max-time-per-run 300
```

> **💡 Tip**: For traffic scenario syntax and examples, see the [Traffic Scenarios Guide](../user-guide/scenario-definition.md).

### `genai-bench excel`

Generate Excel reports from benchmark results.

```bash
genai-bench excel [OPTIONS]
```

**Options:**

- `--experiment-folder` - Path to experiment folder
- `--excel-name` - Name of the Excel file
- `--metric-percentile` - Percentile for metrics (mean, p25, p50, p75, p90, p95, p99)
- `--metrics-time-unit` - Time unit (s, ms)

> **📊 Learn More**: For detailed Excel report generation examples, see the [Excel Reports Guide](../user-guide/generate-excel-sheet.md).

### `genai-bench plot`

Generate plots from benchmark results with flexible configuration.

```bash
genai-bench plot [OPTIONS]
```

**Options:**

- `--experiments-folder` - Path to experiments folder
- `--group-key` - Key to group data by
- `--plot-config` - Path to JSON plot configuration
- `--preset` - Built-in plot preset
- `--filter-criteria` - Filter criteria for data

> **📈 Learn More**: For comprehensive plotting examples and configuration options, see the [Visualizations Guide](../user-guide/generate-plot.md).

## Core Protocol Classes

### Request Classes

#### `UserRequest`
Base class for all user requests.

```python
class UserRequest(BaseModel):
    model: str
    additional_request_params: Dict[str, Any] = Field(default_factory=dict)
```

#### `UserChatRequest`
For text-to-text tasks.

```python
class UserChatRequest(UserRequest):
    prompt: str
    num_prefill_tokens: int | None
    max_tokens: int | None
```

#### `UserEmbeddingRequest`
For text-to-embeddings tasks.

```python
class UserEmbeddingRequest(UserRequest):
    documents: List[str]
    num_prefill_tokens: Optional[int]
```

#### `UserImageChatRequest`
For image-text-to-text tasks.

```python
class UserImageChatRequest(UserChatRequest):
    image_content: List[str]
    num_images: int
```

### Response Classes

#### `UserResponse`
Base class for all user responses.

```python
class UserResponse(BaseModel):
    status_code: int
    time_at_first_token: Optional[float]
    start_time: Optional[float]
    end_time: Optional[float]
    error_message: Optional[str]
    num_prefill_tokens: Optional[int]
```

#### `UserChatResponse`
For chat task responses.

```python
class UserChatResponse(UserResponse):
    generated_text: Optional[str]
    tokens_received: Optional[int]
```

### Experiment Metadata

#### `ExperimentMetadata`
Contains all metadata for an experiment.

```python
class ExperimentMetadata(BaseModel):
    cmd: str
    benchmark_version: str
    api_backend: str
    model: str
    task: str
    num_concurrency: List[int]
    traffic_scenario: List[str]
    max_time_per_run_s: int
    max_requests_per_run: int
    # ... and more fields
```

## Scenario System

> **📖 Learn More**: For comprehensive scenario syntax and usage examples, see the [Traffic Scenarios Guide](../user-guide/scenario-definition.md).

### `Scenario`
Abstract base class for traffic scenarios.

```python
class Scenario(ABC):
    scenario_type: TextDistribution | MultiModality | EmbeddingDistribution | ReRankDistribution | SpecialScenario
    validation_pattern: str
    
    @abstractmethod
    def sample(self) -> Any: ...
    
    @abstractmethod
    def to_string(self) -> str: ...
    
    @classmethod
    @abstractmethod
    def parse(cls, params_str: str) -> "Scenario": ...
```

### Distribution Types

#### `TextDistribution`
```python
class TextDistribution(Enum):
    NORMAL = "N"
    DETERMINISTIC = "D"
    UNIFORM = "U"
```

#### `NormalDistribution`
Normal distribution scenario for text tasks.

```python
class NormalDistribution(Scenario):
    scenario_type = TextDistribution.NORMAL
    validation_pattern = r"^N\(\d+,\d+\)$"
    
    def __init__(self, mean: int, std: int): ...
```

#### `EmbeddingScenario`
Scenario for embedding tasks.

```python
class EmbeddingScenario(Scenario):
    scenario_type = EmbeddingDistribution.EMBEDDING
    validation_pattern = r"^E\(\d+\)$"
    
    def __init__(self, tokens_per_document: int): ...
```

## Sampler Classes

### `Sampler`
Abstract base class for data samplers.

```python
class Sampler(ABC):
    modality_registry: Dict[str, Type["Sampler"]] = {}
    input_modality: str
    supported_tasks: Set[str]
    
    def __init__(self, tokenizer, model: str, output_modality: str, ...): ...
    
    @abstractmethod
    def sample(self, scenario: Scenario) -> UserRequest: ...
    
    @classmethod
    def create(cls, task: str, *args, **kwargs) -> "Sampler": ...
```

### `TextSampler`
For text-based tasks.

```python
class TextSampler(Sampler):
    input_modality = "text"
    supported_tasks = {"text-to-text", "text-to-embeddings", "text-to-rerank"}
    
    def __init__(self, tokenizer, model: str, output_modality: str, data: List[str], ...): ...
```

### `ImageSampler`
For image-based tasks.

```python
class ImageSampler(Sampler):
    input_modality = "image"
    supported_tasks = {"image-text-to-text", "image-to-embeddings"}
    
    def __init__(self, tokenizer, model: str, output_modality: str, data: Any, ...): ...
```

## Data Loading System

> **📖 Learn More**: For dataset configuration examples and advanced usage, see the [Run Benchmark Guide](../user-guide/run-benchmark.md#selecting-datasets).

### `DatasetLoader`
Abstract base class for dataset loaders.

```python
class DatasetLoader(ABC):
    supported_formats: Set[DatasetFormat] = set()
    media_type: str = ""
    
    def __init__(self, dataset_config: DatasetConfig): ...
    
    def load_request(self) -> Union[List[str], List[Tuple[str, Any]]]: ...
```

### `TextDatasetLoader`
For loading text datasets.

```python
class TextDatasetLoader(DatasetLoader):
    supported_formats = {DatasetFormat.TEXT, DatasetFormat.CSV, DatasetFormat.JSON, DatasetFormat.HUGGINGFACE_HUB}
    media_type = "text"
```

### `ImageDatasetLoader`
For loading image datasets.

```python
class ImageDatasetLoader(DatasetLoader):
    supported_formats = {DatasetFormat.CSV, DatasetFormat.JSON, DatasetFormat.HUGGINGFACE_HUB}
    media_type = "image"
```

## Authentication System

> **🔐 Learn More**: For comprehensive authentication setup and multi-cloud configurations, see the [Multi-Cloud Authentication Guide](../user-guide/multi-cloud-auth-storage.md).

### Base Authentication Interfaces

#### `AuthProvider`
Base class for all authentication providers.

```python
class AuthProvider(ABC):
    @abstractmethod
    def get_config(self) -> Dict[str, Any]: ...
    
    @abstractmethod
    def get_credentials(self) -> Any: ...
```

#### `ModelAuthProvider`
Base class for model endpoint authentication.

```python
class ModelAuthProvider(ABC):
    @abstractmethod
    def get_headers(self) -> Dict[str, str]: ...
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]: ...
    
    @abstractmethod
    def get_auth_type(self) -> str: ...
    
    def get_credentials(self) -> Optional[Any]: ...
```

#### `StorageAuthProvider`
Base class for storage authentication.

```python
class StorageAuthProvider(ABC):
    @abstractmethod
    def get_client_config(self) -> Dict[str, Any]: ...
    
    @abstractmethod
    def get_credentials(self) -> Any: ...
    
    @abstractmethod
    def get_storage_type(self) -> str: ...
    
    def get_region(self) -> Optional[str]: ...
```

### Authentication Factory

#### `UnifiedAuthFactory`
Unified factory for creating model and storage authentication providers.

```python
class UnifiedAuthFactory:
    @staticmethod
    def create_model_auth(provider: str, **kwargs) -> ModelAuthProvider: ...
    
    @staticmethod
    def create_storage_auth(provider: str, **kwargs) -> StorageAuthProvider: ...
```

**Supported Model Providers:**
- `openai` - OpenAI API authentication
- `oci` - Oracle Cloud Infrastructure authentication
- `aws-bedrock` - AWS Bedrock authentication
- `azure-openai` - Azure OpenAI authentication
- `gcp-vertex` - Google Cloud Vertex AI authentication

**Supported Storage Providers:**
- `aws` - AWS S3 authentication
- `azure` - Azure Blob Storage authentication
- `gcp` - Google Cloud Storage authentication
- `oci` - Oracle Cloud Infrastructure Object Storage authentication
- `github` - GitHub repository authentication

### Provider-Specific Authentication

#### OpenAI Authentication
```python
# OpenAI API key authentication
auth = UnifiedAuthFactory.create_model_auth(
    "openai",
    api_key="sk-..."
)
```

#### AWS Bedrock Authentication
```python
# AWS Bedrock authentication with multiple options
auth = UnifiedAuthFactory.create_model_auth(
    "aws-bedrock",
    access_key_id="AKIA...",
    secret_access_key="...",
    region="us-east-1"
)

# Or using AWS profile
auth = UnifiedAuthFactory.create_model_auth(
    "aws-bedrock",
    profile="default",
    region="us-west-2"
)
```

#### Azure OpenAI Authentication
```python
# Azure OpenAI authentication
auth = UnifiedAuthFactory.create_model_auth(
    "azure-openai",
    endpoint="https://your-resource.openai.azure.com/",
    deployment="your-deployment",
    api_version="2024-02-15-preview",
    api_key="your-api-key"
)
```

#### GCP Vertex AI Authentication
```python
# GCP Vertex AI authentication
auth = UnifiedAuthFactory.create_model_auth(
    "gcp-vertex",
    project_id="your-project",
    location="us-central1",
    credentials_path="/path/to/credentials.json"
)
```

#### OCI Authentication
```python
# OCI authentication with multiple methods
# User Principal (default)
auth = UnifiedAuthFactory.create_model_auth(
    "oci",
    config_path="~/.oci/config",
    profile="DEFAULT"
)

# Instance Principal
auth = UnifiedAuthFactory.create_model_auth(
    "oci",
    auth_type="instance_principal"
)

# OBO Token
auth = UnifiedAuthFactory.create_model_auth(
    "oci",
    auth_type="obo_token",
    token="your-obo-token"
)
```

### Storage Authentication Examples

#### AWS S3 Storage
```python
# AWS S3 authentication
storage_auth = UnifiedAuthFactory.create_storage_auth(
    "aws",
    access_key_id="AKIA...",
    secret_access_key="...",
    region="us-east-1"
)
```

#### Azure Blob Storage
```python
# Azure Blob Storage authentication
storage_auth = UnifiedAuthFactory.create_storage_auth(
    "azure",
    account_name="your-storage-account",
    account_key="your-account-key"
)

# Or using connection string
storage_auth = UnifiedAuthFactory.create_storage_auth(
    "azure",
    connection_string="DefaultEndpointsProtocol=https;AccountName=..."
)
```

#### Google Cloud Storage
```python
# GCP Cloud Storage authentication
storage_auth = UnifiedAuthFactory.create_storage_auth(
    "gcp",
    project_id="your-project",
    credentials_path="/path/to/credentials.json"
)
```

#### OCI Object Storage
```python
# OCI Object Storage authentication
storage_auth = UnifiedAuthFactory.create_storage_auth(
    "oci",
    config_path="~/.oci/config",
    profile="DEFAULT"
)
```

#### GitHub Storage
```python
# GitHub repository authentication
storage_auth = UnifiedAuthFactory.create_storage_auth(
    "github",
    token="ghp_...",
    owner="your-username",
    repo="your-repo"
)
```

## Storage System

> **💾 Learn More**: For storage configuration and upload examples, see the [Upload Results Guide](../user-guide/upload-benchmark-result.md) and [Multi-Cloud Authentication Guide](../user-guide/multi-cloud-auth-storage.md).

### Base Storage Interface

#### `BaseStorage`
Abstract base class for all storage implementations.

```python
class BaseStorage(ABC):
    @abstractmethod
    def upload_file(
        self, local_path: Union[str, Path], remote_path: str, bucket: str, **kwargs
    ) -> None: ...
    
    @abstractmethod
    def upload_folder(
        self, local_folder: Union[str, Path], bucket: str, prefix: str = "", **kwargs
    ) -> None: ...
    
    @abstractmethod
    def download_file(
        self, remote_path: str, local_path: Union[str, Path], bucket: str, **kwargs
    ) -> None: ...
    
    @abstractmethod
    def list_objects(
        self, bucket: str, prefix: Optional[str] = None, **kwargs
    ) -> Generator[str, None, None]: ...
    
    @abstractmethod
    def delete_object(self, remote_path: str, bucket: str, **kwargs) -> None: ...
    
    @abstractmethod
    def get_storage_type(self) -> str: ...
```

### Storage Factory

#### `StorageFactory`
Factory for creating storage provider instances.

```python
class StorageFactory:
    @staticmethod
    def create_storage(
        provider: str, auth: StorageAuthProvider, **kwargs
    ) -> BaseStorage: ...
```

**Supported Storage Providers:**
- `aws` - AWS S3 storage
- `azure` - Azure Blob Storage
- `gcp` - Google Cloud Storage
- `oci` - Oracle Cloud Infrastructure Object Storage
- `github` - GitHub repository storage

### Storage Provider Implementations

#### AWS S3 Storage
```python
class AWSS3Storage(BaseStorage):
    """AWS S3 storage implementation."""
    
    def __init__(self, auth: StorageAuthProvider, **kwargs): ...
    
    def upload_file(self, local_path, remote_path, bucket, **kwargs): ...
    def upload_folder(self, local_folder, bucket, prefix="", **kwargs): ...
    def download_file(self, remote_path, local_path, bucket, **kwargs): ...
    def list_objects(self, bucket, prefix=None, **kwargs): ...
    def delete_object(self, remote_path, bucket, **kwargs): ...
    def get_storage_type(self) -> str: ...
```

**Features:**
- Full S3 API support
- Automatic multipart uploads for large files
- Server-side encryption support
- Lifecycle policy management
- Cross-region replication support

#### Azure Blob Storage
```python
class AzureBlobStorage(BaseStorage):
    """Azure Blob Storage implementation."""
    
    def __init__(self, auth: StorageAuthProvider, **kwargs): ...
    
    def upload_file(self, local_path, remote_path, bucket, **kwargs): ...
    def upload_folder(self, local_folder, bucket, prefix="", **kwargs): ...
    def download_file(self, remote_path, local_path, bucket, **kwargs): ...
    def list_objects(self, bucket, prefix=None, **kwargs): ...
    def delete_object(self, remote_path, bucket, **kwargs): ...
    def get_storage_type(self) -> str: ...
```

**Features:**
- Blob storage with tier management
- Access control and SAS tokens
- Blob versioning support
- Soft delete capabilities
- Change feed support

#### Google Cloud Storage
```python
class GCPCloudStorage(BaseStorage):
    """Google Cloud Storage implementation."""
    
    def __init__(self, auth: StorageAuthProvider, **kwargs): ...
    
    def upload_file(self, local_path, remote_path, bucket, **kwargs): ...
    def upload_folder(self, local_folder, bucket, prefix="", **kwargs): ...
    def download_file(self, remote_path, local_path, bucket, **kwargs): ...
    def list_objects(self, bucket, prefix=None, **kwargs): ...
    def delete_object(self, remote_path, bucket, **kwargs): ...
    def get_storage_type(self) -> str: ...
```

**Features:**
- Multi-regional and regional storage classes
- Object lifecycle management
- Fine-grained access control
- Data encryption at rest and in transit
- Cloud CDN integration

#### OCI Object Storage
```python
class OCIObjectStorage(BaseStorage):
    """Oracle Cloud Infrastructure Object Storage implementation."""
    
    def __init__(self, auth: StorageAuthProvider, **kwargs): ...
    
    def upload_file(self, local_path, remote_path, bucket, **kwargs): ...
    def upload_folder(self, local_folder, bucket, prefix="", **kwargs): ...
    def download_file(self, remote_path, local_path, bucket, **kwargs): ...
    def list_objects(self, bucket, prefix=None, **kwargs): ...
    def delete_object(self, remote_path, bucket, **kwargs): ...
    def get_storage_type(self) -> str: ...
```

**Features:**
- High-performance object storage
- Automatic data replication
- Object versioning
- Cross-region backup
- Integration with OCI services

#### GitHub Storage
```python
class GitHubStorage(BaseStorage):
    """GitHub repository storage implementation."""
    
    def __init__(self, auth: StorageAuthProvider, **kwargs): ...
    
    def upload_file(self, local_path, remote_path, bucket, **kwargs): ...
    def upload_folder(self, local_folder, bucket, prefix="", **kwargs): ...
    def download_file(self, remote_path, local_path, bucket, **kwargs): ...
    def list_objects(self, bucket, prefix=None, **kwargs): ...
    def delete_object(self, remote_path, bucket, **kwargs): ...
    def get_storage_type(self) -> str: ...
```

**Features:**
- Git-based versioning
- Pull request integration
- Branch-based organization
- GitHub Actions integration
- Collaborative workflows

### Storage Usage Examples

#### Basic File Operations
```python
from genai_bench.storage.factory import StorageFactory
from genai_bench.auth.unified_factory import UnifiedAuthFactory

# Create storage authentication
storage_auth = UnifiedAuthFactory.create_storage_auth(
    "aws",
    access_key_id="AKIA...",
    secret_access_key="...",
    region="us-east-1"
)

# Create storage instance
storage = StorageFactory.create_storage("aws", storage_auth)

# Upload a single file
storage.upload_file(
    local_path="/path/to/file.txt",
    remote_path="benchmarks/2024/file.txt",
    bucket="my-bucket"
)

# Upload entire folder
storage.upload_folder(
    local_folder="/path/to/results",
    bucket="my-bucket",
    prefix="benchmarks/2024/"
)

# Download file
storage.download_file(
    remote_path="benchmarks/2024/file.txt",
    local_path="/path/to/downloaded.txt",
    bucket="my-bucket"
)

# List objects
for obj in storage.list_objects("my-bucket", prefix="benchmarks/"):
    print(f"Object: {obj}")

# Delete object
storage.delete_object(
    remote_path="benchmarks/2024/file.txt",
    bucket="my-bucket"
)
```

#### Multi-Cloud Storage Setup
```python
# AWS S3
aws_storage = StorageFactory.create_storage(
    "aws", 
    UnifiedAuthFactory.create_storage_auth("aws", profile="default")
)

# Azure Blob Storage
azure_storage = StorageFactory.create_storage(
    "azure",
    UnifiedAuthFactory.create_storage_auth("azure", account_name="mystorage")
)

# Google Cloud Storage
gcp_storage = StorageFactory.create_storage(
    "gcp",
    UnifiedAuthFactory.create_storage_auth("gcp", project_id="my-project")
)

# Upload to multiple providers
for storage in [aws_storage, azure_storage, gcp_storage]:
    storage.upload_folder(
        local_folder="/path/to/results",
        bucket="my-bucket",
        prefix="backup/"
    )
```

#### Advanced Storage Operations
```python
# Upload with metadata
storage.upload_file(
    local_path="/path/to/file.txt",
    remote_path="benchmarks/2024/file.txt",
    bucket="my-bucket",
    metadata={"experiment": "llm-benchmark", "version": "1.0"}
)

# Upload with server-side encryption
storage.upload_file(
    local_path="/path/to/file.txt",
    remote_path="benchmarks/2024/file.txt",
    bucket="my-bucket",
    encryption="AES256"
)

# List with filtering
for obj in storage.list_objects(
    bucket="my-bucket",
    prefix="benchmarks/2024/",
    max_keys=100
):
    print(f"Found: {obj}")
```

## UI and Dashboard System

> **📊 Learn More**: For dashboard usage and configuration examples, see the [Run Benchmark Guide](../user-guide/run-benchmark.md#distributed-benchmark).

### Dashboard Components

#### `Dashboard`
Union type for dashboard implementations.

```python
Dashboard = Union[RichLiveDashboard, MinimalDashboard]
```

#### `RichLiveDashboard`
Real-time dashboard with rich UI components for live metrics visualization.

```python
class RichLiveDashboard:
    def __init__(self, metrics_time_unit: str = "s"): ...
    
    def update_metrics_panels(
        self, live_metrics: LiveMetricsData, metrics_time_unit: str = "s"
    ): ...
    
    def update_histogram_panel(
        self, live_metrics: LiveMetricsData, metrics_time_unit: str = "s"
    ): ...
    
    def update_scatter_plot_panel(
        self, scatter_plot_metrics: Optional[List[float]], time_unit: str = "s"
    ): ...
    
    def update_benchmark_progress_bars(self, progress_increment: float): ...
    
    def create_benchmark_progress_task(self, run_name: str): ...
    
    def update_total_progress_bars(self, total_runs: int): ...
    
    def start_run(self, run_time: int, start_time: float, max_requests_per_run: int): ...
    
    def calculate_time_based_progress(self) -> float: ...
    
    def handle_single_request(
        self, live_metrics: LiveMetricsData, total_requests: int, error_code: int | None
    ): ...
    
    def reset_panels(self): ...
```

**Features:**
- Real-time metrics visualization
- Interactive progress tracking
- Histogram and scatter plot displays
- Live updates with configurable refresh rates
- Rich console output with colors and formatting

#### `MinimalDashboard`
Lightweight dashboard for headless or minimal UI scenarios.

```python
class MinimalDashboard:
    def __init__(self, metrics_time_unit: str = "s"): ...
    
    def update_metrics_panels(self, live_metrics: LiveMetricsData, metrics_time_unit: str = "s"): ...
    def update_histogram_panel(self, live_metrics: LiveMetricsData, metrics_time_unit: str = "s"): ...
    def update_scatter_plot_panel(self, scatter_plot_metrics: Optional[List[float]], time_unit: str = "s"): ...
    def update_benchmark_progress_bars(self, progress_increment: float): ...
    def create_benchmark_progress_task(self, run_name: str): ...
    def update_total_progress_bars(self, total_runs: int): ...
    def start_run(self, run_time: int, start_time: float, max_requests_per_run: int): ...
    def calculate_time_based_progress(self) -> float: ...
    def handle_single_request(self, live_metrics: LiveMetricsData, total_requests: int, error_code: int | None): ...
    def reset_panels(self): ...
```

**Features:**
- No-op implementations for all dashboard methods
- Minimal resource usage
- Suitable for automated/CI environments
- Compatible with all dashboard interfaces

### Dashboard Factory

#### `create_dashboard`
Factory function for creating appropriate dashboard based on environment.

```python
def create_dashboard(metrics_time_unit: str = "s") -> Dashboard:
    """Factory function that returns either RichLiveDashboard or MinimalDashboard based on ENABLE_UI."""
```

**Environment Variables:**
- `ENABLE_UI=true` - Creates `RichLiveDashboard`
- `ENABLE_UI=false` - Creates `MinimalDashboard`

### Layout System

#### `create_layout`
Creates the main dashboard layout structure.

```python
def create_layout() -> Layout: ...
```

**Layout Structure:**
- **Row 1**: Total Progress and Benchmark Progress
- **Row 2**: Input and Output metrics panels
- **Row 3**: Scatter plots (TTFT vs Input Throughput, Output Latency vs Output Throughput)
- **Logs**: Log output display

#### `create_metric_panel`
Creates individual metric panels with latency and throughput data.

```python
def create_metric_panel(
    title, latency_data, throughput_data, metrics_time_unit: str = "s"
) -> Panel: ...
```

#### `create_progress_bars`
Creates progress tracking bars.

```python
def create_progress_bars() -> Tuple[Progress, Progress, int]: ...
```

### Plot Components

#### `create_horizontal_colored_bar_chart`
Creates horizontal bar charts for histogram visualization.

```python
def create_horizontal_colored_bar_chart(
    data: List[float], 
    title: str, 
    max_width: int = 50
) -> str: ...
```

#### `create_scatter_plot`
Creates scatter plot visualizations for correlation analysis.

```python
def create_scatter_plot(
    x_data: List[float], 
    y_data: List[float], 
    title: str
) -> str: ...
```

### Live Metrics Data

#### `LiveMetricsData`
Structure for real-time metrics data.

```python
LiveMetricsData = {
    "ttft": List[float],
    "input_throughput": List[float], 
    "output_throughput": List[float],
    "output_latency": List[float],
    "stats": Dict[str, Any]
}
```

### Dashboard Usage Examples

#### Basic Dashboard Setup
```python
from genai_bench.ui.dashboard import create_dashboard

# Create dashboard (automatically selects based on ENABLE_UI)
dashboard = create_dashboard(metrics_time_unit="s")

# Use with context manager for live updates
with dashboard.live:
    # Update metrics
    dashboard.update_metrics_panels(live_metrics)
    
    # Update progress
    dashboard.update_benchmark_progress_bars(0.1)
    
    # Update plots
    dashboard.update_scatter_plot_panel(scatter_data)
```

#### Custom Dashboard Configuration
```python
import os

# Force minimal dashboard
os.environ["ENABLE_UI"] = "false"
dashboard = create_dashboard()

# Force rich dashboard
os.environ["ENABLE_UI"] = "true"
dashboard = create_dashboard()
```

#### Real-time Metrics Update
```python
# Live metrics data structure
live_metrics = {
    "ttft": [0.1, 0.2, 0.15, 0.3],
    "input_throughput": [100, 120, 110, 90],
    "output_throughput": [50, 60, 55, 45],
    "output_latency": [0.5, 0.6, 0.55, 0.7],
    "stats": {
        "mean_ttft": 0.1875,
        "mean_input_throughput": 105.0,
        "mean_output_throughput": 52.5,
        "mean_output_latency": 0.5875
    }
}

# Update dashboard with live data
dashboard.update_metrics_panels(live_metrics, metrics_time_unit="s")
dashboard.update_histogram_panel(live_metrics, metrics_time_unit="s")
dashboard.update_scatter_plot_panel(live_metrics["ttft"], time_unit="s")
```

## User Classes

### `BaseUser`
Abstract base class for user implementations.

```python
class BaseUser(HttpUser):
    supported_tasks: Dict[str, str] = {}
    
    @classmethod
    def is_task_supported(cls, task: str) -> bool: ...
    
    def sample(self) -> UserRequest: ...
    
    def collect_metrics(self, user_response: UserResponse, endpoint: str): ...
```

### Provider-Specific User Classes

- `OpenAIUser` - OpenAI API implementation
- `AWSBedrockUser` - AWS Bedrock implementation
- `AzureOpenAIUser` - Azure OpenAI implementation
- `GCPVertexUser` - GCP Vertex AI implementation
- `OCICohereUser` - OCI Cohere implementation
- `OCIGenAIUser` - OCI GenAI implementation

## Distributed System

> **⚡ Learn More**: For distributed benchmarking setup and best practices, see the [Run Benchmark Guide](../user-guide/run-benchmark.md#distributed-benchmark).

### Distributed Configuration

#### `DistributedConfig`
Configuration for distributed benchmark execution.

```python
@dataclass
class DistributedConfig:
    num_workers: int
    master_host: str = "127.0.0.1"
    master_port: int = 5557
    wait_time: int = 2
    pin_to_cores: bool = False
    cpu_affinity_map: Optional[Dict[int, int]] = None
```

**Configuration Options:**
- `num_workers` - Number of worker processes (0 for local mode)
- `master_host` - Host for master process communication
- `master_port` - Port for master-worker communication
- `wait_time` - Wait time for worker startup
- `pin_to_cores` - Enable CPU core pinning (experimental)
- `cpu_affinity_map` - Custom worker-to-CPU mapping

### Distributed Runner

#### `DistributedRunner`
Manages distributed load test execution with master and worker processes.

```python
class DistributedRunner:
    def __init__(
        self,
        environment: Environment,
        config: DistributedConfig,
        dashboard: Optional[Dashboard] = None,
    ): ...
    
    def setup(self) -> None: ...
    
    def update_scenario(self, scenario: str) -> None: ...
    
    def update_batch_size(self, batch_size: int) -> None: ...
    
    def cleanup(self) -> None: ...
```

**Architecture Overview:**

1. **Process Model:**
   - **Master Process**: Controls test execution and aggregates metrics
   - **Worker Processes**: Execute actual API requests and send metrics to master
   - **Local Mode**: Single process handles both execution and aggregation

2. **Message Flow:**
   - **Master → Workers:**
     - `"update_scenario"`: Updates test scenario configuration
     - `"update_batch_size"`: Updates batch size for requests
   - **Workers → Master:**
     - `"request_metrics"`: Sends metrics from each request for aggregation
     - `"worker_log"`: Sends worker logs to master

3. **Execution Flow:**
   - **Master Process:**
     - Sets up worker processes
     - Controls test scenarios and batch sizes
     - Aggregates metrics from workers
     - Runs the main benchmark loop
     - Updates dashboard with live metrics
   - **Worker Processes:**
     - Receive test configurations from master
     - Execute API requests
     - Send metrics back to master
     - Do NOT execute the main benchmark loop

4. **Message Registration:**
   - **Master**: registers `"request_metrics"` handler
   - **Workers**: register `"update_scenario"`, `"update_batch_size"` handlers
   - **Local mode**: registers all handlers

5. **Metrics Collection:**
   - Only master/local maintains `AggregatedMetricsCollector`
   - Workers collect individual request metrics and send to master
   - Master aggregates metrics and updates dashboard

### Message Handler Protocol

#### `MessageHandler`
Protocol for message handling in distributed system.

```python
class MessageHandler(Protocol):
    def __call__(self, environment: Environment, msg: Any, **kwargs) -> None: ...
```

### Distributed System Usage Examples

#### Basic Distributed Setup
```python
from genai_bench.distributed.runner import DistributedRunner, DistributedConfig
from genai_bench.ui.dashboard import create_dashboard

# Configure distributed execution
config = DistributedConfig(
    num_workers=4,
    master_host="127.0.0.1",
    master_port=5557,
    wait_time=2
)

# Create dashboard
dashboard = create_dashboard()

# Create distributed runner
runner = DistributedRunner(environment, config, dashboard)
runner.setup()

# If worker process, exit after setup
if isinstance(environment.runner, WorkerRunner):
    return

# Master continues with test execution
runner.update_scenario("N(100,50)")
runner.update_batch_size(32)
```

#### Advanced Configuration
```python
# CPU-optimized distributed setup
config = DistributedConfig(
    num_workers=8,
    master_host="0.0.0.0",  # Allow external connections
    master_port=5557,
    wait_time=5,
    pin_to_cores=True,
    cpu_affinity_map={
        0: 0, 1: 1, 2: 2, 3: 3,  # Worker -> CPU mapping
        4: 4, 5: 5, 6: 6, 7: 7
    }
)

runner = DistributedRunner(environment, config, dashboard)
runner.setup()
```

#### Local vs Distributed Mode
```python
# Local mode (single process)
config = DistributedConfig(num_workers=0)
runner = DistributedRunner(environment, config, dashboard)
runner.setup()

# Distributed mode (multiple processes)
config = DistributedConfig(num_workers=4)
runner = DistributedRunner(environment, config, dashboard)
runner.setup()
```

#### Dynamic Scenario Updates
```python
# Update scenario during execution
runner.update_scenario("D(200,100)")  # Deterministic scenario
runner.update_scenario("N(150,75)")  # Normal distribution
runner.update_scenario("U(50,250)")  # Uniform distribution

# Update batch size for embedding tasks
runner.update_batch_size(16)
runner.update_batch_size(32)
```

#### Cleanup and Resource Management
```python
# Automatic cleanup on exit
import atexit
atexit.register(runner.cleanup)

# Manual cleanup
runner.cleanup()
```

### Performance Considerations

#### Worker Process Optimization
- **CPU Pinning**: Pin workers to specific CPU cores for better performance
- **Process Count**: Balance between CPU cores and memory usage
- **Memory Management**: Monitor memory usage with high worker counts

#### Network Configuration
- **Master Host**: Use `0.0.0.0` for external worker connections
- **Port Selection**: Choose non-conflicting ports for multiple instances
- **Wait Time**: Adjust based on worker startup time

#### Resource Monitoring
```python
import psutil

# Monitor system resources
cpu_count = multiprocessing.cpu_count()
memory_gb = psutil.virtual_memory().total / (1024**3)

# Recommended worker count
recommended_workers = min(cpu_count * 2, 16)
```

## Metrics and Analysis

> **📊 Learn More**: For metrics definitions and analysis examples, see the [Metrics Definition Guide](../getting-started/metrics-definition.md) and [Excel Reports Guide](../user-guide/generate-excel-sheet.md).

### Metrics Collection Components

#### `RequestLevelMetrics`
Metrics for individual requests with comprehensive tracking.

```python
class RequestLevelMetrics(BaseModel):
    ttft: Optional[float] = Field(None, description="Time to first token (TTFT)")
    tpot: Optional[float] = Field(None, description="Time per output token (TPOT)")
    e2e_latency: Optional[float] = Field(None, description="End-to-end latency")
    output_latency: Optional[float] = Field(None, description="Output latency")
    output_inference_speed: Optional[float] = Field(
        None, description="Output inference speed in tokens/s"
    )
    num_input_tokens: Optional[int] = Field(None, description="Number of input tokens")
    num_output_tokens: Optional[int] = Field(
        None, description="Number of output tokens"
    )
    total_tokens: Optional[int] = Field(None, description="Total tokens processed")
    input_throughput: Optional[float] = Field(
        None, description="Input throughput in tokens/s"
    )
    output_throughput: Optional[float] = Field(
        None, description="Output throughput in tokens/s"
    )
    error_code: Optional[int] = Field(None, description="Error code")
    error_message: Optional[str] = Field(None, description="Error message")
```

#### `MetricStats`
Statistical analysis for individual metrics.

```python
class MetricStats(BaseModel):
    # Statistical measures for each metric
    ttft: MetricStat = Field(default_factory=MetricStat)
    tpot: MetricStat = Field(default_factory=MetricStat)
    e2e_latency: MetricStat = Field(default_factory=MetricStat)
    output_latency: MetricStat = Field(default_factory=MetricStat)
    output_inference_speed: MetricStat = Field(default_factory=MetricStat)
    num_input_tokens: MetricStat = Field(default_factory=MetricStat)
    num_output_tokens: MetricStat = Field(default_factory=MetricStat)
    total_tokens: MetricStat = Field(default_factory=MetricStat)
    input_throughput: MetricStat = Field(default_factory=MetricStat)
    output_throughput: MetricStat = Field(default_factory=MetricStat)
```

#### `AggregatedMetrics`
Comprehensive aggregated metrics across multiple requests.

```python
class AggregatedMetrics(BaseModel):
    # Run Metadata
    scenario: Optional[str] = Field(None, description="The sample scenario")
    num_concurrency: int = Field(1, description="Number of concurrency")
    batch_size: int = Field(1, description="Batch size for embedding tasks")
    iteration_type: str = Field(
        "num_concurrency",
        description="Type of iteration used (num_concurrency or batch_size)",
    )
    
    # Performance Metrics
    run_duration: float = Field(0.0, description="Run duration in seconds.")
    mean_output_throughput_tokens_per_s: float = Field(0.0, description="Mean output throughput")
    mean_input_throughput_tokens_per_s: float = Field(0.0, description="Mean input throughput")
    mean_total_tokens_throughput_tokens_per_s: float = Field(0.0, description="Mean total throughput")
    mean_total_chars_per_hour: float = Field(0.0, description="Mean chars per hour")
    requests_per_second: float = Field(0.0, description="Average requests per second")
    
    # Error Tracking
    error_codes_frequency: Dict[int, int] = Field(default_factory=dict, description="Error code frequency")
    error_rate: float = Field(0.0, description="Error rate across all requests")
    num_error_requests: int = Field(0, description="Number of error requests")
    num_completed_requests: int = Field(0, description="Number of completed requests")
    num_requests: int = Field(0, description="Number of total requests")
    
    # Statistical Analysis
    stats: MetricStats = Field(default_factory=MetricStats, description="Statistical analysis")
```

### Metrics Collectors

#### `RequestMetricsCollector`
Collects and calculates metrics for individual requests.

```python
class RequestMetricsCollector:
    def __init__(self): ...
    
    def calculate_metrics(self, response: UserResponse): ...
```

**Features:**
- Automatic metric calculation from response data
- Error handling and validation
- Support for different response types
- Token counting and throughput calculation

#### `AggregatedMetricsCollector`
Advanced metrics aggregation with statistical analysis.

```python
class AggregatedMetricsCollector:
    def __init__(self): ...
    
    def add_single_request_metrics(self, metrics: RequestLevelMetrics): ...
    
    def aggregate_metrics_data(
        self,
        start_time: float,
        end_time: float,
        dataset_character_to_token_ratio: float,
        warmup_ratio: Optional[float],
        cooldown_ratio: Optional[float],
    ): ...
    
    def get_live_metrics_data(self) -> LiveMetricsData: ...
```

**Features:**
- Real-time metrics aggregation
- Statistical analysis (percentiles, means, std dev)
- Warmup and cooldown period filtering
- Live metrics data generation
- Error rate calculation

### Time Unit Conversion

#### `TimeUnitConverter`
Converts metrics between different time units.

```python
class TimeUnitConverter:
    @staticmethod
    def convert_time_unit(value: float, from_unit: str, to_unit: str) -> float: ...
    
    @staticmethod
    def convert_throughput_unit(value: float, from_unit: str, to_unit: str) -> float: ...
```

**Supported Units:**
- Time: `s` (seconds), `ms` (milliseconds), `μs` (microseconds)
- Throughput: `tokens/s`, `tokens/ms`, `tokens/μs`

### Live Metrics System

#### `LiveMetricsData`
Real-time metrics data structure.

```python
LiveMetricsData = {
    "ttft": List[float],
    "input_throughput": List[float],
    "output_throughput": List[float],
    "output_latency": List[float],
    "stats": Dict[str, Any]
}
```

### Metrics Usage Examples

#### Basic Metrics Collection
```python
from genai_bench.metrics.request_metrics_collector import RequestMetricsCollector
from genai_bench.metrics.aggregated_metrics_collector import AggregatedMetricsCollector

# Collect individual request metrics
request_collector = RequestMetricsCollector()
request_collector.calculate_metrics(user_response)

# Aggregate metrics across multiple requests
aggregated_collector = AggregatedMetricsCollector()

# Add individual request metrics
for request_metrics in request_metrics_list:
    aggregated_collector.add_single_request_metrics(request_metrics)

# Perform final aggregation
aggregated_collector.aggregate_metrics_data(
    start_time=start_time,
    end_time=end_time,
    dataset_character_to_token_ratio=4.0,
    warmup_ratio=0.1,
    cooldown_ratio=0.1
)
```

#### Time Unit Conversion
```python
from genai_bench.time_units import TimeUnitConverter

# Convert latency from seconds to milliseconds
latency_ms = TimeUnitConverter.convert_time_unit(
    latency_s, "s", "ms"
)

# Convert throughput from tokens/s to tokens/ms
throughput_ms = TimeUnitConverter.convert_throughput_unit(
    throughput_s, "tokens/s", "tokens/ms"
)
```

#### Live Metrics Monitoring
```python
# Get live metrics data
live_metrics = aggregated_collector.get_live_metrics_data()

# Update dashboard with live data
dashboard.update_metrics_panels(live_metrics, metrics_time_unit="s")
dashboard.update_histogram_panel(live_metrics, metrics_time_unit="s")
```

#### Statistical Analysis
```python
# Access statistical data
stats = aggregated_metrics.stats

# Get specific metric statistics
ttft_stats = stats.ttft
print(f"TTFT - Mean: {ttft_stats.mean}, P95: {ttft_stats.p95}")

# Get error analysis
print(f"Error Rate: {aggregated_metrics.error_rate}")
print(f"Error Codes: {aggregated_metrics.error_codes_frequency}")
```

## Advanced Data Loading System

### Dataset Configuration

#### `DatasetConfig`
Complete dataset configuration with flexible source support.

```python
class DatasetConfig(BaseModel):
    source: DatasetSourceConfig
    prompt_column: Optional[str] = None
    image_column: Optional[str] = None
    prompt_lambda: Optional[str] = None
    unsafe_allow_large_images: bool = False
    
    @classmethod
    def from_file(cls, config_path: str) -> "DatasetConfig": ...
    
    @classmethod
    def from_cli_args(
        cls,
        dataset_path: Optional[str] = None,
        prompt_column: Optional[str] = None,
        image_column: Optional[str] = None,
        **kwargs,
    ) -> "DatasetConfig": ...
```

#### `DatasetSourceConfig`
Configuration for different dataset sources.

```python
class DatasetSourceConfig(BaseModel):
    type: str = Field(..., description="Dataset source type: 'file', 'huggingface', or 'custom'")
    path: Optional[str] = Field(None, description="Path to dataset (file path or HuggingFace ID)")
    file_format: Optional[str] = Field(None, description="File format: 'csv', 'txt', 'json'")
    huggingface_kwargs: Optional[Dict[str, Any]] = Field(
        None, description="Keyword arguments passed directly to HuggingFace load_dataset"
    )
    loader_class: Optional[str] = Field(None, description="Python import path for custom dataset loader")
    loader_kwargs: Optional[Dict[str, Any]] = Field(None, description="Keyword arguments for custom loader")
```

### Dataset Sources

#### `DatasetSource`
Abstract base class for dataset sources.

```python
class DatasetSource(ABC):
    def __init__(self, config: DatasetSourceConfig): ...
    
    @abstractmethod
    def load(self) -> Any: ...
```

#### `FileDatasetSource`
Load datasets from local files (txt, csv, json).

```python
class FileDatasetSource(DatasetSource):
    def load(self) -> Union[List[str], List[Tuple[str, Any]]]: ...
    
    def _load_text_file(self, file_path: Path) -> List[str]: ...
    def _load_csv_file(self, file_path: Path) -> Any: ...
    def _load_json_file(self, file_path: Path) -> List[Any]: ...
```

#### `HuggingFaceDatasetSource`
Load datasets from HuggingFace Hub.

```python
class HuggingFaceDatasetSource(DatasetSource):
    def load(self) -> Any: ...
```

### Data Loaders

#### `DatasetLoader`
Abstract base class for dataset loaders.

```python
class DatasetLoader(ABC):
    supported_formats: Set[DatasetFormat] = set()
    media_type: str = ""
    
    def __init__(self, dataset_config: DatasetConfig): ...
    
    def load_request(self) -> Union[List[str], List[Tuple[str, Any]]]: ...
```

#### `TextDatasetLoader`
For loading text datasets.

```python
class TextDatasetLoader(DatasetLoader):
    supported_formats = {DatasetFormat.TEXT, DatasetFormat.CSV, DatasetFormat.JSON, DatasetFormat.HUGGINGFACE_HUB}
    media_type = "text"
```

#### `ImageDatasetLoader`
For loading image datasets.

```python
class ImageDatasetLoader(DatasetLoader):
    supported_formats = {DatasetFormat.CSV, DatasetFormat.JSON, DatasetFormat.HUGGINGFACE_HUB}
    media_type = "image"
```

### Data Loading Factory

#### `DataLoaderFactory`
Factory for creating data loaders and loading data.

```python
class DataLoaderFactory:
    @staticmethod
    def load_data_for_task(
        task: str, dataset_config: DatasetConfig
    ) -> Union[List[str], List[Tuple[str, Any]]]: ...
    
    @staticmethod
    def _load_text_data(
        dataset_config: DatasetConfig, output_modality: str
    ) -> List[str]: ...
    
    @staticmethod
    def _load_image_data(
        dataset_config: DatasetConfig,
    ) -> List[Tuple[str, Any]]: ...
```

### Data Loading Usage Examples

#### File-based Dataset Loading
```python
from genai_bench.data.config import DatasetConfig, DatasetSourceConfig
from genai_bench.data.loaders.factory import DataLoaderFactory

# Load from CSV file
dataset_config = DatasetConfig(
    source=DatasetSourceConfig(
        type="file",
        path="/path/to/dataset.csv",
        file_format="csv"
    ),
    prompt_column="text"
)

data = DataLoaderFactory.load_data_for_task("text-to-text", dataset_config)
```

#### HuggingFace Dataset Loading
```python
# Load from HuggingFace Hub
dataset_config = DatasetConfig(
    source=DatasetSourceConfig(
        type="huggingface",
        path="squad",
        huggingface_kwargs={
            "split": "train",
            "streaming": True
        }
    ),
    prompt_column="question"
)

data = DataLoaderFactory.load_data_for_task("text-to-text", dataset_config)
```

#### Custom Dataset Loading
```python
# Load with custom loader
dataset_config = DatasetConfig(
    source=DatasetSourceConfig(
        type="custom",
        loader_class="my_package.CustomLoader",
        loader_kwargs={
            "api_key": "your-api-key",
            "endpoint": "https://api.example.com/data"
        }
    )
)

data = DataLoaderFactory.load_data_for_task("text-to-text", dataset_config)
```

#### Image Dataset Loading
```python
# Load image dataset
dataset_config = DatasetConfig(
    source=DatasetSourceConfig(
        type="file",
        path="/path/to/images.csv",
        file_format="csv"
    ),
    prompt_column="caption",
    image_column="image_path"
)

data = DataLoaderFactory.load_data_for_task("image-text-to-text", dataset_config)
```

#### Configuration from File
```python
# Load configuration from JSON file
dataset_config = DatasetConfig.from_file("/path/to/dataset_config.json")

# Load data
data = DataLoaderFactory.load_data_for_task("text-to-text", dataset_config)
```

#### CLI Integration
```python
# Create configuration from CLI arguments
dataset_config = DatasetConfig.from_cli_args(
    dataset_path="/path/to/dataset.csv",
    prompt_column="text",
    image_column="image"
)

data = DataLoaderFactory.load_data_for_task("text-to-text", dataset_config)
```

## Analysis and Reporting Classes

### `FlexiblePlotGenerator`
Generates plots using flexible configuration.

```python
class FlexiblePlotGenerator:
    def __init__(self, config: PlotConfig): ...
    
    def generate_plots(
        self,
        run_data_list: List[Tuple[ExperimentMetadata, ExperimentMetrics]],
        group_key: str,
        experiment_folder: str,
        metrics_time_unit: str = "s"
    ) -> None: ...
```

### `PlotConfig`
Configuration for plot generation.

```python
class PlotConfig(BaseModel):
    title: str
    plots: List[PlotSpec]
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 100
    # ... more configuration options
```

### `ExperimentLoader`
Loads experiment data from files.

```python
def load_multiple_experiments(
    folder_name: str, 
    filter_criteria=None
) -> List[Tuple[ExperimentMetadata, ExperimentMetrics]]: ...

def load_one_experiment(
    folder_name: str, 
    filter_criteria: Optional[Dict[str, Any]] = None
) -> Tuple[Optional[ExperimentMetadata], ExperimentMetrics]: ...
```

## Configuration Classes

> **⚙️ Learn More**: For configuration examples and best practices, see the [Run Benchmark Guide](../user-guide/run-benchmark.md#selecting-datasets).

### `DatasetConfig`
Configuration for dataset loading.

```python
class DatasetConfig(BaseModel):
    source: DatasetSourceConfig
    prompt_column: Optional[str] = None
    image_column: Optional[str] = None
    unsafe_allow_large_images: bool = False
```

### `DatasetSourceConfig`
Configuration for dataset sources.

```python
class DatasetSourceConfig(BaseModel):
    type: Literal["file", "huggingface", "custom"]
    path: Optional[str] = None
    file_format: Optional[str] = None
    huggingface_dataset: Optional[str] = None
    huggingface_config: Optional[str] = None
    huggingface_split: Optional[str] = None
    loader_class: Optional[str] = None
    loader_kwargs: Optional[Dict[str, Any]] = None
```

## Comprehensive Examples

> **🚀 Learn More**: For step-by-step tutorials and practical examples, see the [User Guide](../user-guide/index.md).

### Complete Multi-Cloud Benchmarking Setup

#### End-to-End Benchmarking Pipeline
```python
import os
from genai_bench.auth.unified_factory import UnifiedAuthFactory
from genai_bench.storage.factory import StorageFactory
from genai_bench.distributed.runner import DistributedRunner, DistributedConfig
from genai_bench.ui.dashboard import create_dashboard
from genai_bench.data.config import DatasetConfig, DatasetSourceConfig
from genai_bench.data.loaders.factory import DataLoaderFactory

# 1. Configure Authentication
model_auth = UnifiedAuthFactory.create_model_auth(
    "openai",
    api_key=os.getenv("OPENAI_API_KEY")
)

storage_auth = UnifiedAuthFactory.create_storage_auth(
    "aws",
    access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region="us-east-1"
)

# 2. Create Storage
storage = StorageFactory.create_storage("aws", storage_auth)

# 3. Configure Dataset
dataset_config = DatasetConfig(
    source=DatasetSourceConfig(
        type="huggingface",
        path="squad",
        huggingface_kwargs={"split": "train", "streaming": True}
    ),
    prompt_column="question"
)

# 4. Load Data
data = DataLoaderFactory.load_data_for_task("text-to-text", dataset_config)

# 5. Configure Distributed Execution
config = DistributedConfig(
    num_workers=4,
    master_host="127.0.0.1",
    master_port=5557
)

# 6. Create Dashboard
dashboard = create_dashboard(metrics_time_unit="s")

# 7. Run Benchmark
runner = DistributedRunner(environment, config, dashboard)
runner.setup()

# Upload results
storage.upload_folder(
    "/path/to/results",
    "my-bucket",
    prefix="benchmarks/2024/"
)
```

#### Multi-Provider Authentication Setup
```python
# OpenAI + AWS S3
openai_auth = UnifiedAuthFactory.create_model_auth("openai", api_key="sk-...")
aws_storage_auth = UnifiedAuthFactory.create_storage_auth("aws", profile="default")

# Azure OpenAI + Azure Blob
azure_auth = UnifiedAuthFactory.create_model_auth(
    "azure-openai",
    endpoint="https://your-resource.openai.azure.com/",
    deployment="your-deployment",
    api_key="your-api-key"
)
azure_storage_auth = UnifiedAuthFactory.create_storage_auth(
    "azure",
    account_name="your-storage-account",
    account_key="your-account-key"
)

# GCP Vertex + GCP Storage
gcp_auth = UnifiedAuthFactory.create_model_auth(
    "gcp-vertex",
    project_id="your-project",
    location="us-central1"
)
gcp_storage_auth = UnifiedAuthFactory.create_storage_auth(
    "gcp",
    project_id="your-project"
)

# OCI GenAI + OCI Object Storage
oci_auth = UnifiedAuthFactory.create_model_auth(
    "oci",
    config_path="~/.oci/config",
    profile="DEFAULT"
)
oci_storage_auth = UnifiedAuthFactory.create_storage_auth(
    "oci",
    config_path="~/.oci/config",
    profile="DEFAULT"
)
```

#### Advanced Distributed Configuration
```python
# High-performance distributed setup
config = DistributedConfig(
    num_workers=8,
    master_host="0.0.0.0",  # Allow external connections
    master_port=5557,
    wait_time=5,
    pin_to_cores=True,
    cpu_affinity_map={
        0: 0, 1: 1, 2: 2, 3: 3,
        4: 4, 5: 5, 6: 6, 7: 7
    }
)

# Create runner with custom dashboard
dashboard = create_dashboard(metrics_time_unit="ms")
runner = DistributedRunner(environment, config, dashboard)
runner.setup()

# Dynamic scenario updates
scenarios = ["N(100,50)", "N(200,100)", "D(150,150)", "U(50,250)"]
for scenario in scenarios:
    runner.update_scenario(scenario)
    # Run benchmark with this scenario
    # ... benchmark execution ...
```

#### Custom Dataset Loading Examples
```python
# Text dataset from CSV
text_config = DatasetConfig(
    source=DatasetSourceConfig(
        type="file",
        path="/path/to/text_data.csv",
        file_format="csv"
    ),
    prompt_column="text"
)

# Image dataset from JSON
image_config = DatasetConfig(
    source=DatasetSourceConfig(
        type="file",
        path="/path/to/images.json",
        file_format="json"
    ),
    prompt_column="caption",
    image_column="image_path"
)

# HuggingFace dataset with custom parameters
hf_config = DatasetConfig(
    source=DatasetSourceConfig(
        type="huggingface",
        path="squad",
        huggingface_kwargs={
            "split": "train",
            "streaming": True,
            "cache_dir": "/tmp/hf_cache"
        }
    ),
    prompt_column="question"
)

# Custom dataset loader
custom_config = DatasetConfig(
    source=DatasetSourceConfig(
        type="custom",
        loader_class="my_package.CustomDataLoader",
        loader_kwargs={
            "api_endpoint": "https://api.example.com/data",
            "api_key": "your-api-key",
            "batch_size": 1000
        }
    )
)

# Load data for different tasks
text_data = DataLoaderFactory.load_data_for_task("text-to-text", text_config)
image_data = DataLoaderFactory.load_data_for_task("image-text-to-text", image_config)
hf_data = DataLoaderFactory.load_data_for_task("text-to-embeddings", hf_config)
custom_data = DataLoaderFactory.load_data_for_task("text-to-text", custom_config)
```

#### Advanced Storage Operations
```python
# Multi-cloud backup
providers = ["aws", "azure", "gcp", "oci"]
storages = []

for provider in providers:
    auth = UnifiedAuthFactory.create_storage_auth(provider, **provider_configs[provider])
    storage = StorageFactory.create_storage(provider, auth)
    storages.append(storage)

# Upload to all providers
for storage in storages:
    storage.upload_folder(
        local_folder="/path/to/results",
        bucket="benchmark-results",
        prefix="backup/2024/"
    )

# Advanced upload with metadata
storage.upload_file(
    local_path="/path/to/results.json",
    remote_path="benchmarks/2024/results.json",
    bucket="my-bucket",
    metadata={
        "experiment": "llm-benchmark",
        "model": "gpt-4",
        "version": "1.0",
        "timestamp": "2024-01-01T00:00:00Z"
    },
    encryption="AES256"
)

# List and filter objects
for obj in storage.list_objects(
    bucket="my-bucket",
    prefix="benchmarks/2024/",
    max_keys=100
):
    if obj.endswith(".json"):
        print(f"Found result file: {obj}")
```

#### Custom Plot Generation
```python
from genai_bench.analysis.flexible_plot_report import FlexiblePlotGenerator
from genai_bench.analysis.plot_config import PlotConfig, PlotSpec

# Create comprehensive plot configuration
config = PlotConfig(
    title="LLM Performance Analysis",
    plots=[
        PlotSpec(
            x_field="concurrency",
            y_fields=["e2e_latency", "ttft"],
            plot_type="line",
            title="Latency vs Concurrency",
            x_label="Concurrency Level",
            y_label="Latency (ms)"
        ),
        PlotSpec(
            x_field="concurrency",
            y_fields=["input_throughput", "output_throughput"],
            plot_type="bar",
            title="Throughput vs Concurrency",
            x_label="Concurrency Level",
            y_label="Throughput (tokens/s)"
        ),
        PlotSpec(
            x_field="input_throughput",
            y_fields=["e2e_latency"],
            plot_type="scatter",
            title="Latency vs Input Throughput",
            x_label="Input Throughput (tokens/s)",
            y_label="Latency (ms)"
        )
    ],
    figure_size=(15, 10),
    dpi=300
)

# Generate plots
generator = FlexiblePlotGenerator(config)
generator.generate_plots(
    run_data_list,
    group_key="traffic_scenario",
    experiment_folder="/path/to/results",
    metrics_time_unit="ms"
)
```

#### Metrics Analysis and Monitoring
```python
from genai_bench.metrics.aggregated_metrics_collector import AggregatedMetricsCollector
from genai_bench.analysis.experiment_loader import load_multiple_experiments

# Load experiment data
experiments = load_multiple_experiments(
    folder_name="/path/to/experiments",
    filter_criteria={"model": "gpt-4", "task": "text-to-text"}
)

# Analyze metrics
for metadata, metrics in experiments:
    print(f"Experiment: {metadata.experiment_folder_name}")
    print(f"Model: {metadata.model}")
    print(f"Task: {metadata.task}")
    print(f"Concurrency: {metadata.num_concurrency}")
    
    # Performance metrics
    print(f"Mean TTFT: {metrics.stats.ttft.mean:.3f}ms")
    print(f"P95 TTFT: {metrics.stats.ttft.p95:.3f}ms")
    print(f"Mean Throughput: {metrics.mean_output_throughput_tokens_per_s:.2f} tokens/s")
    print(f"Error Rate: {metrics.error_rate:.2%}")
    
    # Error analysis
    if metrics.error_codes_frequency:
        print("Error Codes:")
        for code, count in metrics.error_codes_frequency.items():
            print(f"  {code}: {count} occurrences")
```

#### Dashboard Customization
```python
import os

# Force rich dashboard for development
os.environ["ENABLE_UI"] = "true"
dashboard = create_dashboard(metrics_time_unit="s")

# Force minimal dashboard for production
os.environ["ENABLE_UI"] = "false"
dashboard = create_dashboard(metrics_time_unit="ms")

# Custom dashboard usage
with dashboard.live:
    # Update with live metrics
    live_metrics = {
        "ttft": [0.1, 0.2, 0.15, 0.3],
        "input_throughput": [100, 120, 110, 90],
        "output_throughput": [50, 60, 55, 45],
        "output_latency": [0.5, 0.6, 0.55, 0.7],
        "stats": {
            "mean_ttft": 0.1875,
            "mean_input_throughput": 105.0,
            "mean_output_throughput": 52.5,
            "mean_output_latency": 0.5875
        }
    }
    
    dashboard.update_metrics_panels(live_metrics, metrics_time_unit="s")
    dashboard.update_histogram_panel(live_metrics, metrics_time_unit="s")
    dashboard.update_scatter_plot_panel(live_metrics["ttft"], time_unit="s")
```

#### Time Unit Conversion Examples
```python
from genai_bench.time_units import TimeUnitConverter

# Convert latency metrics
latency_s = 0.5
latency_ms = TimeUnitConverter.convert_time_unit(latency_s, "s", "ms")
latency_us = TimeUnitConverter.convert_time_unit(latency_s, "s", "μs")

print(f"Latency: {latency_s}s = {latency_ms}ms = {latency_us}μs")

# Convert throughput metrics
throughput_s = 100.0  # tokens/s
throughput_ms = TimeUnitConverter.convert_throughput_unit(throughput_s, "tokens/s", "tokens/ms")
throughput_us = TimeUnitConverter.convert_throughput_unit(throughput_s, "tokens/s", "tokens/μs")

print(f"Throughput: {throughput_s} tokens/s = {throughput_ms} tokens/ms = {throughput_us} tokens/μs")
```

#### Error Handling and Recovery
```python
import logging
from genai_bench.logging import init_logger

logger = init_logger(__name__)

try:
    # Create authentication
    auth = UnifiedAuthFactory.create_model_auth("openai", api_key="invalid-key")
except ValueError as e:
    logger.error(f"Authentication failed: {e}")
    # Fallback to different provider
    auth = UnifiedAuthFactory.create_model_auth("azure-openai", **azure_config)

try:
    # Create storage
    storage = StorageFactory.create_storage("aws", storage_auth)
except Exception as e:
    logger.error(f"Storage creation failed: {e}")
    # Fallback to local storage
    storage = None

# Handle distributed runner errors
try:
    runner = DistributedRunner(environment, config, dashboard)
    runner.setup()
except Exception as e:
    logger.error(f"Distributed setup failed: {e}")
    # Fallback to local mode
    config.num_workers = 0
    runner = DistributedRunner(environment, config, dashboard)
    runner.setup()
```

### Production Deployment Examples

#### Docker-based Deployment
```dockerfile
FROM python:3.9-slim

# Install dependencies
RUN pip install genai-bench[all]

# Set environment variables
ENV ENABLE_UI=false
ENV TOKENIZERS_PARALLELISM=false

# Copy configuration
COPY config/ /app/config/
COPY data/ /app/data/

# Run benchmark
CMD ["genai-bench", "benchmark", "--config", "/app/config/benchmark.yaml"]
```

#### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: genai-bench
spec:
  replicas: 1
  selector:
    matchLabels:
      app: genai-bench
  template:
    metadata:
      labels:
        app: genai-bench
    spec:
      containers:
      - name: genai-bench
        image: genai-bench:latest
        env:
        - name: ENABLE_UI
          value: "false"
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: openai-key
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
```

#### CI/CD Integration
```yaml
name: LLM Benchmarking
on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install genai-bench[all]
    
    - name: Run benchmark
      env:
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
        AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
      run: |
        genai-bench benchmark \
          --api-backend openai \
          --model gpt-4 \
          --task text-to-text \
          --traffic-scenario "N(100,50)" \
          --num-concurrency 1,2,4,8 \
          --max-time-per-run 300 \
          --upload-results \
          --storage-provider aws \
          --storage-bucket benchmark-results
```

## Logging and Utilities

### Logging System

#### `LoggingManager`
Centralized logging management for the application.

```python
class LoggingManager:
    def __init__(self): ...
    
    def setup_logging(self, level: str = "INFO"): ...
    
    def get_logger(self, name: str) -> logging.Logger: ...
```

#### `WorkerLoggingManager`
Specialized logging for distributed worker processes.

```python
class WorkerLoggingManager:
    def __init__(self): ...
    
    def setup_worker_logging(self, worker_id: int): ...
    
    def send_log_to_master(self, message: str, level: str): ...
```

#### `init_logger`
Initialize logger for a specific module.

```python
def init_logger(name: str) -> logging.Logger: ...
```

### Utility Functions

#### `calculate_sonnet_char_token_ratio`
Calculate character-to-token ratio for Sonnet model.

```python
def calculate_sonnet_char_token_ratio() -> float: ...
```

#### `sanitize_string`
Sanitize string for safe usage in file paths and identifiers.

```python
def sanitize_string(text: str) -> str: ...
```

### Logging Usage Examples

#### Basic Logging Setup
```python
from genai_bench.logging import init_logger

# Initialize logger for your module
logger = init_logger(__name__)

# Use logger
logger.info("Starting benchmark")
logger.warning("High memory usage detected")
logger.error("Authentication failed")
```

#### Distributed Logging
```python
from genai_bench.logging import WorkerLoggingManager

# Setup worker logging
worker_logger = WorkerLoggingManager()
worker_logger.setup_worker_logging(worker_id=0)

# Send logs to master
worker_logger.send_log_to_master("Worker started", "INFO")
worker_logger.send_log_to_master("Processing request", "DEBUG")
```

#### Utility Function Usage
```python
from genai_bench.utils import calculate_sonnet_char_token_ratio, sanitize_string

# Calculate token ratio
ratio = calculate_sonnet_char_token_ratio()
print(f"Sonnet char/token ratio: {ratio}")

# Sanitize strings
safe_name = sanitize_string("My Experiment (v1.0)")
print(f"Sanitized: {safe_name}")
```

## CLI System Enhancements

### Option Groups

#### API Options
```python
api_options = [
    click.option("--api-backend", required=True, help="API backend"),
    click.option("--api-base", help="API base URL"),
    click.option("--api-key", help="API key"),
    click.option("--model", required=True, help="Model name"),
    click.option("--task", required=True, help="Task type")
]
```

#### Authentication Options
```python
model_auth_options = [
    click.option("--model-auth-type", help="Model authentication type"),
    click.option("--aws-access-key-id", help="AWS access key"),
    click.option("--aws-secret-access-key", help="AWS secret key"),
    click.option("--azure-endpoint", help="Azure endpoint"),
    click.option("--gcp-project-id", help="GCP project ID")
]

storage_auth_options = [
    click.option("--storage-provider", help="Storage provider"),
    click.option("--storage-bucket", help="Storage bucket"),
    click.option("--storage-prefix", help="Storage prefix")
]
```

#### Distributed Options
```python
distributed_locust_options = [
    click.option("--num-workers", default=0, help="Number of worker processes"),
    click.option("--master-port", default=5557, help="Master port"),
    click.option("--spawn-rate", default=1, help="Spawn rate")
]
```

### Validation Functions

#### `validate_tokenizer`
Validate tokenizer configuration.

```python
def validate_tokenizer(tokenizer_name: str, model: str) -> bool: ...
```

### CLI Usage Examples

#### Basic CLI Usage
```bash
# Run benchmark with OpenAI
genai-bench benchmark \
  --api-backend openai \
  --api-key $OPENAI_KEY \
  --model gpt-4 \
  --task text-to-text \
  --traffic-scenario "N(100,50)" \
  --num-concurrency 1,2,4,8

# Run with Azure OpenAI
genai-bench benchmark \
  --api-backend azure-openai \
  --azure-endpoint https://your-resource.openai.azure.com/ \
  --azure-deployment your-deployment \
  --model gpt-4 \
  --task text-to-text

# Run with distributed workers
genai-bench benchmark \
  --api-backend openai \
  --model gpt-4 \
  --task text-to-text \
  --num-workers 4 \
  --master-port 5557
```

#### Advanced CLI Usage
```bash
# Multi-cloud setup
genai-bench benchmark \
  --api-backend openai \
  --model gpt-4 \
  --task text-to-text \
  --upload-results \
  --storage-provider aws \
  --storage-bucket my-bucket \
  --storage-prefix benchmarks/2024

# Custom dataset
genai-bench benchmark \
  --api-backend openai \
  --model gpt-4 \
  --task text-to-text \
  --dataset-path /path/to/dataset.csv \
  --dataset-prompt-column text

# HuggingFace dataset
genai-bench benchmark \
  --api-backend openai \
  --model gpt-4 \
  --task text-to-text \
  --dataset-config /path/to/dataset_config.json
```

## Contributing to API Documentation

We welcome contributions to improve our API documentation! If you'd like to help:

1. **Add docstrings** to undocumented functions and classes
2. **Provide usage examples** for complex components
3. **Document edge cases** and common gotchas
4. **Update examples** with new features and best practices
5. **Add troubleshooting sections** for common issues
6. **Submit a pull request** with your improvements

### Documentation Guidelines

- **Code Examples**: Include complete, runnable examples
- **Error Handling**: Show how to handle common errors
- **Best Practices**: Highlight recommended usage patterns
- **Cross-References**: Link related components and concepts
- **Version Compatibility**: Note any version-specific features

### Areas Needing Documentation

- **Custom Authentication Providers**: How to implement custom auth
- **Custom Storage Providers**: How to add new storage backends
- **Custom Dataset Loaders**: How to create custom data sources
- **Performance Tuning**: Optimization strategies and tips
- **Troubleshooting**: Common issues and solutions

See our [Contributing Guide](../development/contributing.md) for more details on how to contribute to the project.

## Troubleshooting and Support

### Common Issues

- **Authentication Problems**: See the [Multi-Cloud Authentication Guide](../user-guide/multi-cloud-auth-storage.md) for detailed setup instructions
- **Performance Issues**: Check the [Distributed Benchmarking Guide](../user-guide/run-benchmark.md#distributed-benchmark) for optimization tips
- **Dataset Loading**: Refer to the [Dataset Configuration Examples](../user-guide/run-benchmark.md#selecting-datasets) for proper setup
- **Storage Upload**: See the [Upload Results Guide](../user-guide/upload-benchmark-result.md) for troubleshooting storage issues

### Additional Resources

- **[Development Guide](../development/index.md)** - Contributing and development setup
- **[Multi-Cloud Quick Reference](../user-guide/multi-cloud-quick-reference.md)** - Quick setup reference for all providers

### Getting Help

If you encounter issues not covered in the documentation:

1. Check the [GitHub Issues](https://github.com/sgl-project/genai-bench/issues) for known problems
2. Review the [Multi-Cloud Authentication Guide](../user-guide/multi-cloud-auth-storage.md) for provider-specific issues
3. Consult the [Run Benchmark Guide](../user-guide/run-benchmark.md) for usage examples
4. Open a new issue with detailed error information and configuration