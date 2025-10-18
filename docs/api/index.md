# API Reference

This section provides comprehensive API documentation for GenAI Bench components, including CLI commands, core classes, and usage examples.

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

## Scenario Classes

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

## Data Loading Classes

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

## Authentication Classes

### `UnifiedAuthFactory`
Factory for creating authentication providers.

```python
class UnifiedAuthFactory:
    @staticmethod
    def create_model_auth(provider: str, **kwargs) -> ModelAuthProvider: ...
    
    @staticmethod
    def create_storage_auth(provider: str, **kwargs) -> StorageAuthProvider: ...
```

### `ModelAuthProvider`
Base class for model authentication.

```python
class ModelAuthProvider(ABC):
    @abstractmethod
    def get_auth_headers(self) -> Dict[str, str]: ...
    
    @abstractmethod
    def get_auth_params(self) -> Dict[str, Any]: ...
```

## Storage Classes

### `BaseStorage`
Abstract base class for storage implementations.

```python
class BaseStorage(ABC):
    @abstractmethod
    def upload_file(self, file_path: str, bucket: str, key: str) -> None: ...
    
    @abstractmethod
    def upload_folder(self, folder_path: str, bucket: str, prefix: str = "") -> None: ...
```

### `StorageFactory`
Factory for creating storage providers.

```python
class StorageFactory:
    @staticmethod
    def create_storage(provider: str, auth: StorageAuthProvider) -> BaseStorage: ...
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

## Metrics Classes

### `RequestLevelMetrics`
Metrics for individual requests.

```python
class RequestLevelMetrics(BaseModel):
    ttft: Optional[float] = None  # Time to first token
    tpot: Optional[float] = None  # Time per output token
    e2e_latency: Optional[float] = None  # End-to-end latency
    output_latency: Optional[float] = None  # Output latency
    output_inference_speed: Optional[float] = None  # Output inference speed
    num_input_tokens: Optional[int] = None
    num_output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    input_throughput: Optional[float] = None
    output_throughput: Optional[float] = None
    error_code: Optional[int] = None
    error_message: Optional[str] = None
```

### `AggregatedMetrics`
Aggregated metrics across multiple requests.

```python
class AggregatedMetrics(BaseModel):
    # Contains aggregated statistics for all metrics
    # including percentiles, means, etc.
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

## Usage Examples

### Basic Benchmarking

```python
from genai_bench.auth.unified_factory import UnifiedAuthFactory
from genai_bench.storage.factory import StorageFactory
from genai_bench.scenarios.base import Scenario

# Create authentication
auth = UnifiedAuthFactory.create_model_auth(
    "openai",
    api_key="sk-..."
)

# Create storage
storage_auth = UnifiedAuthFactory.create_storage_auth(
    "aws",
    profile="default",
    region="us-east-1"
)
storage = StorageFactory.create_storage("aws", storage_auth)

# Create scenario
scenario = Scenario.from_string("N(100,50)")

# Upload results
storage.upload_folder(
    "/path/to/results",
    "my-bucket",
    prefix="benchmarks/2024"
)
```

### Custom Plot Generation

```python
from genai_bench.analysis.flexible_plot_report import FlexiblePlotGenerator
from genai_bench.analysis.plot_config import PlotConfig, PlotSpec

# Create plot configuration
config = PlotConfig(
    title="Performance Analysis",
    plots=[
        PlotSpec(
            x_field="concurrency",
            y_fields=["e2e_latency"],
            plot_type="line",
            title="Latency vs Concurrency"
        )
    ]
)

# Generate plots
generator = FlexiblePlotGenerator(config)
generator.generate_plots(
    run_data_list,
    group_key="traffic_scenario",
    experiment_folder="/path/to/results"
)
```

### Custom Dataset Loading

```python
from genai_bench.data.config import DatasetConfig, DatasetSourceConfig
from genai_bench.data.loaders.factory import DataLoaderFactory

# Configure dataset
dataset_config = DatasetConfig(
    source=DatasetSourceConfig(
        type="file",
        path="/path/to/dataset.csv",
        file_format="csv"
    ),
    prompt_column="text"
)

# Load data
data = DataLoaderFactory.load_data_for_task(
    "text-to-text",
    dataset_config
)
```

## Contributing to API Documentation

We welcome contributions to improve our API documentation! If you'd like to help:

1. Add docstrings to undocumented functions
2. Provide usage examples
3. Document edge cases and gotchas
4. Submit a pull request

See our [Contributing Guide](../development/contributing.md) for more details.