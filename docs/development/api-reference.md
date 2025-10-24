# API Reference

This section provides comprehensive API documentation for all GenAI Bench components, organized by functional category.

## Project Structure

```
genai-bench/
├── genai_bench/        # Main package
│   ├── analysis/       # Result analysis and reporting
│   ├── auth/           # Authentication providers
│   ├── cli/            # CLI implementation
│   ├── data/           # Dataset loading and management
│   ├── distributed/    # Distributed execution
│   ├── metrics/        # Metrics collection
│   ├── sampling/       # Data sampling
│   ├── scenarios/      # Traffic generation scenarios
│   ├── storage/        # Storage providers
│   ├── ui/             # User interface components
│   └── user/           # User implementations
├── tests/              # Test suite
└── docs/               # Documentation
```

## Analysis

Components for analyzing benchmark results and generating reports.

### Data Loading

- **`ExperimentLoader`** - Loads experiment data from files
- **`load_multiple_experiments()`** - Loads multiple experiment results
- **`load_one_experiment()`** - Loads single experiment result

### Plot Generation

- **`FlexiblePlotGenerator`** - Generates plots using flexible configuration
- **`plot_experiment_data_flexible()`** - Generates flexible plots

### Configuration

- **`PlotConfig`** - Configuration for plot generation
- **`PlotConfigManager`** - Manages plot configurations
- **`PlotSpec`** - Specification for individual plots

### Report Generation

- **`create_workbook()`** - Creates Excel workbooks from experiment data

### Data Types

- **`ExperimentMetrics`** - Metrics data structure for experiments
- **`MetricsData`** - Union type for aggregated or individual metrics

## Authentication

Components for handling authentication across different cloud providers and services.

### Base Classes

- **`AuthProvider`** - Base class for authentication providers

### Factories

- **`UnifiedAuthFactory`** - Unified factory for creating authentication providers
- **`AuthFactory`** - Legacy factory for authentication providers

### Model Authentication Providers

- **`ModelAuthProvider`** - Base class for model endpoint authentication
- **`OpenAIAuth`** - OpenAI API authentication
- **`AWSBedrockAuth`** - AWS Bedrock authentication
- **`AzureOpenAIAuth`** - Azure OpenAI authentication
- **`GCPVertexAuth`** - GCP Vertex AI authentication
- **`OCIModelAuthAdapter`** - OCI model authentication adapter

### Storage Authentication Providers

- **`StorageAuthProvider`** - Base class for storage authentication
- **`AWSS3Auth`** - AWS S3 authentication
- **`AzureBlobAuth`** - Azure Blob Storage authentication
- **`GCPStorageAuth`** - GCP Cloud Storage authentication
- **`GitHubAuth`** - GitHub authentication
- **`OCIStorageAuthAdapter`** - OCI storage authentication adapter

### OCI Authentication Providers

- **`OCIUserPrincipalAuth`** - OCI user principal authentication
- **`OCIInstancePrincipalAuth`** - OCI instance principal authentication
- **`OCISessionAuth`** - OCI session authentication
- **`OCIOBOTokenAuth`** - OCI on-behalf-of token authentication

## Storage

Components for multi-cloud storage operations.

### Base Classes

- **`BaseStorage`** - Abstract base class for storage providers
- **`StorageFactory`** - Factory for creating storage providers

### Storage Implementations

- **`AWSS3Storage`** - AWS S3 storage implementation
- **`AzureBlobStorage`** - Azure Blob Storage implementation
- **`GCPCloudStorage`** - GCP Cloud Storage implementation
- **`OCIObjectStorage`** - OCI Object Storage implementation
- **`GitHubStorage`** - GitHub storage implementation

### OCI Object Storage Components

- **`DataStore`** - Interface for data store operations
- **`OSDataStore`** - OCI Object Storage data store
- **`ObjectURI`** - Object URI representation

### Operations

- **File Operations**: `upload_file`, `download_file`, `delete_object`
- **Folder Operations**: `upload_folder`
- **Listing**: `list_objects`
- **Multi-cloud Support**: AWS, Azure, GCP, OCI, GitHub

## CLI

Command-line interface components for user interaction.

### Commands

- **`cli`** - Main CLI entry point
- **`benchmark`** - Benchmark command
- **`excel`** - Excel report generation command
- **`plot`** - Plot generation command

### Option Groups

- **`api_options`** - API-related CLI options
- **`model_auth_options`** - Model authentication options
- **`storage_auth_options`** - Storage authentication options
- **`distributed_locust_options`** - Distributed execution options
- **`experiment_options`** - Experiment configuration options
- **`sampling_options`** - Data sampling options
- **`server_options`** - Server configuration options
- **`object_storage_options`** - Object storage options
- **`oci_auth_options`** - OCI-specific authentication options

### Utilities

- **`get_experiment_path()`** - Get experiment file paths
- **`get_run_params()`** - Extract run parameters
- **`manage_run_time()`** - Manage run time limits
- **`validate_tokenizer()`** - Validate tokenizer configuration

### Validation

- **`validate_api_backend()`** - Validate API backend selection
- **`validate_api_key()`** - Validate API keys
- **`validate_task()`** - Validate task selection
- **`validate_dataset_config()`** - Validate dataset configuration
- **`validate_additional_request_params()`** - Validate request parameters

## Data

Components for loading and managing datasets.

### Configuration

- **`DatasetConfig`** - Configuration for dataset loading
- **`DatasetSourceConfig`** - Configuration for dataset sources

### Loaders

- **`DatasetLoader`** - Abstract base class for dataset loaders
- **`TextDatasetLoader`** - Text dataset loader
- **`ImageDatasetLoader`** - Image dataset loader
- **`DataLoaderFactory`** - Factory for creating data loaders

### Sources

- **`DatasetSource`** - Abstract base class for dataset sources
- **`FileDatasetSource`** - Local file dataset source
- **`HuggingFaceDatasetSource`** - HuggingFace Hub dataset source
- **`CustomDatasetSource`** - Custom dataset source
- **`DatasetSourceFactory`** - Factory for creating dataset sources

## Distributed

Components for distributed benchmark execution.

### Core Components

- **`DistributedRunner`** - Manages distributed load test execution
- **`DistributedConfig`** - Configuration for distributed runs
- **`MessageHandler`** - Protocol for message handling

### Architecture Features

- Master-worker architecture
- Message passing between processes
- Metrics aggregation
- Process management and cleanup

## Metrics

Components for collecting and analyzing performance metrics.

### Data Structures

- **`RequestLevelMetrics`** - Metrics for individual requests
- **`AggregatedMetrics`** - Aggregated metrics for entire runs
- **`MetricStats`** - Statistical metrics (mean, std, percentiles)

### Collectors

- **`AggregatedMetricsCollector`** - Collects and aggregates metrics
- **`RequestMetricsCollector`** - Collects per-request metrics

### Metric Types

- **Time Metrics**: TTFT (Time to First Token), TPOT (Time Per Output Token), E2E Latency
- **Throughput Metrics**: Input/Output throughput in tokens/second
- **Token Metrics**: Input/output token counts
- **Error Metrics**: Error rates and codes
- **Performance Metrics**: Requests per second, run duration

## Sampling

Components for sampling data and creating requests.

### Base Classes

- **`Sampler`** - Abstract base class for samplers

### Sampler Implementations

- **`TextSampler`** - Sampler for text-based tasks
- **`ImageSampler`** - Sampler for image-based tasks

### Supported Tasks

- **Text Tasks**: text-to-text, text-to-embeddings, text-to-rerank
- **Image Tasks**: image-text-to-text, image-to-embeddings

### Features

- Automatic task registry
- Modality-based sampling
- Dataset integration
- Request generation

## Scenarios

Components for defining traffic generation scenarios.

### Base Classes

- **`Scenario`** - Abstract base class for scenarios

### Scenario Implementations

- **`DatasetScenario`** - Dataset-based scenario
- **`NormalDistribution`** - Normal distribution scenario
- **`DeterministicDistribution`** - Deterministic scenario
- **`EmbeddingScenario`** - Embedding-specific scenario
- **`ReRankScenario`** - Re-ranking scenario
- **`ImageModality`** - Image modality scenario

### Distribution Types

- **`TextDistribution`** - NORMAL, DETERMINISTIC, UNIFORM
- **`EmbeddingDistribution`** - Embedding-specific distributions
- **`ReRankDistribution`** - Re-ranking distributions
- **`MultiModality`** - Multi-modal scenarios

### Features

- String-based scenario parsing
- Automatic scenario registry
- Parameter validation
- Distribution sampling

## UI

Components for user interface and visualization.

### Dashboard Implementations

- **`Dashboard`** - Union type for dashboard implementations
- **`RichLiveDashboard`** - Rich library-based dashboard
- **`MinimalDashboard`** - Minimal dashboard for non-UI scenarios

### Layout Functions

- **`create_layout()`** - Creates dashboard layout
- **`create_metric_panel()`** - Creates metric display panels
- **`create_progress_bars()`** - Creates progress tracking bars

### Visualization Functions

- **`create_horizontal_colored_bar_chart()`** - Creates histogram charts
- **`create_scatter_plot()`** - Creates scatter plots
- **`update_progress()`** - Updates progress displays

### Features

- Real-time metrics visualization
- Progress tracking
- Interactive charts and histograms
- Configurable UI components

## User

Components for interacting with different model APIs.

### Base Classes

- **`BaseUser`** - Abstract base class for user implementations

### User Implementations

- **`OpenAIUser`** - OpenAI API user
- **`AWSBedrockUser`** - AWS Bedrock user
- **`AzureOpenAIUser`** - Azure OpenAI user
- **`GCPVertexUser`** - GCP Vertex AI user
- **`OCICohereUser`** - OCI Cohere user
- **`OCIGenAIUser`** - OCI Generative AI user
- **`CohereUser`** - Cohere API user

### Supported Tasks

Each user implementation supports different combinations of:

- **text-to-text**: Chat and generation tasks
- **image-text-to-text**: Vision-based chat tasks
- **text-to-embeddings**: Text embedding generation
- **image-to-embeddings**: Image embedding generation
- **text-to-rerank**: Text re-ranking tasks

### Features

- Task-based request handling
- Metrics collection
- Error handling
- Authentication integration

## Example Usage

### Creating an Authentication Provider

```python
from genai_bench.auth.unified_factory import UnifiedAuthFactory

# Create OpenAI auth
auth = UnifiedAuthFactory.create_model_auth(
    "openai",
    api_key="sk-..."
)

# Create AWS Bedrock auth
auth = UnifiedAuthFactory.create_model_auth(
    "aws-bedrock",
    access_key_id="AKIA...",
    secret_access_key="...",
    region="us-east-1"
)
```

### Creating a Storage Provider

```python
from genai_bench.auth.unified_factory import UnifiedAuthFactory
from genai_bench.storage.factory import StorageFactory

# Create storage auth
storage_auth = UnifiedAuthFactory.create_storage_auth(
    "aws",
    profile="default",
    region="us-east-1"
)

# Create storage instance
storage = StorageFactory.create_storage(
    "aws",
    storage_auth
)

# Upload a folder
storage.upload_folder(
    "/path/to/results",
    "my-bucket",
    prefix="benchmarks/2024"
)
```

### Loading Datasets

```python
from genai_bench.data.config import DatasetConfig, DatasetSourceConfig
from genai_bench.data.loaders.factory import DataLoaderFactory

# Load from HuggingFace Hub
config = DatasetConfig(
    source=DatasetSourceConfig(
        type="huggingface",
        path="squad",
        huggingface_kwargs={"split": "train"}
    ),
    prompt_column="question"
)
data = DataLoaderFactory.load_data_for_task("text-to-text", config)

# Load from local CSV file
config = DatasetConfig(
    source=DatasetSourceConfig(
        type="file",
        path="/path/to/dataset.csv",
        file_format="csv"
    ),
    prompt_column="text"
)
data = DataLoaderFactory.load_data_for_task("text-to-text", config)
```

### Running Programmatic Benchmarks

```python
from genai_bench.distributed.runner import DistributedRunner, DistributedConfig
from genai_bench.ui.dashboard import create_dashboard

# Configure distributed execution
config = DistributedConfig(
    num_workers=4,
    master_host="127.0.0.1",
    master_port=5557
)

# Create dashboard
dashboard = create_dashboard(metrics_time_unit="s")

# Create and setup runner
runner = DistributedRunner(environment, config, dashboard)
runner.setup()

# Update scenario and run benchmark
runner.update_scenario("N(100,50)")
runner.update_batch_size(32)
```

### Analyzing Results

```python
from genai_bench.analysis.experiment_loader import load_multiple_experiments
from genai_bench.analysis.flexible_plot_report import FlexiblePlotGenerator
from genai_bench.analysis.plot_config import PlotConfig, PlotSpec

# Load experiment data
experiments = load_multiple_experiments(
    folder_name="/path/to/experiments",
    filter_criteria={"model": "gpt-4"}
)

# Create plot configuration
config = PlotConfig(
    title="Performance Analysis",
    plots=[
        PlotSpec(
            x_field="concurrency",
            y_fields=["e2e_latency", "ttft"],
            plot_type="line",
            title="Latency vs Concurrency"
        )
    ]
)

# Generate plots
generator = FlexiblePlotGenerator(config)
generator.generate_plots(
    experiments,
    group_key="traffic_scenario",
    experiment_folder="/path/to/results"
)
```

## Contributing to API Documentation

We welcome contributions to improve our API documentation! If you'd like to help:

1. Add docstrings to undocumented functions
2. Provide usage examples
3. Document edge cases and gotchas
4. Submit a pull request

See our [Development Guide](index.md) for more details.