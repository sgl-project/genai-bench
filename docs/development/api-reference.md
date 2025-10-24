# API Reference

This section provides detailed API documentation for GenAI Bench components.

!!! info "Coming Soon"
    Comprehensive API documentation is being developed. In the meantime, please refer to the source code docstrings.

## Core Components

### Authentication

- **UnifiedAuthFactory** - Factory for creating authentication providers
- **ModelAuthProvider** - Base class for model authentication
- **StorageAuthProvider** - Base class for storage authentication

### Storage

- **BaseStorage** - Abstract base class for storage implementations
- **StorageFactory** - Factory for creating storage providers

### CLI

- **option_groups** - Modular CLI option definitions
- **validation** - Input validation functions

### Metrics

- **AggregatedMetricsCollector** - Collects and aggregates benchmark metrics
- **RequestMetricsCollector** - Collects per-request metrics

### Data Loading

- **DatasetConfig** - Configuration for dataset loading
- **DatasetSourceConfig** - Configuration for dataset sources
- **DataLoaderFactory** - Factory for loading datasets
- **TextDatasetLoader** - Text dataset loader
- **ImageDatasetLoader** - Image dataset loader

### Benchmarking

- **DistributedRunner** - Distributed benchmark execution with multiple workers
- **DistributedConfig** - Configuration for distributed runs
- **BaseUser** - Abstract base class for user implementations
- **OpenAIUser** - OpenAI API implementation
- **AWSBedrockUser** - AWS Bedrock implementation
- **AzureOpenAIUser** - Azure OpenAI implementation
- **GCPVertexUser** - GCP Vertex AI implementation
- **OCICohereUser** - OCI Cohere implementation

### Analysis

- **PlotConfig** - Configuration for visualizations
- **ExperimentLoader** - Loading experiment results
- **FlexiblePlotGenerator** - Generate plots with flexible configuration

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
