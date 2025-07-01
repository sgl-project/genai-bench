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

### User Classes

- **BaseUser** - Abstract base class for user implementations
- **OpenAIUser** - OpenAI API implementation
- **AWSBedrockUser** - AWS Bedrock implementation
- **AzureOpenAIUser** - Azure OpenAI implementation
- **GCPVertexUser** - GCP Vertex AI implementation
- **OCICohereUser** - OCI Cohere implementation

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

## Contributing to API Documentation

We welcome contributions to improve our API documentation! If you'd like to help:

1. Add docstrings to undocumented functions
2. Provide usage examples
3. Document edge cases and gotchas
4. Submit a pull request

See our [Contributing Guide](../development/contributing.md) for more details.