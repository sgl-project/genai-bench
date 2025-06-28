# API Overview

This section provides comprehensive documentation for GenAI Bench's APIs, interfaces, and extension points. Whether you're integrating GenAI Bench into your workflow or extending its functionality, this documentation will help you understand the available interfaces.

## API Categories

GenAI Bench provides several types of APIs:

1. **[CLI API](cli.md)** - Command-line interface for running benchmarks
2. **[Python API](python.md)** - Programmatic interface for Python applications
3. **[REST API](rest.md)** - HTTP API for remote access (when using web UI)
4. **[Extension APIs](extensions.md)** - Interfaces for extending functionality

## Core Concepts

### Benchmark Configuration

All APIs work with a common configuration structure:

```python
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class BenchmarkConfig:
    # API Configuration
    api_backend: str
    api_base: str
    api_key: str
    api_model_name: str
    
    # Task Configuration
    task: str
    dataset_name: str
    
    # Load Configuration
    num_users: int
    max_time_per_run: int
    max_requests_per_run: Optional[int] = None
    spawn_rate: Optional[float] = None
    
    # Output Configuration
    output_dir: str = "./results"
    ui: bool = False
    
    # Advanced Configuration
    additional_sampling_params: Optional[Dict[str, Any]] = None
    server_info: Optional[Dict[str, Any]] = None
```

### Task Types

GenAI Bench supports several task types:

```python
from enum import Enum

class TaskType(Enum):
    TEXT_TO_TEXT = "text-to-text"
    TEXT_TO_EMBEDDINGS = "text-to-embeddings"
    IMAGE_TEXT_TO_TEXT = "image-text-to-text"
    IMAGE_TO_EMBEDDINGS = "image-to-embeddings"
```

### Metrics Structure

All APIs return metrics in a consistent format:

```python
@dataclass
class BenchmarkMetrics:
    # Request-level metrics
    request_metrics: List[RequestMetric]
    
    # Aggregated metrics
    aggregated_metrics: AggregatedMetrics
    
    # System metrics
    system_metrics: SystemMetrics
    
    # Metadata
    benchmark_info: BenchmarkInfo
```

## Quick Start Examples

### Python API

```python
from genai_bench import Benchmark

# Create benchmark configuration
config = BenchmarkConfig(
    api_backend="openai",
    api_base="http://localhost:8082",
    api_key="your-api-key",
    api_model_name="llama-2-7b",
    task="text-to-text",
    dataset_name="sonnet.txt",
    num_users=2,
    max_time_per_run=60
)

# Run benchmark
benchmark = Benchmark(config)
results = benchmark.run()

# Access metrics
print(f"Average latency: {results.aggregated_metrics.avg_latency}")
print(f"Throughput: {results.aggregated_metrics.throughput}")
```

### CLI API

```bash
# Basic benchmark
genai-bench benchmark \
    --api-backend openai \
    --api-base "http://localhost:8082" \
    --api-key "your-api-key" \
    --api-model-name "llama-2-7b" \
    --task text-to-text \
    --dataset-name "sonnet.txt" \
    --num-users 2 \
    --max-time-per-run 60

# Export results
genai-bench export \
    --results-dir ./results \
    --output-format excel

# Generate plots
genai-bench plot \
    --results-dir ./results \
    --plot-type latency
```

### REST API

```bash
# Start benchmark via REST API
curl -X POST http://localhost:8080/api/v1/benchmarks \
  -H "Content-Type: application/json" \
  -d '{
    "config": {
      "api_backend": "openai",
      "api_base": "http://localhost:8082",
      "api_key": "your-api-key",
      "api_model_name": "llama-2-7b",
      "task": "text-to-text",
      "dataset_name": "sonnet.txt",
      "num_users": 2,
      "max_time_per_run": 60
    }
  }'

# Get benchmark status
curl http://localhost:8080/api/v1/benchmarks/{benchmark_id}/status

# Get results
curl http://localhost:8080/api/v1/benchmarks/{benchmark_id}/results
```

## API Stability

### Stability Levels

- **Stable**: APIs that are guaranteed to be backward compatible
- **Beta**: APIs that may change but with deprecation notices
- **Alpha**: Experimental APIs that may change without notice

### Current Stability

| API Component | Stability Level | Notes |
|---------------|-----------------|-------|
| CLI Commands | Stable | Core commands are stable |
| Python Core API | Stable | Main interfaces are stable |
| Configuration Schema | Stable | Core configuration is stable |
| Metrics Format | Beta | May add new metrics |
| REST API | Beta | Under active development |
| Extension APIs | Alpha | Subject to change |

## Versioning

GenAI Bench follows semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking changes to stable APIs
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

### API Versioning

- **CLI**: Version specified via `--version` flag
- **Python API**: Version available via `genai_bench.__version__`
- **REST API**: Version in URL path (`/api/v1/`)

## Authentication

### API Keys

All APIs support various authentication methods:

```python
# Environment variable
import os
os.environ["OPENAI_API_KEY"] = "your-key"

# Direct configuration
config.api_key = "your-key"

# File-based (for CLI)
# ~/.genai-bench/credentials
```

### Security Best Practices

1. **Never hardcode API keys** in source code
2. **Use environment variables** for credentials
3. **Rotate keys regularly** for production use
4. **Use least privilege** access for API keys

## Error Handling

### Error Types

All APIs use consistent error types:

```python
class GenAIBenchError(Exception):
    """Base exception for GenAI Bench."""
    pass

class ConfigurationError(GenAIBenchError):
    """Configuration-related errors."""
    pass

class APIError(GenAIBenchError):
    """API communication errors."""
    pass

class DatasetError(GenAIBenchError):
    """Dataset-related errors."""
    pass

class MetricsError(GenAIBenchError):
    """Metrics collection errors."""
    pass
```

### Error Response Format

```python
@dataclass
class ErrorResponse:
    error_type: str
    error_message: str
    error_code: int
    details: Optional[Dict[str, Any]] = None
    timestamp: str
    request_id: Optional[str] = None
```

## Rate Limiting

### Default Limits

- **CLI**: No built-in rate limiting (respects API limits)
- **Python API**: Configurable rate limiting
- **REST API**: 100 requests/minute per client

### Configuration

```python
# Python API rate limiting
config.rate_limit = RateLimitConfig(
    requests_per_second=10,
    burst_size=20
)
```

## Monitoring and Observability

### Logging

All APIs support structured logging:

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Enable debug logging
logging.getLogger('genai_bench').setLevel(logging.DEBUG)
```

### Metrics Export

```python
# Export metrics to various formats
benchmark.export_metrics(
    format="prometheus",
    endpoint="http://prometheus:9090"
)

benchmark.export_metrics(
    format="json",
    file_path="./metrics.json"
)
```

### Health Checks

```python
# Check system health
from genai_bench.health import HealthChecker

health = HealthChecker()
status = health.check_all()

if status.is_healthy:
    print("System is healthy")
else:
    print(f"Issues found: {status.issues}")
```

## Integration Examples

### CI/CD Integration

```yaml
# GitHub Actions example
- name: Run GenAI Bench
  run: |
    genai-bench benchmark \
      --config ./benchmark-config.yaml \
      --output-dir ./results
    
    genai-bench export \
      --results-dir ./results \
      --output-format json \
      --output-file ./benchmark-results.json
```

### Docker Integration

```dockerfile
FROM python:3.11-slim

RUN pip install genai-bench

COPY benchmark-config.yaml /config/
COPY datasets/ /datasets/

CMD ["genai-bench", "benchmark", "--config", "/config/benchmark-config.yaml"]
```

### Kubernetes Integration

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: genai-bench-job
spec:
  template:
    spec:
      containers:
      - name: genai-bench
        image: genai-bench:latest
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: openai-key
        command: ["genai-bench", "benchmark"]
        args: ["--config", "/config/benchmark.yaml"]
      restartPolicy: Never
```

## Performance Considerations

### Resource Usage

- **Memory**: ~100MB base + ~1MB per concurrent user
- **CPU**: Scales with number of users and request rate
- **Network**: Depends on request/response sizes
- **Disk**: Results storage scales with benchmark duration

### Optimization Tips

1. **Use appropriate user counts** for your hardware
2. **Configure timeouts** to prevent hanging requests
3. **Monitor resource usage** during benchmarks
4. **Use distributed mode** for large-scale testing

## Migration Guide

### From CLI to Python API

```python
# CLI command
# genai-bench benchmark --api-backend openai --num-users 5

# Equivalent Python API
config = BenchmarkConfig(
    api_backend="openai",
    num_users=5,
    # ... other parameters
)
benchmark = Benchmark(config)
results = benchmark.run()
```

### From v1 to v2 (when available)

Migration guides will be provided for major version updates.

## Support and Community

### Getting Help

1. **Documentation**: Check this documentation first
2. **GitHub Issues**: Report bugs and request features
3. **GitHub Discussions**: Ask questions and share ideas
4. **Examples**: Check the examples directory

### Contributing

See the [Contributing Guide](../development/contributing.md) for information on:

- Setting up development environment
- Code standards and testing
- Submitting pull requests
- API design guidelines

## API Reference Links

- **[CLI API Reference](cli.md)** - Complete CLI command documentation
- **[Python API Reference](python.md)** - Python classes and methods
- **[REST API Reference](rest.md)** - HTTP endpoints and schemas
- **[Extension API Reference](extensions.md)** - Extension interfaces

## Changelog

### v0.1.0 (Current)

- Initial API release
- Core CLI and Python APIs
- Basic metrics and export functionality
- OpenAI and Cohere backend support

### Upcoming Features

- Enhanced REST API
- Plugin system for extensions
- Advanced metrics and analysis
- Distributed benchmarking improvements 