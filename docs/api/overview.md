# API Overview

This section provides comprehensive documentation for GenAI Bench's APIs, interfaces, and extension points. Whether you're integrating GenAI Bench into your workflow or extending its functionality, this documentation will help you understand the available interfaces.

## API Categories

GenAI Bench provides several types of APIs:

1. **CLI API** - Command-line interface for running benchmarks (see [CLI Reference](../user-guide/cli.md))
2. **Python API** - Programmatic interface for Python applications
3. **REST API** - HTTP API for remote access (when using web UI)
4. **Extension APIs** - Interfaces for extending functionality

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
genai-bench excel \
    --experiment-dir ./experiments/latest \
    --output-file results.xlsx

# Generate plots
genai-bench plot \
    --experiment-dirs ./experiments/latest \
    --output-file results.png
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

GenAI Bench supports various authentication methods:

```python
# Environment variable (recommended)
import os
config.api_key = os.getenv("OPENAI_API_KEY")

# Direct assignment
config.api_key = "your-api-key"

# File-based
with open("api_key.txt") as f:
    config.api_key = f.read().strip()
```

### Security Best Practices

1. **Never hardcode API keys** in source code
2. **Use environment variables** for sensitive data
3. **Rotate keys regularly** for production systems
4. **Limit key permissions** where possible

## Error Handling

### Common Error Types

```python
from genai_bench.exceptions import (
    BenchmarkError,
    ConfigurationError,
    APIError,
    DatasetError
)

try:
    results = benchmark.run()
except ConfigurationError as e:
    print(f"Configuration error: {e}")
except APIError as e:
    print(f"API error: {e}")
except DatasetError as e:
    print(f"Dataset error: {e}")
except BenchmarkError as e:
    print(f"Benchmark error: {e}")
```

### Error Response Format

```json
{
  "error": {
    "type": "ConfigurationError",
    "message": "Invalid API endpoint",
    "details": {
      "field": "api_base",
      "value": "invalid-url"
    }
  }
}
```

## Integration Examples

### CI/CD Integration

```yaml
# GitHub Actions example
name: Performance Benchmark
on: [push]
jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run Benchmark
        run: |
          pip install genai-bench
          genai-bench benchmark \
            --api-backend openai \
            --api-base "${{ secrets.API_BASE }}" \
            --api-key "${{ secrets.API_KEY }}" \
            --api-model-name "test-model" \
            --task text-to-text \
            --num-users 2 \
            --max-time-per-run 60
```

### Docker Integration

```dockerfile
FROM python:3.11-slim

RUN pip install genai-bench

COPY benchmark-config.yaml /app/
WORKDIR /app

CMD ["genai-bench", "benchmark", "--config", "benchmark-config.yaml"]
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
      - name: benchmark
        image: genai-bench:latest
        env:
        - name: API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: openai-key
        command: ["genai-bench", "benchmark"]
        args: [
          "--api-backend", "openai",
          "--api-base", "http://model-service:8080",
          "--task", "text-to-text",
          "--num-users", "10",
          "--max-time-per-run", "300"
        ]
      restartPolicy: Never
```

## Performance Considerations

### Resource Usage

- **Memory**: Scales with number of concurrent users and dataset size
- **CPU**: Depends on tokenization and metrics processing
- **Network**: Proportional to request rate and payload size

### Optimization Tips

1. **Use appropriate concurrency** for your target system
2. **Monitor resource usage** during benchmarks
3. **Batch requests** when possible
4. **Cache tokenizers** for repeated use

## Troubleshooting

### Common Issues

1. **Connection timeouts**: Check network connectivity and API endpoint
2. **Authentication errors**: Verify API keys and permissions
3. **Rate limiting**: Reduce concurrency or add delays
4. **Memory issues**: Reduce dataset size or concurrency

### Debug Mode

```bash
# Enable debug logging
genai-bench benchmark --log-level DEBUG

# Save detailed logs
genai-bench benchmark --log-file benchmark.log
```

### Getting Help

- Check the [User Guide](../user-guide/overview.md) for detailed documentation
- Review [Examples](../examples/basic-benchmarks.md) for practical use cases
- Open an [issue](https://github.com/sgl-project/genai-bench/issues) for bugs or questions

## Next Steps

- Explore the [CLI Reference](../user-guide/cli.md) for command-line usage
- Read about [Tasks and Benchmarks](../user-guide/tasks.md) for different benchmark types
- Check out [Examples](../examples/basic-benchmarks.md) for practical scenarios 