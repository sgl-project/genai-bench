# Welcome to GenAI Bench

<div align="center">

![GenAI Bench Logo](assets/logo.png){ width="200" }

**Unified, accurate, and beautiful LLM Benchmarking**

[![PyPI version](https://img.shields.io/pypi/v/genai-bench)](https://pypi.org/project/genai-bench/)
[![Python versions](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fsgl-project%2Fgenai-bench%2Fmain%2Fpyproject.toml)](https://github.com/sgl-project/genai-bench)
[![Types - Mypy](https://img.shields.io/badge/types-mypy-blue)](https://github.com/sgl-project/genai-bench)
[![Coverage - coverage](https://img.shields.io/badge/coverage-93%25-brightgreen)](https://github.com/sgl-project/genai-bench)
[![License](https://img.shields.io/github/license/sgl-project/genai-bench)](https://github.com/sgl-project/genai-bench/blob/main/LICENSE)

</div>

## What is GenAI Bench?

GenAI Bench is a powerful benchmark tool designed for comprehensive token-level performance evaluation of large language model (LLM) serving systems. It provides detailed insights into model serving performance, offering both a user-friendly CLI and a live UI for real-time progress monitoring.

## Key Features

<div class="grid" markdown>

<div class="cell" markdown>

### 🛠️ CLI Tool
Validates user inputs and initiates benchmarks seamlessly with comprehensive command-line options.

</div>

<div class="cell" markdown>

### 📊 Live UI Dashboard
Displays current progress, logs, and real-time metrics for monitoring benchmark execution.

</div>

<div class="cell" markdown>

### 📝 Rich Logging
Automatically flushed to both terminal and file upon experiment completion with detailed metrics.

</div>

<div class="cell" markdown>

### 📈 Advanced Analysis
Generates comprehensive Excel reports with pricing and raw metrics data, plus flexible plot configurations.

</div>

</div>

## Quick Start

Get started with GenAI Bench in minutes:

```bash
# Install from PyPI
pip install genai-bench

# Run your first benchmark
genai-bench benchmark --help
```

For detailed installation and usage instructions, see our [Getting Started Guide](getting-started/quick-start.md).

## Supported Tasks

GenAI Bench supports multiple benchmark types:

| Task | Description | Use Case |
|------|-------------|----------|
| `text-to-text` | Generate text from text input | Chat, QA, summarization |
| `text-to-embeddings` | Generate embeddings from text | Semantic search, similarity |
| `image-text-to-text` | Generate text from images + text | Visual QA, image understanding |
| `image-to-embeddings` | Generate embeddings from images | Image similarity, search |

## Documentation Sections

<div class="grid" markdown>

<div class="cell" markdown>

### 🚀 Getting Started
- [Quick Start](getting-started/quick-start.md) - Get up and running in minutes
- [Installation](getting-started/installation.md) - Detailed installation guide
- [Configuration](getting-started/configuration.md) - Configure your environment

</div>

<div class="cell" markdown>

### 📖 User Guide
- [Overview](user-guide/overview.md) - Understanding GenAI Bench concepts
- [CLI Reference](user-guide/cli.md) - Complete command-line interface guide
- [Tasks & Benchmarks](user-guide/tasks.md) - Running different types of benchmarks
- [Analysis](user-guide/analysis.md) - Understanding your results

</div>

<div class="cell" markdown>

### 💡 Examples
- [Basic Benchmarks](examples/basic-benchmarks.md) - Simple benchmark examples
- [Advanced Configurations](examples/advanced-configs.md) - Complex scenarios
- [Plot Configurations](examples/plot-configs.md) - Customizing visualizations

</div>

<div class="cell" markdown>

### 🔧 Development
- [Contributing](development/contributing.md) - How to contribute to GenAI Bench
- [Architecture](development/architecture.md) - Understanding the codebase
- [API Reference](api/core.md) - Developer documentation

</div>

</div>

## Community

- **GitHub**: [sgl-project/genai-bench](https://github.com/sgl-project/genai-bench)
- **PyPI**: [genai-bench](https://pypi.org/project/genai-bench/)
- **Issues**: [GitHub Issues](https://github.com/sgl-project/genai-bench/issues)

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/sgl-project/genai-bench/blob/main/LICENSE) file for details. 