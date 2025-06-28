# GenAI Bench

<img src="assets/logo.png" alt="GenAI Bench Logo" width="150">

**Unified, accurate, and beautiful LLM Benchmarking**

[![PyPI version](https://img.shields.io/pypi/v/genai-bench)](https://pypi.org/project/genai-bench/)
[![Python versions](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fsgl-project%2Fgenai-bench%2Fmain%2Fpyproject.toml)](https://github.com/sgl-project/genai-bench)
[![Types - Mypy](https://img.shields.io/badge/types-mypy-blue)](https://github.com/sgl-project/genai-bench)
[![Coverage - coverage](https://img.shields.io/badge/coverage-93%25-brightgreen)](https://github.com/sgl-project/genai-bench)
[![License](https://img.shields.io/github/license/sgl-project/genai-bench)](https://github.com/sgl-project/genai-bench/blob/main/LICENSE)

---

## What is GenAI Bench?

Genai-bench is a powerful benchmark tool designed for comprehensive token-level performance evaluation of large language model (LLM) serving systems.

It provides detailed insights into model serving performance, offering both a user-friendly CLI and a live UI for real-time progress monitoring.

## Live UI Dashboard

GenAI Bench includes a real-time dashboard that provides live monitoring of your benchmarks:

![GenAI Bench UI Dashboard](assets/ui_dashboard.png)

## Key Features

- 🛠️ **CLI Tool**: Validates user inputs and initiates benchmarks seamlessly.
- 📊 **Live UI Dashboard**: Displays current progress, logs, and real-time metrics.
- 📝 **Rich Logs**: Automatically flushed to both terminal and file upon experiment completion.
- 📈 **Experiment Analyzer**: Generates comprehensive Excel reports with pricing and raw metrics data, plus flexible plot configurations (default 2x4 grid) that visualize key performance metrics including throughput, latency (TTFT, E2E, TPOT), error rates, and RPS across different traffic scenarios and concurrency levels. Supports custom plot layouts and multi-line comparisons.

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
| `text-to-text` | Benchmarks generating text output from text input | Chat, QA |
| `text-to-embeddings` | Benchmarks generating embeddings from text input | Semantic search |
| `image-text-to-text` | Benchmarks generating text from images and text prompts | Visual question answering |
| `image-to-embeddings` | Benchmarks generating embeddings from images | Image similarity |

## Documentation Sections

### 🚀 Getting Started
- [Quick Start](getting-started/quick-start.md) - Get up and running in minutes
- [Installation](getting-started/installation.md) - Detailed installation guide
- [Configuration](getting-started/configuration.md) - Configure your environment

### 📖 User Guide
- [Overview](user-guide/overview.md) - Understanding GenAI Bench concepts
- [CLI Reference](user-guide/cli.md) - Complete command-line interface guide
- [Tasks & Benchmarks](user-guide/tasks.md) - Running different types of benchmarks
- [Analysis](user-guide/analysis.md) - Understanding your results

### 💡 Examples
- [Basic Benchmarks](examples/basic-benchmarks.md) - Simple benchmark examples

### 🔧 Development
- [Contributing](development/contributing.md) - How to contribute to GenAI Bench
- [Architecture](development/architecture.md) - Understanding the codebase
- [API Reference](api/overview.md) - Developer documentation

## Community

- **GitHub**: [sgl-project/genai-bench](https://github.com/sgl-project/genai-bench)
- **PyPI**: [genai-bench](https://pypi.org/project/genai-bench/)
- **Issues**: [GitHub Issues](https://github.com/sgl-project/genai-bench/issues)

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/sgl-project/genai-bench/blob/main/LICENSE) file for details. 