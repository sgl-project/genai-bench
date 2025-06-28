# Quick Start Guide

Get up and running with GenAI Bench in minutes! This guide will walk you through the essential steps to run your first benchmark.

## Prerequisites

- Python 3.11 or 3.12
- An LLM serving endpoint (e.g., vLLM, OpenAI API, etc.)
- Basic familiarity with command-line tools

## Installation

### Option 1: Install from PyPI (Recommended)

```bash
pip install genai-bench
```

### Option 2: Development Setup

```bash
# Clone the repository
git clone https://github.com/sgl-project/genai-bench.git
cd genai-bench

# Set up virtual environment
make uv
source .venv/bin/activate

# Install in editable mode
make install
```

## Your First Benchmark

Let's run a simple text-to-text benchmark to get familiar with GenAI Bench:

### 1. Basic Chat Benchmark

```bash
genai-bench benchmark \
    --api-backend openai \
    --api-base "http://localhost:8082" \
    --api-key "your-api-key" \
    --api-model-name "your-model" \
    --task text-to-text \
    --max-time-per-run 60 \
    --max-requests-per-run 100
```

### 2. Monitor Your Benchmark

GenAI Bench provides a live UI dashboard. Open your browser and navigate to:
```
http://localhost:8089
```

You'll see real-time progress, logs, and metrics as your benchmark runs.

### 3. Analyze Results

After the benchmark completes, generate an Excel report:

```bash
genai-bench excel --experiment-dir ./experiments/latest
```

## Key Concepts

### Tasks

GenAI Bench supports different types of benchmarks:

- **`text-to-text`**: Generate text responses (chat, QA)
- **`text-to-embeddings`**: Generate embeddings from text
- **`image-text-to-text`**: Generate text from images + text
- **`image-to-embeddings`**: Generate embeddings from images

### API Backends

Supported API backends:

- **`openai`**: OpenAI-compatible APIs (vLLM, OpenAI, etc.)
- **`cohere`**: Cohere API
- **`oci_cohere`**: Oracle Cloud Infrastructure Cohere

### Traffic Scenarios

GenAI Bench supports various traffic patterns:

- **`constant`**: Steady request rate
- **`burst`**: Sudden traffic spikes
- **`ramp`**: Gradually increasing load

## Next Steps

- Read the [Installation Guide](installation.md) for detailed setup instructions
- Explore [Tasks and Benchmarks](../user-guide/tasks.md) for advanced usage
- Check out [Examples](../examples/basic-benchmarks.md) for more scenarios
- Learn about [Results Analysis](../user-guide/analysis.md) to understand your metrics

## Getting Help

- Use `genai-bench --help` for command-line help
- Check the [User Guide](../user-guide/overview.md) for detailed documentation
- Open an [issue](https://github.com/sgl-project/genai-bench/issues) for bugs or questions 