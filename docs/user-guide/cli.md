# Command Line Interface Reference

This guide provides a complete reference for all GenAI Bench command-line interface commands and options.

## Overview

GenAI Bench provides three main commands:

```bash
genai-bench [OPTIONS] COMMAND [ARGS]...
```

### Available Commands

- `benchmark` - Run a benchmark based on user-defined scenarios
- `excel` - Export experiment results to an Excel file
- `plot` - Generate plots from experiment results

## Global Options

```bash
genai-bench [OPTIONS] COMMAND [ARGS]...

Options:
  --version    Show the version and exit.
  --help       Show this message and exit.
```

## Benchmark Command

The `benchmark` command is the main command for running performance benchmarks.

### Basic Syntax

```bash
genai-bench benchmark [OPTIONS]
```

### Required Options

| Option | Type | Description | Example |
|--------|------|-------------|---------|
| `--api-backend` | string | API backend type | `openai`, `cohere`, `oci_cohere` |
| `--api-base` | string | API base URL | `http://localhost:8082` |
| `--api-key` | string | API authentication key | `sk-...` |
| `--api-model-name` | string | Model name identifier | `gpt-3.5-turbo` |
| `--task` | string | Benchmark task type | `text-to-text` |

### API Configuration Options

#### OpenAI-compatible Backends

```bash
--api-backend openai
--api-base "http://localhost:8082"
--api-key "your-api-key"
--api-model-name "your-model"
```

#### Cohere Backend

```bash
--api-backend cohere
--api-base "https://api.cohere.ai"
--api-key "your-cohere-key"
--api-model-name "command-r-plus"
```

#### OCI Cohere Backend

```bash
--api-backend oci_cohere
--api-base "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
--api-key "your-oci-key"
--api-model-name "cohere.command-r-plus"
```

### Task Configuration

#### Supported Tasks

| Task | Description | Use Case |
|------|-------------|----------|
| `text-to-text` | Generate text from text input | Chat, QA, summarization |
| `text-to-embeddings` | Generate embeddings from text | Semantic search |
| `image-text-to-text` | Generate text from images + text | Visual QA |
| `image-to-embeddings` | Generate embeddings from images | Image similarity |

#### Task Examples

```bash
# Text-to-text benchmark
--task text-to-text

# Embeddings benchmark
--task text-to-embeddings

# Vision benchmark
--task image-text-to-text
```

### Traffic Configuration

#### Traffic Scenarios

| Scenario | Description | Use Case |
|----------|-------------|----------|
| `constant` | Steady request rate | Baseline performance |
| `burst` | Sudden traffic spikes | Stress testing |
| `ramp` | Gradually increasing load | Capacity testing |

#### Concurrency and Load

```bash
# Set number of concurrent users
--num-users 10

# Set request rate (requests per second)
--spawn-rate 5

# Set maximum time per run (seconds)
--max-time-per-run 300

# Set maximum requests per run
--max-requests-per-run 1000
```

### Data Configuration

#### Dataset Options

```bash
# Use built-in dataset
--dataset-name "sonnet.txt"

# Use custom dataset file
--dataset-path "/path/to/your/dataset.txt"

# Use Hugging Face dataset
--dataset-name "your-dataset"
--dataset-split "train"
```

#### Tokenizer Configuration

```bash
# Use model's tokenizer
--model-tokenizer "/path/to/tokenizer"

# Use Hugging Face tokenizer
--model-tokenizer "meta-llama/Llama-2-7b-chat-hf"
```

### Advanced Options

#### Server Information

```bash
# Server engine type
--server-engine "vLLM"

# GPU type
--server-gpu-type "H100"

# Model size
--server-model-size "7B"
```

#### Monitoring and Logging

```bash
# Enable UI dashboard
--ui

# Set log level
--log-level INFO

# Output directory
--output-dir "./experiments"
```

### Complete Example

```bash
genai-bench benchmark \
    --api-backend openai \
    --api-base "http://localhost:8082" \
    --api-key "your-api-key" \
    --api-model-name "llama-2-7b" \
    --task text-to-text \
    --num-users 5 \
    --spawn-rate 2 \
    --max-time-per-run 120 \
    --max-requests-per-run 500 \
    --dataset-name "sonnet.txt" \
    --model-tokenizer "/path/to/tokenizer" \
    --server-engine "vLLM" \
    --server-gpu-type "H100" \
    --ui \
    --output-dir "./my-experiment"
```

## Excel Command

Generate Excel reports from benchmark results.

### Basic Syntax

```bash
genai-bench excel [OPTIONS]
```

### Options

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--experiment-dir` | string | Experiment directory | `./experiments/latest` |
| `--output-file` | string | Output Excel file | `experiment_results.xlsx` |
| `--include-raw-data` | flag | Include raw metrics data | `False` |
| `--include-pricing` | flag | Include pricing calculations | `True` |

### Example

```bash
genai-bench excel \
    --experiment-dir "./experiments/my-benchmark" \
    --output-file "results.xlsx" \
    --include-raw-data \
    --include-pricing
```

## Plot Command

Generate visualizations from benchmark results.

### Basic Syntax

```bash
genai-bench plot [OPTIONS]
```

### Options

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--experiment-dirs` | string | Comma-separated experiment directories | `./experiments/latest` |
| `--output-file` | string | Output plot file | `experiment_plots.png` |
| `--plot-config` | string | Plot configuration file | Default 2x4 grid |
| `--filters` | string | Filter criteria | None |
| `--group-by` | string | Grouping criteria | None |

### Plot Configuration

Create a custom plot configuration file:

```yaml
# plot_config.yaml
layout:
  rows: 2
  cols: 4

plots:
  - title: "Throughput vs Concurrency"
    type: "line"
    x_axis: "num_users"
    y_axis: "mean_total_tokens_throughput"
    
  - title: "Latency vs Concurrency"
    type: "line"
    x_axis: "num_users"
    y_axis: "mean_e2e_latency"
```

### Example

```bash
genai-bench plot \
    --experiment-dirs "./exp1,./exp2,./exp3" \
    --output-file "comparison.png" \
    --plot-config "my_config.yaml" \
    --filters "server_engine=vLLM" \
    --group-by "num_users"
```

## Environment Variables

Set these environment variables for configuration:

```bash
# Hugging Face token (for tokenizer downloads)
export HF_TOKEN="your-huggingface-token"

# Disable torch warnings
export TRANSFORMERS_VERBOSITY=error

# Log level
export GENAI_BENCH_LOG_LEVEL=INFO

# API keys
export OPENAI_API_KEY="your-openai-key"
export COHERE_API_KEY="your-cohere-key"
```

## Getting Help

### Command Help

```bash
# General help
genai-bench --help

# Benchmark command help
genai-bench benchmark --help

# Excel command help
genai-bench excel --help

# Plot command help
genai-bench plot --help
```

### Interactive Mode

For guided setup, run without options:

```bash
genai-bench benchmark
```

This will prompt you for required parameters interactively.

## Common Patterns

### Quick Benchmark

```bash
genai-bench benchmark \
    --api-backend openai \
    --api-base "http://localhost:8082" \
    --api-key "your-key" \
    --api-model-name "your-model" \
    --task text-to-text \
    --max-time-per-run 60
```

### Production Benchmark

```bash
genai-bench benchmark \
    --api-backend openai \
    --api-base "https://your-api.com" \
    --api-key "your-key" \
    --api-model-name "gpt-4" \
    --task text-to-text \
    --num-users 20 \
    --spawn-rate 5 \
    --max-time-per-run 600 \
    --max-requests-per-run 2000 \
    --dataset-name "production-data" \
    --ui \
    --output-dir "./production-benchmark"
```

### Comparison Benchmark

```bash
# Run multiple experiments
for model in "model1" "model2" "model3"; do
    genai-bench benchmark \
        --api-model-name "$model" \
        --output-dir "./experiments/$model" \
        --max-time-per-run 300
done

# Generate comparison plot
genai-bench plot \
    --experiment-dirs "./experiments/model1,./experiments/model2,./experiments/model3" \
    --output-file "model_comparison.png"
``` 