# CLI Guidelines

GenAI Bench provides three main CLI commands for running benchmarks, generating reports, and creating visualizations. This guide covers the options for each command.

## Overview

```shell
Commands:
  benchmark  Run a benchmark based on user defined scenarios.
  excel      Exports the experiment results to an Excel file.
  plot       Plots the experiment(s) results based on filters and group...
```

## Benchmark

The `benchmark` command runs performance tests against AI models. It's the core command for executing benchmarks.

### Example Usage
```bash
# Start a chat benchmark
genai-bench benchmark --api-backend openai \
            --api-base "http://localhost:8082" \
            --api-key "your-openai-api-key" \
            --api-model-name "meta-llama/Meta-Llama-3-70B-Instruct" \
            --model-tokenizer "/mnt/data/models/Meta-Llama-3.1-70B-Instruct" \
            --task text-to-text \
            --max-time-per-run 15 \
            --max-requests-per-run 300 \
            --server-engine "SGLang" \
            --server-gpu-type "H100" \
            --server-version "v0.6.0" \
            --server-gpu-count 4
```

### Options

#### **API Configuration** (Required)
- `--api-backend` - Choose your model provider (openai, oci-cohere, aws-bedrock, azure-openai, gcp-vertex, vllm, sglang) **(required)**
- `--api-base` - API endpoint URL **(required)**
- `--api-model-name` - Model name for the request body **(required)**
- `--task` - Task type (text-to-text, text-to-embeddings, image-text-to-text, etc.) **(required)**

#### **Authentication**
- `--api-key` - API key (for OpenAI)
- `--model-api-key` - Alternative API key parameter
- Cloud-specific auth options (AWS, Azure, GCP, OCI)

#### **Experiment Parameters** (Required)
- `--max-requests-per-run` - Maximum requests to send each run **(required)**
- `--max-time-per-run` - Maximum duration for each run in minutes **(required)**
- `--model-tokenizer` - Path to the model tokenizer **(required)**
- `--num-concurrency` - Number of concurrent requests to send (multiple values supported in different runs)
- `--batch-size` - Batch sizes for embeddings/rerank tasks
- `--traffic-scenario` - Define input/output token distributions, more info in [Traffic Scenarios](../user-guide/scenario-definition.md)

#### **Dataset Options**
- `--dataset-path` - Path to dataset (local file, HuggingFace ID, or 'default')
- `--dataset-config` - JSON config file for advanced dataset options, more info in [Selecting Datasets](../user-guide/run-benchmark.md/#selecting-datasets)
- `--dataset-prompt-column` - Column name for prompts
- `--dataset-image-column` - Column name for images (multimodal)

#### **Server Information**
- `--server-engine` - Backend engine (vLLM, SGLang, TGI, etc.)
- `--server-version` - Server version
- `--server-gpu-type` - GPU type (H100, A100-80G, etc.)
- `--server-gpu-count` - Number of GPUs

For more information and examples, check out [Run Benchmark](../user-guide/run-benchmark.md).

## Excel

The `excel` command exports experiment results to Excel spreadsheets for detailed analysis.

### Example Usage

```bash
# Export with mean metrics in seconds
genai-bench excel \
  --experiment-folder ./experiments/openai_gpt-3.5-turbo_20241201_120000 \
  --excel-name benchmark_results \
  --metric-percentile mean \
  --metrics-time-unit s

# Export with 95th percentile in milliseconds
genai-bench excel \
  --experiment-folder ./experiments/my_experiment \
  --excel-name detailed_analysis \
  --metric-percentile p95 \
  --metrics-time-unit ms
```

### Options

- `--experiment-folder` - Path to experiment results folder **(required)**
- `--excel-name` - Name for the output Excel file **(required)**
- `--metric-percentile` - Statistical percentile (mean, p25, p50, p75, p90, p95, p99) to select from
- `--metrics-time-unit [s|ms]` - Time unit to use when showing latency metrics in the spreadsheet. Defaults to seconds

## Plot

The `plot` command generates visualizations from experiment data with flexible configuration options.

### Example Usage

```bash
# Simple plot with default 2x4 layout
genai-bench plot \
  --experiments-folder ./experiments \
  --group-key traffic_scenario \
  --filter-criteria "{'model': 'gpt-3.5-turbo'}"

# Use built-in preset for latency analysis
genai-bench plot \
  --experiments-folder ./experiments \
  --group-key server_version \
  --preset multi_line_latency \
  --metrics-time-unit ms
```

### Options

- `--experiments-folder` - Path to experiments folder, can be more than one experiment **(required)**
- `--group-key` - Key to group data by (e.g., 'traffic_scenario', 'server_version', 'none') **(required)**
- `--filter-criteria` - Dictionary of filter criteria
- `--plot-config` - Path to JSON plot configuration file. For more information use [Advanced Plot Configuration](../user-guide/generate-plot.md/#advanced-plot-configuration)
- `--preset` - Built-in plot presets (2x4_default, simple_2x2, multi_line_latency, single_scenario_analysis). Overrides `--plot-config` if both given
- `--metrics-time-unit [s|ms]` - Time unit for latency display, defaults to seconds

### Advanced Options

- `--list-fields` - List available data fields and exit
- `--validate-only` - Validate configuration without generating plots
- `--verbose` - Enable detailed logging

For more information and examples, check out [Generate Plot](../user-guide/generate-plot.md).

## Getting Help

For detailed help on any command:

```bash
genai-bench --help
genai-bench benchmark --help
genai-bench excel --help
genai-bench plot --help
```

For further information, refer to the [User Guide](../user-guide/index.md). You can also look at [option_groups.py](https://github.com/sgl-project/genai-bench/blob/main/genai_bench/cli/option_groups.py) directly.