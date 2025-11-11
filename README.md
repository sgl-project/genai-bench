<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/sgl-project/genai-bench">
    <img src="https://raw.githubusercontent.com/sgl-project/genai-bench/main/docs/assets/logo.png" alt="Logo" width="" height="150">
  </a>

<h3 align="center">
Unified, accurate, and beautiful LLM Benchmarking
</h3>

<div align="center">

[![PyPI version](https://img.shields.io/pypi/v/genai-bench)](https://pypi.org/project/genai-bench/)
[![Python versions](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fsgl-project%2Fgenai-bench%2Fmain%2Fpyproject.toml)](https://github.com/sgl-project/genai-bench)
[![Types - Mypy](https://img.shields.io/badge/types-mypy-blue)](https://github.com/sgl-project/genai-bench)
[![Coverage - coverage](https://img.shields.io/badge/coverage-93%25-brightgreen)](https://github.com/sgl-project/genai-bench)
[![License](https://img.shields.io/github/license/sgl-project/genai-bench)](https://github.com/sgl-project/genai-bench/blob/main/LICENSE)

</div>

<p align="center">
| <a href="https://docs.sglang.ai/genai-bench/user-guide/"><b>User Guide</b></a> | <a href="https://docs.sglang.ai/genai-bench/development/contributing/"><b>Contribution Guideline</b></a> |
</p>

</div>

<p align="center"><img src="https://raw.githubusercontent.com/sgl-project/genai-bench/main/docs/assets/ui_dashboard.png" alt="UI" width="1000" height=""></p>

## Introduction

Genai-bench is a powerful benchmark tool designed for comprehensive token-level performance evaluation of large language model (LLM) serving systems.

It provides detailed insights into model serving performance, offering both a user-friendly CLI and a live UI for real-time progress monitoring.

## Features

- 🛠️ **CLI Tool**: Validates user inputs and initiates benchmarks seamlessly.
- 📊 **Live UI Dashboard**: Displays current progress, logs, and real-time metrics.
- 📝 **Rich Logs**: Automatically flushed to both terminal and file upon experiment completion.
- 📈 **Experiment Analyzer**: Generates comprehensive Excel reports with pricing and raw metrics data, plus flexible plot configurations (default 2x4 grid) that visualize key performance metrics including throughput, latency (TTFT, E2E, TPOT), error rates, and RPS across different traffic scenarios and concurrency levels. Supports custom plot layouts and multi-line comparisons.

- 🧪 **Synthetic Tore-style prompts (optional)**: Generate synthetic requests that mimic tore-speed’s synthetic dataset prep, including a cached prefix region and exact input/output token counts for precise performance experiments.

### Open-loop QPS mode (non-Locust)

- Enable with `--non-locust` to use an open-loop arrival process (tore-speed style). Arrivals are scheduled globally by inter-arrival intervals; completions may lag depending on server speed.
- Use `--qps-level` (repeatable; floats allowed) to specify QPS levels and `--qps-distribution` (uniform|exponential|constant) for inter-arrival sampling.
- Duration of each level comes from `--max-time-per-run` (in minutes; floats allowed). Internally converted to seconds.
- Example (tore-speed compatible synthetic run):

```bash
genai-bench benchmark \
  --non-locust \
  --qps-level 0.1 --qps-level 0.3 \
  --qps-distribution uniform \
  --max-requests-per-run 1500 --max-time-per-run 2 \
  --api-backend together --api-base https://api.together.xyz \
  --api-model-name <model> --model-tokenizer <hf-tokenizer> \
  --task text-to-text \
  --traffic-scenario "D(10000,825)" \
  --synthetic --synthetic-cached-input-length 3000
```

Notes:
- Arrival rate (QPS) is the planned schedule; observed RPS depends on completions within the time window.
- In synthetic mode, dataset file loading is skipped; prompts are constructed to exact token counts with a cached prefix region matching tore-speed semantics.

## How to Start

Please check [User Guide](https://docs.sglang.ai/genai-bench/user-guide/) and [CONTRIBUTING.md](https://docs.sglang.ai/genai-bench/development/contributing/) for how to install and use genai-bench.

### Synthetic data mode (tore-speed compatible)

Genai-bench can synthesize prompts similar to tore-speed’s `--dataset_type synthetic`, with a fixed-size cached prefix and exact token counts enforced at the tokenizer level.

- Enable with the `--synthetic` flag and provide a deterministic traffic scenario for input/output tokens (e.g., `D(10000,825)`).
- Specify the cached prefix size (in tokens) with `--synthetic-cached-input-length`.

Example (concurrency mode):

```bash
genai-bench benchmark \
  --api-backend together \
  --api-base https://api.together.xyz \
  --api-model-name <model> \
  --model-tokenizer <hf-tokenizer> \
  --task text-to-text \
  --traffic-scenario "D(10000,825)" \
  --max-requests-per-run 1500 --max-time-per-run 2 \
  --num-concurrency 128 --spawn-rate 128 \
  --synthetic --synthetic-cached-input-length 3000 \
  --additional-request-params '{"stream": true}'
```

Notes:
- The sampler ensures the prompt contains exactly the requested number of input tokens. The leading `--synthetic-cached-input-length` tokens are filled with a repeated base phrase to emulate a cacheable prefix; a unique marker and a long instruction are appended to the uncached suffix region.
- This is useful for cache stress tests and apples-to-apples comparisons with tore-speed’s synthetic mode.

## Benchmark Metrics Definition

This section puts together the standard metrics required for LLM serving performance analysis. We classify metrics to two types: **single-request level metrics**, representing the metrics collected from one request. And **aggregated level metrics**, summarizing the single-request metrics from one run (with specific traffic scenario and num concurrency).

**NOTE**:

- Each single-request metric includes standard statistics: **percentile**, **min**, **max**, **stddev**, and **mean**.
- The following metrics cover **input**, **output**, and **end-to-end (e2e)** stages. For *chat* tasks, all stages are relevant for evaluation. For *embedding* tasks, where there is no output stage, output metrics will be set to 0. For details about output metrics collection, please check out `OUTPUT_METRICS_FIELDS` in [metrics.py](genai_bench/metrics/metrics.py).

### Single Request Level Metrics

The following metrics capture token-level performance for a single request, providing insights into server efficiency for each individual request.

| Glossary               | Meaning                                                                                                                                                   | Calculation Formula                                            | Units         |
|------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------|---------------|
| TTFT                   | Time to First Token. Initial response time when the first output token is generated. <br/> This is also known as the latency for the input (input) stage. | `TTFT = time_at_first_token - start_time`                      | seconds       |
| End-to-End Latency     | End-to-End latency. This metric indicates how long it takes from submitting a query to receiving the full response, including network latencies.          | `e2e_latency = end_time - start_time`                          | seconds       |
| TPOT                   | Time Per Output Token. The average time between two subsequent generated tokens.                                                                          | `TPOT = (e2e_latency - TTFT) / (num_output_tokens - 1)`        | seconds       |
| Output Latency         | Output latency. This metric indicates how long it takes to receive the full response after the first token is generated. | `output_latency = e2e_latency - TTFT`                           | seconds       |
| Output Inference Speed | The rate of how many tokens the model can generate per second for a single request.                                                                       | `inference_speed = 1 / TPOT`                                   | tokens/second |
| Num of Input Tokens    | Number of prompt tokens.                                                                                                                                  | `num_input_tokens = tokenizer.encode(prompt)`                  | tokens        |
| Num of Output Tokens   | Number of output tokens.                                                                                                                                  | `num_output_tokens = num_completion_tokens`                    | tokens        |
| Num of Request Tokens  | Total number of tokens processed in one request.                                                                                                          | `num_request_tokens = num_input_tokens + num_output_tokens`    | tokens        |
| Input Throughput       | The overall throughput of input (input process).                                                                                                          | `input_throughput = num_input_tokens / TTFT`                   | tokens/second |
| Output Throughput      | The throughput of output (output generation) for a single request.                                                                                        | `output_throughput = (num_output_tokens - 1) / output_latency` | tokens/second |

### Aggregated Metrics

This metrics collection summarizes the metrics relevant to a specific traffic load pattern, defined by the traffic scenario and the num of concurrency. It provides insights into server capacity and performance under pressure.

| Glossary                  | Meaning                                                                                                                      | Calculation Formula                                                                         | Units         |
|---------------------------|------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|---------------|
| Mean Input Throughput     | The average throughput of how many input tokens can be processed by the model in one run with multiple concurrent requests.  | `mean_input_throughput = sum(input_tokens_for_all_requests) / run_duration`                 | tokens/second |
| Mean Output Throughput    | The average throughput of how many output tokens can be processed by the model in one run with multiple concurrent requests. | `mean_output_throughput = sum(output_tokens_for_all_requests) / run_duration`               | tokens/second |
| Total Tokens Throughput   | The average throughput of how many tokens can be processed by the model, including both input and output tokens.             | `mean_total_tokens_throughput = all_requests["total_tokens"]["sum"] / run_duration`         | tokens/second |
| Total Chars Per Hour[^1]  | The average total characters can be processed by the model per hour.                                                         | `total_chars_per_hour = total_tokens_throughput * dataset_chars_to_token_ratio * 3600`      | Characters    |
| Requests Per Minute       | The number of requests processed by the model per minute.                                                                    | `num_completed_requests_per_min = num_completed_requests / (end_time - start_time) * 60`    | Requests      |
| Error Codes to Frequency  | A map that shows the returned error status code to its frequency.                                                            |                                                                                             |               |
| Error Rate                | The rate of error requests over total requests.                                                                              | `error_rate = num_error_requests / num_requests`                                            |               |
| Num of Error Requests     | The number of error requests in one load.                                                                                    | <pre><code>if requests.status_code != '200': <br/> num_error_requests += 1</code></pre>     |               |
| Num of Completed Requests | The number of completed requests in one load.                                                                                | <pre><code>if requests.status_code == '200': <br/> num_completed_requests += 1</code></pre> |               |
| Num of Requests           | The total number of requests processed for one load.                                                                         | `total_requests = num_completed_requests + num_error_requests`                              |               |

[^1]: *Total Chars Per Hour* is derived from a character-to-token ratio based on sonnet.txt and the model’s tokenizer. This metric aids in pricing decisions for an LLM serving solution. For tasks with multi-modal inputs, non-text tokens are converted to an equivalent character count using the same character-to-token ratio.
