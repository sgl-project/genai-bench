# Run Benchmark

> **Note**: GenAI Bench now supports multiple cloud providers for both model endpoints and storage. For detailed multi-cloud configuration, see the [Multi-Cloud Authentication & Storage Guide](multi-cloud-auth-storage.md) or the [Quick Reference](multi-cloud-quick-reference.md).

## Start a chat benchmark

**IMPORTANT**: Use `genai-bench benchmark --help` to check out each command option and how to use it.

For starter, you can try to type `genai-bench benchmark`, it will prompt the list of options you need to specify.

Below is a sample command you can use to start a benchmark. The command will connect with a server running on address
`http://localhost:8082`, using the default traffic scenario and num concurrency, and run each combination 1 minute.

```shell
# Optional. Only needed for private/gated repositories or higher rate-limits
export HF_TOKEN="<your-token>"
# HF transformers will log a warning about torch not installed, since benchmark doesn't really need torch
# and cuda, we use this env to disable the warning
export TRANSFORMERS_VERBOSITY=error

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

## Start a vision based chat benchmark

**IMPORTANT**: Image auto-generation pipeline is not yet implemented in this repository, hence we will be using a huggingface dataset instead.

* **Image Datasets**: [Huggingface Llava Benchmark Images](https://huggingface.co/datasets/shenoyvvarun/llava-bench-in-the-wild)

Below is a sample command to trigger a vision benchmark task.

```shell
genai-bench benchmark \
            --api-backend openai \
            --api-key "your-openai-api-key" \
            --api-base "http://localhost:8180" \
            --api-model-name "/models/Phi-3-vision-128k-instruct" \
            --model-tokenizer "/models/Phi-3-vision-128k-instruct" \
            --task image-text-to-text \
            --max-time-per-run 15 \
            --max-requests-per-run 300 \
            --server-engine "SGLang" \
            --server-gpu-type A100-80G \
            --server-version "v0.4.10" \
            --server-gpu-count 4 \
            --traffic-scenario "I(256,256)" \
            --traffic-scenario "I(1024,1024)" \
            --num-concurrency 1 \
            --num-concurrency 8 \
            --dataset-config ./examples/dataset_configs/config_llava-bench-in-the-wild.json
```

For complex setups, we recommend use of [dataset configs](#using-dataset-configurations).

## Start an embedding benchmark

Below is a sample command to trigger an embedding benchmark task. Note: when running an embedding benchmark, it is recommended to set `--num-concurrency` to 1.

```shell
genai-bench benchmark --api-backend openai \
            --api-base "http://172.18.0.3:8000" \
            --api-key "xxx" \
            --api-model-name "/models/e5-mistral-7b-instruct" \
            --model-tokenizer "/mnt/data/models/e5-mistral-7b-instruct" \
            --task text-to-embeddings \
            --server-engine "SGLang" \
            --max-time-per-run 15 \
            --max-requests-per-run 1500 \
            --traffic-scenario "E(64)" \
            --traffic-scenario "E(128)" \
            --traffic-scenario "E(512)" \
            --traffic-scenario "E(1024)" \
            --server-gpu-type "H100" \
            --server-version "v0.4.2" \
            --server-gpu-count 1 \
            --batch-size 1 \
            --batch-size 8

## Start a rerank benchmark against OCI Cohere

Below is a sample command to trigger a benchmark against cohere chat API.

```shell
genai-bench benchmark --api-backend oci-cohere \
            --config-file /home/ubuntu/.oci/config \
            --api-base "https://ppe.inference.generativeai.us-chicago-1.oci.oraclecloud.com" \
            --api-model-name "rerank-v3.5" \
            --model-tokenizer "Cohere/rerank-v3.5" \
            --server-engine "cohere-TensorRT" \
            --task text-to-rerank \
            --num-concurrency 1 \
            --server-gpu-type A100-80G \
            --server-version "1.7.0" \
            --server-gpu-count 4 \
            --max-time-per-run 15 \
            --max-requests-per-run 3 \
            --additional-request-params '{"compartmentId": "COMPARTMENTID", "endpointId": "ENDPOINTID", "servingType": "DEDICATED"}' \
            --num-workers 4
```

## Start a benchmark against OCI Cohere

Below is a sample command to trigger a benchmark against cohere chat API.

```shell
genai-bench benchmark --api-backend oci-cohere \
            --config-file /home/ubuntu/.oci/config \
            --api-base "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com" \
            --api-model-name "c4ai-command-r-08-2024" \
            --model-tokenizer "/home/ubuntu/c4ai-command-r-08-2024" \
            --server-engine "SGLang" \
            --task text-to-text \
            --num-concurrency 1 \
            --server-gpu-type A100-80G \
            --server-version "command_r_082024_v1_7" \
            --server-gpu-count 4 \
            --max-time-per-run 15 \
            --max-requests-per-run 300 \
            --additional-request-params '{"compartmentId": "COMPARTMENTID", "endpointId": "ENDPOINTID", "servingType": "DEDICATED"}' \
            --num-workers 4
```

## Monitor a benchmark

**IMPORTANT**: logs in genai-bench are all useful. Please keep an eye on WARNING logs when you finish one benchmark.

### Specify --traffic-scenario and --num-concurrency

**IMPORTANT**: Please use `genai-bench benchmark --help` to check out the latest default value of `--num-concurrency`
and `--traffic-scenario`.

Both options are defined as [multi-value options](https://click.palletsprojects.com/en/8.1.x/options/#multi-value-options) in click. Meaning you can pass this command multiple times. If you want to define your own `--num-concurrency` or `--traffic-scenario`, you can use

```shell
genai-bench benchmark \
            --api-backend openai \
            --task text-to-text \
            --max-time-per-run 10 \
            --max-requests-per-run 300 \
            --num-concurrency 1 --num-concurrency 2 --num-concurrency 4 \
            --num-concurrency 8 --num-concurrency 16 --num-concurrency 32 \
            --traffic-scenario "N(480,240)/(300,150)" --traffic-scenario "D(100,100)"
```

### Notes on specific options

To manage each run or iteration in an experiment, genai-bench uses two parameters to control the exit logic. You can find more details in the `manage_run_time` function located in [utils.py](https://github.com/sgl-project/genai-bench/blob/main/genai_bench/cli/utils.py). Combination of `--max-time-per-run` and `--max-requests-per-run` should save overall time of one benchmark.

For light traffic scenarios, such as D(7800,200) or lighter, we recommend the following settings:

```shell
            --max-time-per-run 10 \
            --max-requests-per-run 300 \
```

For heavier traffic scenarios, like `D(16000,200)` or `D(128000,200)`, use the following configuration:

```shell
            --max-time-per-run 30 \
            --max-requests-per-run 100 \
            --traffic-scenario "D(16000,200)" \
            --traffic-scenario "D(32000,200)" \
            --traffic-scenario "D(128000,200)" \
            --num-concurrency 1 \
            --num-concurrency 2 \
            --num-concurrency 4 \
            --num-concurrency 8 \
            --num-concurrency 16 \
            --num-concurrency 32 \
```

## Distributed Benchmark

If you see the message below in the genai-bench logs, it indicates that a single process is insufficient to generate the desired load.

```log
CPU usage above 90%! This may constrain your throughput and may even give inconsistent response time measurements!
```

To address this, you can increase the number of worker processes using the `--num-workers` option. For example, to spin up 4 worker processes, use:

```shell
    --num-workers 4
    --master-port 5577
```

This distributes the load across multiple processes on a single machine, improving performance and ensuring your benchmark runs smoothly.

### Controlling User Spawn Rate

When running high-concurrency benchmarks with large payloads (e.g., 20k+ tokens), workers may become overwhelmed if all users are spawned immediately. This can cause worker heartbeat failures and restarts.

To prevent this, use the `--spawn-rate` option to control how quickly users are spawned:

```shell
    --num-concurrency 500 \
    --num-workers 16 \
    --spawn-rate 50
```

**Examples:**

- `--spawn-rate 50`: Spawn 50 users per second (takes 10 seconds to reach 500 users)
- `--spawn-rate 100`: Spawn 100 users per second (takes 5 seconds to reach 500 users)
- `--spawn-rate 500`: Spawn all users immediately (default behavior)

### Notes on Usage

1. This feature is experimental, so monitor the system's behavior when enabling multiple workers.
2. Recommended Limit: Do **not** set the number of workers to more than 16, as excessive worker processes can lead to resource contention and diminished performance.
3. Ensure your system has sufficient CPU and memory resources to support the desired number of workers.
4. Adjust the number of workers based on your target load and system capacity to achieve optimal results.
5. For high-concurrency tests with large payloads, use `--spawn-rate` to prevent worker overload.

## Execution Engine Options

GenAI Bench supports two execution engine options: **Locust** (default) and **async runner**. The async runner is a native async execution engine that provides better streaming performance and precise timing measurements than Locust.

### When to Use the Async Runner

Use the async runner (specify `--execution-engine=async`) when:
- ✅ You need precise timing measurements for streaming responses
- ✅ You're running single-machine benchmarks (async runner is single-process)
- ✅ You're using OpenAI-compatible backends (OpenAI, Baseten)
- ✅ You need better streaming metrics than Locust (even in closed-loop mode)

Use Locust when:
- ✅ You need distributed/multi-machine testing
- ✅ You're using non-OpenAI backends (AWS, Azure, GCP, OCI, etc.)
- ✅ You need worker process isolation

### Async Runner Usage

The async runner supports two execution modes:

1. **Open-loop mode** (QPS-based): Specify `--qps-level` to control the fixed arrival rate of requests
2. **Closed-loop mode** (concurrency-based): Specify `--num-concurrency` (without `--qps-level`) to maintain a fixed number of concurrent requests

**Understanding the difference:**
- **Open-loop (QPS-based)**: You specify `--qps-level` (e.g., 5.0), and concurrency emerges naturally based on request latency. QPS is the input.
- **Closed-loop (concurrency-based)**: You specify `--num-concurrency` (e.g., 10), and QPS emerges naturally based on request latency. Concurrency is the input.

**Note**: The async runner's closed-loop mode provides better streaming metrics than Locust, making it ideal when you want concurrency-based testing with improved streaming performance.

**Choosing QPS for Open-Loop Mode:**

If you're using open-loop mode (`--qps-level`), you need to calculate QPS based on your target concurrency and expected latency using Little's Law:

```
QPS = Target Concurrency / Expected Average Latency
```

For example:
- Target: ~10 concurrent requests
- Expected latency: ~2 seconds
- QPS: `10 / 2 = 5.0`

```shell
genai-bench benchmark \
    --api-backend openai \
    --api-key "your-api-key" \
    --api-base "http://localhost:8082" \
    --api-model-name "meta-llama/Meta-Llama-3-70B-Instruct" \
    --model-tokenizer "/path/to/tokenizer" \
    --task text-to-text \
    --execution-engine async \
    --qps-level 5.0 \
    --distribution exponential \
    --max-time-per-run 60 \
    --max-requests-per-run 600
```

**Example: Closed-Loop Mode (Concurrency-Based)**

The async runner also supports closed-loop mode when you specify `--num-concurrency` without `--qps-level`. This provides better streaming metrics than Locust while maintaining a fixed concurrency level:

```shell
genai-bench benchmark \
    --api-backend openai \
    --api-key "your-api-key" \
    --api-base "http://localhost:8082" \
    --api-model-name "meta-llama/Meta-Llama-3-70B-Instruct" \
    --model-tokenizer "/path/to/tokenizer" \
    --task text-to-text \
    --execution-engine async \
    --num-concurrency 10 \
    --max-time-per-run 60 \
    --max-requests-per-run 600
```

In this mode, the async runner maintains exactly 10 concurrent requests (similar to Locust), but with improved streaming metrics collection.

### Async Runner Parameters

- **`--execution-engine`**: Choose `"locust"` (default) or `"async"`
- **`--qps-level`**: Specifies queries per second (e.g., `5.0` for 5 QPS). Required for open-loop mode. Use Little's Law to choose: `QPS = Target Concurrency / Expected Average Latency`. Must be between 0 and 10000.
- **`--num-concurrency`**: 
  - **In closed-loop mode**: Required. Specifies the target number of concurrent requests (e.g., `10` for 10 concurrent requests).
  - **In open-loop mode**: Ignored (QPS determines concurrency).
- **`--distribution`**: Inter-arrival distribution for requests (open-loop mode only):
  - `"exponential"` (default): Realistic arrival pattern
  - `"uniform"`: Uniform distribution between 0 and 2×mean
  - `"constant"`: Fixed inter-arrival time

**Execution Modes Comparison:**

| Aspect | Locust | Async Runner (Open-Loop) | Async Runner (Closed-Loop) |
|--------|--------|-------------------------|---------------------------|
| **Input** | Concurrency (e.g., `--num-concurrency 10`) | QPS (e.g., `--qps-level 5.0`) | Concurrency (e.g., `--num-concurrency 10`) |
| **Output** | QPS emerges from latency | Concurrency emerges from latency | QPS emerges from latency |
| **Model** | N concurrent users continuously sending requests | Fixed rate of request arrivals | N concurrent requests continuously maintained |
| **Relationship** | `QPS ≈ Concurrency / Avg Latency` | `Concurrency ≈ QPS × Avg Latency` | `QPS ≈ Concurrency / Avg Latency` |
| **Streaming Metrics** | Good | Excellent | Excellent (better than Locust) |

**Choosing Between Open-Loop and Closed-Loop Mode:**

- **Use open-loop mode** (`--qps-level`) when:
  - You want to test a fixed arrival rate (e.g., simulating production traffic patterns)
  - You're doing capacity planning based on expected request rates
  - You want to measure how concurrency scales with fixed QPS

- **Use closed-loop mode** (`--num-concurrency` without `--qps-level`) when:
  - You want to test a fixed concurrency level (similar to Locust)
  - You need better streaming metrics than Locust provides
  - You're comparing against Locust results with the same concurrency

**How to Determine QPS for Open-Loop Mode:**

Since you don't know latency until you run requests, use one of these approaches:

1. **Run a quick closed-loop test first**: Use the async runner in closed-loop mode with `--num-concurrency` to measure latency, then calculate QPS manually: `QPS = Concurrency / Measured Latency`.

2. **Use an educated guess**: If you have experience with similar models/deployments, estimate latency and calculate QPS.

**Example Workflow for New Deployment:**
```shell
# Step 1: Measure latency using closed-loop mode
genai-bench benchmark \
    --execution-engine async \
    --num-concurrency 10 \
    # ... other options ...
# Note the average latency from the results

# Step 2: Calculate QPS and use in open-loop mode
# If average latency was ~2 seconds: QPS = 10 / 2 = 5.0
genai-bench benchmark \
    --execution-engine async \
    --qps-level 5.0 \
    # ... other options ...
```

### Example: Streaming Performance Benchmark

```shell
genai-bench benchmark \
    --api-backend openai \
    --api-key "your-api-key" \
    --api-base "http://localhost:8082" \
    --api-model-name "gpt-4" \
    --model-tokenizer "/path/to/tokenizer" \
    --task text-to-text \
    --execution-engine async \
    --qps-level 5.0 \
    --distribution exponential \
    --max-time-per-run 120 \
    --traffic-scenario "N(1000,200)/(500,100)"
```

**Note**: In this example, `--qps-level 5.0` would result in ~5-10 concurrent requests if average latency is 1-2 seconds (per Little's Law: Concurrency = QPS × Latency).

### Async Runner Performance Optimizations

The async runner includes several performance optimizations:

- **Session Reuse**: HTTP connections are reused across requests for better performance
- **Memory Efficiency**: Requests are generated on-demand rather than pre-allocated, reducing memory footprint for long-running benchmarks
- **Precise Timing**: Better streaming metrics collection with accurate time-to-first-token measurements
- **Resource Cleanup**: Automatic cleanup of HTTP sessions and connections after benchmark completion

### Backend Compatibility

The async runner currently supports:
- ✅ OpenAI
- ✅ Baseten (if OpenAI-compatible)

For other backends (AWS Bedrock, Azure OpenAI, GCP Vertex, OCI), use Locust with `--execution-engine locust` (default).

## Using Dataset Configurations

Genai-bench supports flexible dataset configurations through two approaches:

### Simple CLI Usage (for basic datasets)

```shell
# Local CSV file
--dataset-path /path/to/data.csv \
--dataset-prompt-column "prompt"

# HuggingFace dataset with simple options
--dataset-path squad \
--dataset-prompt-column "question"

# Local text file (default)
--dataset-path /path/to/prompts.txt
```

### Advanced Configuration Files (for complex setups)

For advanced HuggingFace configurations, create a JSON config file:

**Important Note for HuggingFace Datasets:**
When using HuggingFace datasets, you should always check if you need a `split`, `subset`, or `name` parameter to avoid errors. If you don't specify, HuggingFace's `load_dataset` may return a `DatasetDict` object instead of a `Dataset`, which will cause the benchmark to fail. Some datasets require additional configuration parameters like `name` to specify which subset of the dataset to load.

To specify a dataset config, use: `--dataset-config config.json`.

**config.json:**

```json
{
  "source": {
    "type": "huggingface",
    "path": "ccdv/govreport-summarization",
    "huggingface_kwargs": {
      "split": "train",
      "revision": "main"
    }
  },
  "prompt_column": "report"
}
```

**Vision dataset config:**

```json
{
  "source": {
    "type": "huggingface",
    "path": "BLINK-Benchmark/BLINK",
    "huggingface_kwargs": {
      "split": "test",
      "name": "Jigsaw"
    }
  },
  "prompt_column": "question",
  "image_column": "image_1"
}
```

**Example for the llava-bench-in-the-wild dataset:**

```json
{
  "source": {
    "type": "huggingface",
    "path": "lmms-lab/llava-bench-in-the-wild",
    "huggingface_kwargs": {
      "split": "train"
    }
  },
  "prompt_column": "question",
  "image_column": "image"
}
```

**Benchmarking with large images:**
When benchmarking with very large images, the pillow library throws an exception. To get around this, use a config with the argument `unsafe_allow_large_images`, which disables the warning.

```json
{
  "source": {
    "type": "huggingface",
    "path": "zhang0jhon/Aesthetic-4K",
    "huggingface_kwargs": {
      "split": "train"
    }
  },
  "prompt_column": "text",
  "image_column": "image",
  "unsafe_allow_large_images": true
}
```

**Using prompt lambdas (vision tasks only):**
If you want to benchmark a specific portion of a vision dataset, you can use the `prompt_lambda` argument to select only the desired section. When using `prompt_lambda`, you don't need to specify `prompt_column` as the lambda function generates the prompts dynamically. Note that `prompt_lambda` is only available for vision/multimodal tasks.

```json
{
  "source": {
    "type": "huggingface",
    "path": "lmms-lab/LLaVA-OneVision-Data",
    "huggingface_kwargs": {
      "split": "train",
      "name": "CLEVR-Math(MathV360K)"
    }
  },
  "image_column": "image",
  "prompt_lambda": "lambda x: x['conversations'][0]['value'] if len(x['conversations']) > 1 else ''"
}
```

**Benefits of config files:**

- Access to ALL HuggingFace `load_dataset` parameters
- Reusable and version-controllable
- Support for complex configurations
- Future-proof (no CLI updates needed for new HuggingFace features)

## Picking units

Genai-bench defaults to measuring latency (End-to-end latency, TTFT, TPOT, Input/Output latencies) in seconds. If you prefer milliseconds, you can select them with `--metrics-time-unit [s|ms]`. 