# Run Benchmark

> **Note**: GenAI Bench now supports multiple cloud providers for both model endpoints and storage. For detailed multi-cloud configuration, see the [Multi-Cloud Authentication & Storage Guide](multi-cloud-auth-storage.md) or the [Multi-Cloud Quick Reference](multi-cloud-quick-reference.md).

**IMPORTANT**: Use `genai-bench benchmark --help` to check out each command option and how to use it.

For starter, you can try to type `genai-bench benchmark`, it will prompt the list of options you need to specify.

## Examples

### Start a chat benchmark

Below is a sample command you can use to start a benchmark. The command will connect with a server running on address
`http://localhost:8082`, using the default traffic scenario and num concurrency, and run each combination 1 minute.

```shell
# Optional. Only needed for private/gated repositories or higher rate-limits
export HF_TOKEN="<your-token>"
# HF transformers will log a warning about torch not installed, since benchmark doesn't really need torch
# and cuda, we use this env to disable the warning
export TRANSFORMERS_VERBOSITY=error

genai-bench benchmark --api-backend sglang \
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

### Start a vision based chat benchmark

**IMPORTANT**: Image auto-generation pipeline is not yet implemented in this repository, hence we will be using a huggingface dataset instead.

* **Image Datasets**: [Huggingface Llava Benchmark Images](https://huggingface.co/datasets/shenoyvvarun/llava-bench-in-the-wild)

Below is a sample command to trigger a vision benchmark task.

```shell
genai-bench benchmark \
            --api-backend sglang \
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

For complex setups, we recommend use of [dataset configs](#selecting-datasets).

### Start an embedding benchmark

Below is a sample command to trigger an embedding benchmark task. Note: when running an embedding benchmark, it is recommended to set `--num-concurrency` to 1.

```shell
genai-bench benchmark --api-backend sglang \
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
```

### Start a rerank benchmark against OCI Cohere

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

### Start a benchmark against OCI Cohere

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

## Specify a custom benchmark load

**IMPORTANT**: logs in genai-bench are all useful. Please keep an eye on WARNING logs when you finish one benchmark.

You can specify a custom load to benchmark through setting traffic scenarios and concurrencies to benchmark at. 

Traffic scenarios let you define the shape of requests when benchmarking. See [Traffic Scenarios](./scenario-definition.md) for more information.

You can control benchmark intensity using either concurrency levels or request rates:

- **Concurrency-based**: The number of concurrent users making requests. Running various concurrencies allows you to benchmark performance at different loads. Each specified scenario is run at each concurrency. Specify concurrencies to run with `--num-concurrency`.
- **Rate-based**: Target request rates (requests/second) using token bucket rate limiting for precise rate control. Specify request rates with `--request-rate`.

**IMPORTANT**: `--num-concurrency` and `--request-rate` are mutually exclusive (except for embedding/rerank tasks which use `--batch-size`). By default, GenAI Bench uses `--num-concurrency`. Please use `genai-bench benchmark --help` to check out the latest default value of `--num-concurrency` and `--traffic-scenario`.

Both `--num-concurrency` and `--request-rate` are defined as [multi-value options](https://click.palletsprojects.com/en/8.1.x/options/#multi-value-options) in click. Meaning you can pass these commands multiple times. 

For example, the below benchmark command runs a scenario with a normal distribution of input and output tokens (Input mean=480, st.dev=240), (Output mean=300, st.dev=150) at concurrencies 1, 2, 4, 8, 16 and 32. 

```shell
genai-bench benchmark \
            --api-backend sglang \
            --task text-to-text \
            --max-time-per-run 10 \
            --max-requests-per-run 300 \
            --num-concurrency 1 --num-concurrency 2 --num-concurrency 4 \
            --num-concurrency 8 --num-concurrency 16 --num-concurrency 32 \
            --traffic-scenario "N(480,240)/(300,150)" --traffic-scenario "D(100,100)"
```

### Using Request Rate

Alternatively, you can use `--request-rate` to specify target request rates (requests/second) instead of concurrency levels. For request rate runs, maximum concurrency defaults to 5000 to ensure sufficient workers are available, but can be overridden with `--max-concurrency`. 
```shell
genai-bench benchmark \
            --api-backend openai \
            --task text-to-text \
            --max-time-per-run 10 \
            --max-requests-per-run 300 \
            --request-rate 1 --request-rate 5 --request-rate 10 --request-rate 20 \
            --traffic-scenario "N(480,240)/(300,150)" --traffic-scenario "D(100,100)"
```

When using `--request-rate`, the benchmark automatically uses request rate iteration instead of concurrency-based iteration. The rate limiter ensures requests are sent at the specified target rate with precise timing control.

**Concurrency and Spawn Rate**: For request rate runs, maximum concurrency defaults to 5000 to ensure sufficient workers are available, but can be overridden with `--max-concurrency`. The spawn rate defaults to the maximum concurrency value unless otherwise specified with `--spawn-rate`. You can override the spawn rate using `--spawn-rate` if needed, but this is generally not recommended as it may affect rate limiting accuracy.

**Note**: In distributed mode (when using `--num-workers`), the target request rate is automatically divided among all workers. For example, with `--request-rate 20` and `--num-workers 4`, each worker will target 5 requests/second. If the per-worker rate is very low (< 0.1 req/s), a warning will be displayed suggesting fewer workers or a higher target rate for better accuracy.

### Notes on benchmark duration

To manage each run or iteration in an experiment, genai-bench uses two parameters to control the exit logic. Benchmark runs terminate after exceeding either the maximum time limit the maximum number of requests. These are specified with `--max-time-per-run` and `--max-requests-per-run`. You can find more details in the `manage_run_time` function located in [utils.py](https://github.com/sgl-project/genai-bench/blob/main/genai_bench/cli/utils.py). 

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

## Request Rate vs Concurrency

GenAI Bench supports two approaches to control benchmark intensity:

- **`--num-concurrency`**: Sets the number of concurrent users/requests. This is useful for testing how the system performs under different levels of concurrent load. The actual request rate will depend on how quickly requests complete.

- **`--request-rate`**: Sets a target request rate (requests/second). This is useful when you need to test specific throughput targets or want precise control over request timing. For request rate runs, maximum concurrency defaults to 5000 to ensure sufficient workers are available, but can be overridden with `--max-concurrency`. The spawn rate defaults to the maximum concurrency value unless otherwise specified with `--spawn-rate`.

Choose the approach that best matches your testing needs:
- Use `--num-concurrency` when you want to test system behavior under specific concurrent load levels
- Use `--request-rate` when you need to test specific throughput targets or want precise rate control

**Important**: These options are mutually exclusive (except for embedding/rerank tasks which use `--batch-size`).

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

### Notes on Usage

1. This feature is experimental, so monitor the system's behavior when enabling multiple workers.
2. Recommended Limit: Do **not** set the number of workers to more than 16, as excessive worker processes can lead to resource contention and diminished performance.
3. Ensure your system has sufficient CPU and memory resources to support the desired number of workers.
4. Adjust the number of workers based on your target load and system capacity to achieve optimal results.
5. For high-concurrency tests with large payloads, use `--spawn-rate` to prevent worker overload.

### Controlling User Spawn Rate

By default, users are spawned at a rate equal to the concurrency, meaning it takes one second for all users to be created. When running high-concurrency benchmarks with large payloads (e.g., 20k+ tokens), workers may become overwhelmed if all users are spawned immediately. This can cause worker heartbeat failures and restarts.

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

## Selecting datasets

By default, genai-bench samples tokens to benchmark from [sonnet.txt](https://github.com/sgl-project/genai-bench/blob/main/genai_bench/data/sonnet.txt) for `text-to-text` or `text-to-embeddings` tasks. Image tasks do not have a default dataset. To select a dataset to benchmark from, genai-bench supports flexible dataset configurations through two approaches:

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

## Picking Metrics Time Units

Genai-bench defaults to measuring latency metrics (End-to-end latency, TTFT, TPOT, Input/Output latencies) in seconds. If you prefer milliseconds, you can select them with `--metrics-time-unit [s|ms]`. 