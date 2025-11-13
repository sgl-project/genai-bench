# Migrating from SGLang/vLLM Benchmarking to GenAI Bench

This guide shows how to migrate from SGLang and vLLM benchmarking to GenAI Bench.

## SGLang Benchmarking

GenAI Bench uses [traffic scenarios](../user-guide/scenario-definition.md) to control input and output token lengths instead of SGLang's `--random-input-len` and `--random-output-len` parameters. For benchmark intensity, SGLang uses `--request-rate` parameters to control how requests are sent over time, while GenAI Bench uses concurrency levels (`--num-concurrency`) to manage benchmark intensity instead.

### Basic Text Benchmark

**SGLang Command:**
```bash
python -m sglang.bench_serving \
    --backend sglang-oai-chat \
    --num-prompts 500 \
    --random-input-len 512 \
    --random-output-len 512 \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --base-url http://127.0.0.1:8080
```

**GenAI Bench Equivalent:**
```bash
genai-bench benchmark \
    --api-backend sglang \
    --api-base http://127.0.0.1:8080 \
    --api-model-name sglang-model \
    --model-tokenizer meta-llama/Llama-3.1-8B-Instruct \
    --task text-to-text \
    --max-requests-per-run 500 \
    --max-time-per-run 30 \
    --traffic-scenario "D(512,512)" \
    --num-concurrency 1 \
    --server-engine "SGLang"
```

### Multiple Concurrency Levels

GenAI Bench can test multiple concurrency levels in a single run:

```bash
genai-bench benchmark \
    --api-backend sglang \
    --api-base http://127.0.0.1:8080 \
    --api-model-name sglang-model \
    --model-tokenizer meta-llama/Llama-3.1-8B-Instruct \
    --task text-to-text \
    --max-requests-per-run 500 \
    --max-time-per-run 30 \
    --traffic-scenario "D(512,512)" \
    --num-concurrency 1 \
    --num-concurrency 4 \
    --num-concurrency 8 \
    --num-concurrency 16 \
    --server-engine "SGLang"
```

**Note:** SGLang provides controls for request rate and max concurrency. GenAI Bench does not yet have request rate control, but allows explicitly setting the number of concurrent users.

### Image Benchmarking

**SGLang Command:**
```bash
python -m sglang.bench_serving \
    --backend sglang-oai-chat \
    --dataset-name image \
    --num-prompts 500 \
    --image-count 3 \
    --image-resolution 720p \
    --random-input-len 512 \
    --random-output-len 512
```

**GenAI Bench Equivalent:**
```bash
genai-bench benchmark \
    --api-backend sglang \
    --api-base http://127.0.0.1:8080 \
    --api-model-name sglang-model \
    --model-tokenizer Qwen/Qwen2.5-VL-3B-Instruct \
    --task image-text-to-text \
    --max-requests-per-run 500 \
    --max-time-per-run 30 \
    --traffic-scenario "I(1280,720,3)" \
    --num-concurrency 1 \
    --server-engine "SGLang" \
    --dataset-config ./examples/dataset_configs/config_llava-bench-in-the-wild.json
```

**Note:** GenAI Bench does not explicitly allow control of input and output tokens for image scenarios. Use a [dataset config](../user-guide/run-benchmark.md#selecting-datasets) to choose a column of text prompts, and add a prompt lambda function to filter inputs to the desired size. Output size cannot be directly controlled for image tasks (defaults to unlimited).

### Traffic Distributions

SGLang's `--random-input-len` and `--random-output-len` parameters generate requests with random token lengths. GenAI Bench provides more control through [traffic distributions](../user-guide/scenario-definition.md), allowing you to specify the exact distribution type (deterministic, normal, uniform) for input and output tokens.

**Example: Normal Distribution**

The `N(mean,stddev)/(mean,stddev)` syntax specifies normal distributions for both input and output tokens:

```bash
genai-bench benchmark \
    --api-backend sglang \
    --api-base http://127.0.0.1:8080 \
    --api-model-name sglang-model \
    --model-tokenizer meta-llama/Llama-3.1-8B-Instruct \
    --task text-to-text \
    --max-requests-per-run 500 \
    --max-time-per-run 30 \
    --traffic-scenario "N(480,240)/(300,150)" \
    --num-concurrency 8 \
    --server-engine "SGLang"
```

This example uses:
- **Input tokens**: Normal distribution with mean=480, standard deviation=240
- **Output tokens**: Normal distribution with mean=300, standard deviation=150

For other distribution types (deterministic, uniform, embeddings) and more examples, see the [Traffic Scenarios](../user-guide/scenario-definition.md) documentation.

## vLLM Benchmarking

To set benchmark intensity, vLLM uses `--request-rate` parameters to control how requests are sent over time. GenAI Bench does not currently support request rate benchmarks and instead uses concurrency levels (`--num-concurrency`) to manage benchmark intensity. Additionally, vLLM's `--burstiness` parameter controls traffic variability using a Gamma distribution to create variable bursts of traffic. GenAI Bench handles similar variability through [traffic distributions](../user-guide/scenario-definition.md), where distributions with higher variance (e.g., normal distributions with larger standard deviations) can simulate bursty traffic patterns, while deterministic distributions or low-variance distributions create more uniform patterns. Traffic distributions can also be used to set input and output token length.

### Basic Text Benchmark

**vLLM Command:**
```bash
vllm bench serve \
    --backend vllm \
    --model NousResearch/Hermes-3-Llama-3.1-8B \
    --endpoint /v1/completions \
    --dataset-name sharegpt \
    --dataset-path <your data path>/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 10
```

**GenAI Bench Equivalent:**
```bash
genai-bench benchmark \
    --api-backend vllm \
    --api-base http://127.0.0.1:8000 \
    --api-model-name vllm-model \
    --model-tokenizer NousResearch/Hermes-3-Llama-3.1-8B \
    --task text-to-text \
    --max-requests-per-run 10 \
    --max-time-per-run 30 \
    --traffic-scenario dataset \
    --dataset-config <configs path>/sharegpt_config.json \
    --num-concurrency 1000 \
    --server-engine "vLLM"
```

**Note:** vLLM's `--request-rate` defaults to infinite, sending all requests immediately. The `--max-concurrency` parameter defaults to unlimited if not specified. Since GenAI Bench does not support unlimited concurrency, use a high value (e.g., 1000) to approximate this behavior. For benchmarks where a column of a dataset should be used as input, use `--traffic-scenario dataset` and and configure the dataset using a [dataset config](../user-guide/run-benchmark.md#selecting-datasets).

### Ramp-up Request Rate

GenAI Bench can test multiple concurrency levels in a single run:

```bash
genai-bench benchmark \
    --api-backend vllm \
    --api-base http://127.0.0.1:8000 \
    --api-model-name vllm-model \
    --model-tokenizer NousResearch/Hermes-3-Llama-3.1-8B \
    --task text-to-text \
    --max-requests-per-run 10 \
    --max-time-per-run 30 \
    --traffic-scenario dataset \
    --dataset-config <configs path>/sharegpt_config.json \
    --num-concurrency 1 \
    --num-concurrency 4 \
    --num-concurrency 8 \
    --num-concurrency 16 \
    --server-engine "vLLM"
```

**Note:** GenAI Bench does not have explicit request rate control, but allows testing multiple concurrency levels in a single run, which provides similar stress testing capabilities. vLLM supports ramp-up request rate (linear or exponential) using `--ramp-up-strategy`, `--ramp-up-start-rps`, and `--ramp-up-end-rps`. GenAI Bench's equivalent is to run benchmarks at different concurrency levels to test how the server performs at different throughputs.

### Image Benchmarking

**vLLM Command:**
```bash
vllm bench serve \
    --backend openai-chat \
    --model Qwen/Qwen2-VL-7B-Instruct \
    --endpoint /v1/chat/completions \
    --dataset-name hf \
    --dataset-path lmarena-ai/VisionArena-Chat \
    --hf-split train \
    --num-prompts 1000
```

**GenAI Bench Equivalent:**
```bash
genai-bench benchmark \
    --api-backend vllm \
    --api-base http://127.0.0.1:8000 \
    --api-model-name vllm-model \
    --model-tokenizer Qwen/Qwen2-VL-7B-Instruct \
    --task image-text-to-text \
    --max-requests-per-run 1000 \
    --max-time-per-run 30 \
    --traffic-scenario dataset \
    --dataset-config ./examples/dataset_configs/config_vision_arena.json \
    --num-concurrency 1000 \
    --server-engine "vLLM"
```

### Traffic Burstiness

vLLM's `--burstiness` parameter controls traffic variability using a Gamma distribution (range: > 0). Lower values create bursty traffic (more variable request patterns), while higher values create uniform traffic (more consistent request patterns).

GenAI Bench provides similar control through [traffic distributions](../user-guide/scenario-definition.md), which allow you to specify different distribution types for token lengths. While GenAI Bench's distributions control token length variability rather than request timing variability, you can achieve similar effects:

- **Bursty traffic patterns**: Use distributions with higher variance (e.g., normal distributions with larger standard deviations, or uniform distributions over wider ranges)
- **Uniform traffic patterns**: Use deterministic distributions `D(input,output)` or distributions with low variance

For example, to simulate bursty traffic similar to vLLM's low burstiness values, you might use a normal distribution with a high standard deviation:

```bash
genai-bench benchmark \
    --api-backend vllm \
    --api-base http://127.0.0.1:8000 \
    --api-model-name vllm-model \
    --model-tokenizer NousResearch/Hermes-3-Llama-3.1-8B \
    --task text-to-text \
    --max-requests-per-run 500 \
    --max-time-per-run 30 \
    --traffic-scenario "N(512,256)/(512,256)" \
    --num-concurrency 8 \
    --server-engine "vLLM"
```

For more uniform traffic patterns (similar to high burstiness values), use deterministic distributions:

```bash
genai-bench benchmark \
    --api-backend vllm \
    --api-base http://127.0.0.1:8000 \
    --api-model-name vllm-model \
    --model-tokenizer NousResearch/Hermes-3-Llama-3.1-8B \
    --task text-to-text \
    --max-requests-per-run 500 \
    --max-time-per-run 30 \
    --traffic-scenario "D(512,512)" \
    --num-concurrency 8 \
    --server-engine "vLLM"
```

For more details on available distribution types, see the [Traffic Scenarios](../user-guide/scenario-definition.md) documentation.



## See Also

For more detailed information on using GenAI Bench, including advanced configuration options, dataset setup, and result analysis, see the [User Guide](../user-guide/index.md). For information on command-line arguments and options, see the [CLI Guidelines](command-guidelines.md).
