# Migrating from SGLang/vLLM Benchmarking to GenAI Bench

This guide shows how to migrate from SGLang and vLLM benchmarking to GenAI Bench.

## SGLang Benchmarking

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
    --api-base http://localhost:8080 \
    --api-model-name sglang-model \
    --model-tokenizer meta-llama/Llama-3.1-8B-Instruct \
    --task text-to-text \
    --max-requests-per-run 500 \
    --max-time-per-run 30 \
    --traffic-scenario "D(512,512)" \
    --num-concurrency 1 \
    --server-engine "SGLang"
```

Notes: SGLang benchmark provides controls for request rate and max concurrency. GenAI-bench currently does not have request rate control, but does permit the user to select multiple concurrencies to benchmark, as shown below:

### Multiple Concurrency Levels Example

```bash
# GenAI Bench can test multiple concurrency levels in a single run
genai-bench benchmark \
    --api-backend sglang \
    --api-base http://localhost:8000 \
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
    --api-base http://localhost:8080 \
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

Notes: GenAI-bench does not explicitly allow control of input and output tokens for image scenarios. Using a [dataset config](../user-guide/run-benchmark.md#selecting-datasets) allows you to choose a column of text prompts, and adding a prompt lambda function can further filter inputs to the desired size. Output size cannot be directly controlled for image tasks yet.

### Traffic Distributions

SGLang's `--random-input-len` and `--random-output-len` parameters generate requests with random token lengths. GenAI Bench provides more control through [traffic distributions](../user-guide/scenario-definition.md), allowing you to specify the exact distribution type (deterministic, normal, uniform) for input and output tokens.

**Example: Normal Distribution**

The `N(mean,stddev)/(mean,stddev)` syntax specifies normal distributions for both input and output tokens:

```bash
genai-bench benchmark \
    --api-backend sglang \
    --api-base http://localhost:8000 \
    --api-model-name meta-llama/Llama-2-7b-hf \
    --model-tokenizer meta-llama/Llama-2-7b-hf \
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
