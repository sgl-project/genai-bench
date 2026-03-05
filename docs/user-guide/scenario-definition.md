---
title: Traffic Scenarios
---

### What are scenarios?

Scenarios describe how genai-bench should shape requests during a benchmark run. They define token distributions (for text), per-document token budgets (for embeddings), re-rank budgets (for rerank), or modality properties (for images). Scenarios are parsed from concise strings and used by samplers to construct inputs and expected output lengths.

Scenarios are optional. If you don’t provide any and you supply a dataset, genai-bench runs in dataset mode, sampling raw entries from your dataset without token shaping.

### How scenarios are used

- The CLI accepts one or more scenarios via `--traffic-scenario`. Each run iterates over the supplied scenarios and the selected iteration parameter (concurrency or batch size).
- Internally, each scenario string is parsed into a Scenario class and passed to samplers to control request construction.
- Scenarios are defined as [multi-value options](https://click.palletsprojects.com/en/8.1.x/options/#multi-value-options) in click. Meaning you can pass this command multiple times to benchmark different loads.

### Scenario types and formats

- Text distributions
    - Deterministic: `D(num_input_tokens,num_output_tokens)`
        - Example: `D(100,1000)`
    - Normal: `N(mean_input_tokens,stddev_input_tokens)/(mean_output_tokens,stddev_output_tokens)`
        - Example: `N(480,240)/(300,150)`
    - Uniform: `U(min_input_tokens,max_input_tokens)/(min_output_tokens,max_output_tokens)` or `U(max_input_tokens,max_output_tokens)`
        - Examples: `U(50,100)/(200,250)` or `U(100,200)`

- Embeddings
    - Embedding: `E(tokens_per_document)`
    - Example: `E(1024)`

- Re-Rank
    - ReRank: `R(tokens_per_document,tokens_per_query)`
    - Example: `R(1024,100)`

- Image multi-modality
    - Image: `I(width,height)` or `I(width,height,num_images)`
    - Examples: `I(512,512)`, `I(2048,2048,2)`

- Special (no token shaping)
    - Dataset scenario: `dataset`
    - Use raw dataset entries without token shaping. See details below.

Notes

- Scenario strings are validated. If you mistype a scenario, you’ll get an error that shows the supported scenario types and the expected format for the type you chose.

### Dataset mode (scenarios are optional)

Scenarios are optional when a dataset is provided. If you don’t pass `--traffic-scenario`, genai-bench uses dataset mode automatically (equivalent to `--traffic-scenario dataset`).

Behavior by task in dataset mode:

- text-to-text
      - Picks a single line from your dataset as the prompt
      - Sets `max_tokens=None` and `ignore_eos=False` to avoid token shaping
- text-to-embeddings
      - Samples `batch_size` lines from your dataset per request
      - No token shaping is applied
- text-to-rerank
      - Samples a query and `batch_size` documents from your dataset
      - No token shaping is applied
- image-text-to-text and image-to-embeddings
      - Samples images from your dataset (defaults to 1 image per request in dataset mode)

Tip

- If you prefer to be explicit, you can pass `--traffic-scenario dataset` even when providing a dataset.

### Examples

Run using dataset-only mode (no token shaping):

```bash
genai-bench \
  --task text-to-text \
  --dataset-path data.csv \
  --api-backend openai \
  --api-base http://localhost:8000 \
  --api-model-name gpt-test \
  --model-tokenizer gpt2
```

Explicit dataset sentinel:

```bash
genai-bench \
  --task image-text-to-text \
  --dataset-path images.json \
  --traffic-scenario dataset \
  --api-backend openai \
  --api-base http://localhost:8000 \
  --api-model-name gpt-test \
  --model-tokenizer gpt2
```

Scenario-based runs:

```bash
genai-bench \
  --task text-to-text \
  --dataset-path prompts.txt \
  --traffic-scenario "N(480,240)/(300,150)" \
  --traffic-scenario "D(100,1000)" \
  --api-backend openai \
  --api-base http://localhost:8000 \
  --api-model-name gpt-test \
  --model-tokenizer gpt2
```

### Prefix Caching

Prefix caching allows you to benchmark how LLM servers handle shared prefix caching (also known as prompt caching or KV cache reuse). When enabled, all requests share a common prefix, with only the suffix varying between requests.

#### What is prefix caching?

Many LLM inference servers (like vLLM, SGLang, and others) can cache and reuse the KV (key-value) cache for common prefixes across multiple requests. This significantly improves performance when many requests share the same initial tokens.

#### How to enable

Choose one of two mutually exclusive options:

1. **`--prefix-len`**: Specify an absolute prefix length in tokens
   - Same prefix length for all requests
   - Example: `--prefix-len 1500` uses exactly 1500 tokens as prefix
   - Works with any scenario type (deterministic, normal, uniform)
   - Must be ≤ minimum possible input tokens for all scenarios

2. **`--prefix-ratio`**: Specify a ratio of prefix length to input length
   - Value in range [0.0, 1.0] (e.g., `0.75` for 75%, `1.0` for 100%)
   - Prefix length computed per-request as `int(num_input_tokens * ratio)`
   - Requires deterministic scenarios like `D(input_tokens,output_tokens)`

#### Request format

Each request follows this format:

```
<shared_prefix><separator><unique_suffix>
```

Where:
- `<shared_prefix>`: Generated once from the dataset and reused for all requests (length specified by `--prefix-len`)
- `<separator>`: A random 4-character hex hash that makes each request unique (e.g., `a7f3`, `2b9c`). Typically 1 token, but may be truncated if space is limited.
- `<unique_suffix>`: Generated from the dataset for each request

The separator and suffix lengths are automatically adjusted to ensure the total token count matches the scenario specification exactly.

#### Requirements and restrictions

Prefix caching has the following requirements:

1. **Task type**: Only supported for `text-to-text` tasks
2. **Traffic scenarios**:
   - **`--prefix-len`**: Works with any scenario type (D, N, U)
     - Must be ≤ minimum possible input tokens
     - For `D(1000,100)`: min = 1000
     - For `N(1000,200)/(500,100)`: min = max(1, mean - 3*stddev) = 400
     - For `U(500,1000)/(100,200)`: min = 500
     - If a sample results in `num_input_tokens < prefix_len`, it will be resampled
   - **`--prefix-ratio`**: Requires deterministic scenarios only
     - All scenarios must be deterministic starting with `D(`
   - **Not supported** with dataset mode (when using `--dataset-path` without `--traffic-scenario`)
3. **Prefix length range**: Must be in the range `[0, min_input_tokens]`
   - For multiple scenarios, `prefix_len` must be `<= min(input_tokens)` across all scenarios
   - The separator will be truncated automatically if needed to maintain exact token counts
4. **Multi-worker consistency**: The prefix is generated using a fixed random seed `(42, prefix_len)` so all workers produce the same prefix text. Different prefix lengths produce different prefix text to avoid cross-scenario cache overlap. Note: When using `--prefix-len` with multiple scenarios sharing the same prefix length, the server-side KV cache may already be warm for subsequent scenarios. Use `--warmup-ratio` to mitigate this. Note: This approach may be refined in future versions.

#### Examples

Basic usage with 75% prefix caching:

```bash
genai-bench benchmark \
  --task text-to-text \
  --traffic-scenario "D(2000,200)" \
  --prefix-len 1500 \
  --num-concurrency 10 \
  --api-backend sglang \
  --api-base http://localhost:8000 \
  --api-model-name meta-llama/Meta-Llama-3-8B-Instruct \
  --model-tokenizer meta-llama/Meta-Llama-3-8B-Instruct
```

This creates requests with:
- 2000 input tokens total (from `D(2000,200)`)
- 1500 tokens as **shared prefix** (same across all requests)
- ~496 tokens as **unique suffix** (varies per request, adjusted for 4-token separator)

Compare different prefix ratios:

```bash
# No prefix caching (baseline)
genai-bench benchmark --task text-to-text --traffic-scenario "D(1000,100)" --num-concurrency 8 ...

# 50% prefix
genai-bench benchmark --task text-to-text --traffic-scenario "D(1000,100)" --prefix-len 500 --num-concurrency 8 ...

# 90% prefix
genai-bench benchmark --task text-to-text --traffic-scenario "D(1000,100)" --prefix-len 900 --num-concurrency 8 ...
```

**Using --prefix-len with multiple scenarios** (same absolute prefix, different ratios):

```bash
# Test how cache efficiency changes across different input sizes with same prefix length
genai-bench benchmark \
  --task text-to-text \
  --traffic-scenario "D(1000,100)" \
  --traffic-scenario "D(2000,200)" \
  --traffic-scenario "D(4000,400)" \
  --prefix-len 800 \
  --num-concurrency 10 \
  --api-backend sglang \
  --api-base http://localhost:8000 \
  --api-model-name meta-llama/Meta-Llama-3-8B-Instruct \
  --model-tokenizer meta-llama/Meta-Llama-3-8B-Instruct
```

This tests scenarios with different cache ratios (80%, 40%, 20%) while keeping the same 800-token prefix.

**Using --prefix-ratio with multiple scenarios** (same ratio, different absolute lengths):

```bash
# Test how 75% prefix caching performs across different input sizes
genai-bench benchmark \
  --task text-to-text \
  --traffic-scenario "D(1000,100)" \
  --traffic-scenario "D(2000,200)" \
  --traffic-scenario "D(4000,400)" \
  --prefix-ratio 0.75 \
  --num-concurrency 10 \
  --api-backend sglang \
  --api-base http://localhost:8000 \
  --api-model-name meta-llama/Meta-Llama-3-8B-Instruct \
  --model-tokenizer meta-llama/Meta-Llama-3-8B-Instruct
```

This automatically uses 750, 1500, and 3000 token prefixes respectively, maintaining 75% ratio across all scenarios.

#### Use cases

Prefix caching is particularly useful for:

- **RAG systems**: Where the system prompt and context are shared across many queries
- **Multi-turn conversations**: Where conversation history is reused
- **Batch processing**: Where a template or instruction is prepended to many different inputs
- **Benchmarking cache efficiency**: Comparing server performance with and without prefix caching

#### Implementation details

- The prefix is generated **once** at the start of the benchmark from the dataset and reused for all requests
- Each request gets a unique random 4-character hex hash separator (e.g., `a7f3`, `2b9c`) to distinguish it from others
- The separator is typically 1 token but may be truncated when `prefix_len` is close to `input_tokens`
- The suffix varies for each request, generated from the dataset to simulate real-world usage patterns
- Total token count always matches the scenario specification exactly (e.g., `D(1000,100)` always sends exactly 1000 input tokens)
