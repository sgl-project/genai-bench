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

Use the `--prefix-len` option to specify the length of the shared prefix in tokens. Must be used with a deterministic scenario like `D(input_tokens,output_tokens)`.

#### Request format

Each request follows this format:

```
<shared_prefix>
#N
<unique_suffix>
```

Where:
- `<shared_prefix>`: Generated once from the dataset and reused for all requests (length specified by `--prefix-len`)
- `#N`: A numbered separator that makes each request unique (e.g., `\n#1\n`, `\n#2\n`, `\n#3\n`). Typically 4 tokens, but may be truncated if space is limited.
- `<unique_suffix>`: Generated from the dataset for each request

The separator and suffix lengths are automatically adjusted to ensure the total token count matches the scenario specification exactly.

#### Requirements and restrictions

Prefix caching has the following requirements:

1. **Task type**: Only supported for `text-to-text` tasks
2. **Traffic scenarios**: Requires a deterministic traffic scenario (e.g., `D(100,50)`)
   - Must use at least one deterministic scenario starting with `D(`
   - **Not supported** with dataset mode (when using `--dataset-path` without `--traffic-scenario`)
3. **Prefix length range**: Must be in the range `[0, input_tokens]`
   - For `D(1000,100)`, `--prefix-len` can be 0 to 1000
   - The separator will be truncated automatically if needed to maintain exact token counts

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

#### Use cases

Prefix caching is particularly useful for:

- **RAG systems**: Where the system prompt and context are shared across many queries
- **Multi-turn conversations**: Where conversation history is reused
- **Batch processing**: Where a template or instruction is prepended to many different inputs
- **Benchmarking cache efficiency**: Comparing server performance with and without prefix caching

#### Implementation details

- The prefix is generated **once** at the start of the benchmark from the dataset and reused for all requests
- Each request gets a unique numbered separator (`\n#1\n`, `\n#2\n`, etc.) to distinguish it from others
- The separator is typically 4 tokens but may be truncated when `prefix_len` is close to `input_tokens`
- The suffix varies for each request, generated from the dataset to simulate real-world usage patterns
- Total token count always matches the scenario specification exactly (e.g., `D(1000,100)` always sends exactly 1000 input tokens)
