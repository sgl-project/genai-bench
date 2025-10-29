---
title: Traffic Scenarios
---

### What are scenarios?

Scenarios describe how genai-bench should shape requests during a benchmark run. They define token distributions (for text), per-document token budgets (for embeddings), re-rank budgets (for rerank), or modality properties (for images). Scenarios are parsed from concise strings and used by samplers to construct inputs and expected output lengths.

Scenarios are optional. If you don’t provide any and you supply a dataset, genai-bench runs in dataset mode, sampling raw entries from your dataset without token shaping.

### How scenarios are used

- The CLI accepts one or more scenarios via `--traffic-scenario`. Each run iterates over the supplied scenarios and the selected iteration parameter (concurrency or batch size).
- Internally, each scenario string is parsed into a Scenario class and passed to samplers to control request construction.

### Scenario types and formats

- Text distributions
    - Deterministic: `D(num_input_tokens,num_output_tokens)`
        - Example: `D(100,1000)`
    - Normal: `N(mean_input_tokens,stddev_input_tokens)/(mean_output_tokens,stddev_output_tokens)`
        - Example: `N(480,240)/(300,150)`
    - Uniform: `U(min_input_tokens,max_input_tokens)/(min_output_tokens,max_output_tokens)` or `U(max_input_tokens,max_output_tokens)`
        - Examples: `U(50,100)/(200,250)` or `U(100,200)`
    - **Prefix Repetition (for KV cache benchmarking)**: `P(prefix_len,suffix_len)/output_len`
        - Example: `P(2000,500)/200`
        - All requests share the same prefix (first request caches it, subsequent requests reuse cached KV)
        - Each request has a unique suffix to ensure different completions
        - Useful for benchmarking automatic prefix caching (APC), chunked prefill, and TTFT improvements

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
