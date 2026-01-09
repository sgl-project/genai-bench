# Performance Troubleshooting Guide

When benchmark results show unexpected performance, check these common issues before escalating.

## Quick Checklist

### 1. Apple-to-Apple Comparison

Ensure you're comparing equivalent configurations:

#### Quantization
- Verify both deployments use the same quantization (FP16, INT8, FP8, etc.)
- Different quantization levels significantly impact throughput and latency

#### Input/Output Token Counts
- Check actual request and response sizes in your deployment metrics
- **Quick token estimation**: `response_size_bytes / 4 ≈ token_count` (1 token ≈ 4 characters ≈ 4 bytes)
- **Streaming overhead**: Response size may be larger than actual token count due to streaming mode's extra statistics output. Disable with `--disable-streaming` for accurate measurements

#### Latency Metrics
- **TTFT (Time to First Token)**: Measures prompt processing speed
- **E2E (End-to-End) Latency**: Total request time including all token generation
- Don't compare TTFT with E2E latency—they measure different things

#### Queuing Latency
- Compare **End-to-end response time** vs **Inference time** in deployment metrics
- Large differences indicate queuing delays (requests waiting for available capacity)
- High queuing latency suggests the deployment needs more replicas or the concurrency is too high

### 2. Speculative Decoding Considerations

When speculative decoding is enabled (e.g., `speculative_decoding_mode: LOOKAHEAD_DECODING` in config.yaml):

- **Use real datasets**: Speculative decoding performs well only with realistic data patterns
- **Random data = poor performance**: Acceptance rate is low with random tokens, negating the benefits of speculative decoding
- **Always benchmark with production-representative data** when speculative decoding is enabled

#### Common Mistake: Token Distribution Override

The `--traffic-scenario` distribution format (e.g., `N(2700,2300)/(350,150)`) **overwrites** the actual dataset content with synthetic data. This defeats the purpose of using a real dataset for speculative decoding benchmarks.

**Correct approach:**

```bash
--traffic-scenario dataset \
--additional-request-params '{"max_tokens": 100}'
```

This uses the real dataset for input tokens while controlling max output tokens.

## Using Real Datasets

When benchmarking speculative decoding or production-representative workloads, use `--traffic-scenario dataset` instead of token distribution formats.

### Example 1: HuggingFace Dataset

```bash
export MODEL="zed-industries/zeta"
genai-bench benchmark \
  --api-backend openai \
  --qps-level 4 \
  --api-base "http://localhost:8000/v1/chat/completions" \
  --dataset-config "data_config.json" \
  --task "text-to-text" \
  --api-model-name "${MODEL}" \
  --model-tokenizer "${MODEL}" \
  --num-workers 8 \
  --max-requests-per-run 5000 \
  --additional-request-params '{"max_tokens": 512}' \
  --max-time-per-run 20 \
  --disable-streaming \
  --execution-engine async \
  --traffic-scenario dataset
```

Where `data_config.json`:

```json
{
  "source": {
    "type": "huggingface",
    "path": "zed-industries/zeta",
    "huggingface_kwargs": {
      "split": "train",
      "revision": "main"
    }
  },
  "prompt_column": "input"
}
```

### Example 2: Local CSV Dataset

```bash
genai-bench benchmark \
  --api-backend openai \
  --api-base "http://localhost:8000/v1/chat/completions" \
  --task "text-to-text" \
  --api-model-name "your-model" \
  --model-tokenizer "your-model" \
  --dataset-path shared_dataset_input100_output100.csv \
  --dataset-prompt-column "prompt" \
  --traffic-scenario dataset \
  --additional-request-params '{"max_tokens": 100, "ignore_eos": true}' \
  --max-time-per-run 10 \
  --max-requests-per-run 1000
```

### Key Points

| Scenario | Traffic Scenario | Result |
|----------|------------------|--------|
| Synthetic load testing | `D(100,100)`, `N(480,240)/(300,150)` | Generates random tokens matching distribution |
| Real dataset benchmarking | `dataset` | Uses actual dataset prompts |
| Speculative decoding testing | `dataset` | Required for realistic acceptance rates |

## Summary

Before reporting unexpected performance:

1. Verify quantization matches between compared deployments
2. Check actual token counts (not just configured values)
3. Distinguish TTFT vs E2E latency
4. Look for queuing latency in deployment metrics
5. Use `--traffic-scenario dataset` for speculative decoding benchmarks
6. Use `--disable-streaming` for accurate response size measurements
