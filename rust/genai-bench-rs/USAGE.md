# GenAI Bench RS - Usage Guide

## Status

- ✅ **Phase 1: OpenAI E2E MVP - COMPLETE**
- ✅ **Phase 2: Output Formats - COMPLETE**
- ✅ **Phase 3: Visualization - COMPLETE**

The OpenAI end-to-end benchmark is fully functional with comprehensive export and visualization capabilities!

## Quick Start

### Installation

```bash
cd genai-bench/rust/genai-bench-rs
cargo build --release
```

The binary will be available at `target/release/genai-bench-rs`

### Basic Usage

#### 1. Simple Sequential Benchmark (Non-Streaming)

```bash
# Set your API key
export API_KEY="sk-your-openai-key"

# Run 10 requests sequentially
cargo run --release -- \
  --provider openai \
  --model gpt-3.5-turbo \
  --num-requests 10
```

#### 2. Streaming Benchmark (Recommended for TTFT)

**Important:** Streaming mode provides accurate Time to First Token (TTFT) measurements!

```bash
# Run with streaming enabled
cargo run --release -- \
  --provider openai \
  --model gpt-3.5-turbo \
  --num-requests 10 \
  --stream
```

**Why use streaming?**
- ✅ Accurate TTFT measurement (time to first token)
- ✅ Real-world usage pattern (most apps use streaming)
- ✅ Better user experience insights
- ✅ Token-by-token latency tracking

**When to use non-streaming?**
- Testing legacy non-streaming endpoints
- Comparing with streaming performance
- Bulk batch processing scenarios

#### 3. Concurrent Benchmark

```bash
# Run 100 requests with 10 concurrent workers
cargo run --release -- \
  --provider openai \
  --model gpt-3.5-turbo \
  --num-requests 100 \
  --concurrency 10
```

#### 3. With Traffic Scenarios

**Deterministic** - Fixed delay between requests:
```bash
cargo run --release -- \
  --provider openai \
  --model gpt-3.5-turbo \
  --num-requests 50 \
  --scenario "D(100,100)" \
  --concurrency 5
```

**Normal Distribution** - Delays from normal distribution (mean=100ms, std=20ms):
```bash
cargo run --release -- \
  --provider openai \
  --model gpt-3.5-turbo \
  --num-requests 50 \
  --scenario "N(100,20)" \
  --concurrency 5
```

**Uniform Distribution** - Random delays between min and max (50-150ms):
```bash
cargo run --release -- \
  --provider openai \
  --model gpt-3.5-turbo \
  --num-requests 50 \
  --scenario "U(50,150)" \
  --concurrency 5
```

#### 4. With Custom Dataset

Create a text file with prompts (one per line):

```bash
# prompts.txt
What is the capital of France?
Explain quantum computing in simple terms.
Write a haiku about programming.
What are the benefits of Rust?
```

Run with dataset:
```bash
cargo run --release -- \
  --provider openai \
  --model gpt-3.5-turbo \
  --num-requests 20 \
  --dataset-path prompts.txt \
  --concurrency 4
```

#### 5. With Custom Parameters

```bash
cargo run --release -- \
  --provider openai \
  --model gpt-4 \
  --num-requests 50 \
  --max-tokens 200 \
  --temperature 0.7 \
  --concurrency 10 \
  --scenario "D(50,50)"
```

## CLI Options Reference

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--provider` | `-p` | `openai` | Provider to benchmark |
| `--api-key` | `-k` | (env: `API_KEY`) | API key for authentication |
| `--base-url` | `-b` | `https://api.openai.com/v1` | Base URL for API |
| `--model` | `-m` | `gpt-3.5-turbo` | Model name |
| `--num-requests` | `-n` | `100` | Total number of requests |
| `--concurrency` | `-c` | `1` | Number of concurrent requests |
| `--scenario` | `-s` | `D(0,0)` | Traffic scenario pattern |
| `--dataset-path` | `-d` | (none) | Path to prompts file |
| `--default-prompt` | | `Hello, how are you today?` | Default prompt if no dataset |
| `--max-tokens` | | `100` | Max tokens in response |
| `--temperature` | | `1.0` | Sampling temperature (0-2) |
| `--stream` | | `false` | Enable streaming (SSE) mode |
| `--output-dir` | | `results` | Output directory for exports and plots |
| `--excel` | | `false` | Export results to Excel (.xlsx) |
| `--csv` | | `false` | Export results to CSV |
| `--json` | | `false` | Export results to JSON |
| `--plot` | | `false` | Generate visualizations (PNG) |
| `--verbose` | `-v` | `false` | Enable verbose logging |

## Scenario Format

Scenarios control the timing pattern of requests:

### D(delay, delay) - Deterministic
Fixed delay between requests.

Example: `D(100,100)` - Wait 100ms between each request

### N(mean, std_dev) - Normal Distribution
Delays sampled from normal distribution.

Example: `N(100,20)` - Average 100ms delay with 20ms standard deviation

### U(min, max) - Uniform Distribution
Random delay between min and max.

Example: `U(50,150)` - Random delay between 50-150ms

## Output Explanation

### Sample Output

```
======================================================================
   GenAI Bench - High-Performance LLM Benchmarking
======================================================================

Configuration:
  Provider:     openai
  Model:        gpt-3.5-turbo
  Requests:     100
  Concurrency:  10
  Scenario:     D(0,0)
======================================================================

Starting benchmark...

 ✔ [00:00:15] [####################] 100/100 (00:00:00) Benchmark complete

======================================================================
   Benchmark Results
======================================================================

📊 Overall Statistics:
  Total Requests:       100
  Successful:           100 (100.0%)
  Failed:               0

⏱️  Time to First Token (TTFT):
  Average:              245.50 ms
  Median (P50):         240.00 ms
  95th Percentile:      280.00 ms
  99th Percentile:      295.00 ms

⏰ Total Request Time:
  Average:              450.25 ms
  Median (P50):         445.00 ms
  95th Percentile:      520.00 ms
  99th Percentile:      550.00 ms

🔢 Token Statistics:
  Total Input Tokens:   2500
  Total Output Tokens:  10000
  Total Tokens:         12500
  Avg Tokens/Second:    22.22

======================================================================
```

### Metrics Explained

**TTFT (Time to First Token)**
- Time from request start until first token arrives
- Measures initial response latency
- Lower is better for user experience
- **⚠️ Only accurate in `--stream` mode!** In non-streaming, TTFT equals total time

**Total Request Time**
- Complete end-to-end request duration
- Includes TTFT + token generation
- Important for throughput calculations

**Token Statistics**
- Input tokens: Prompt size
- Output tokens: Generated response size
- Tokens/second: Generation speed (output tokens / total time)

**Percentiles**
- P50 (Median): 50% of requests were faster
- P95: 95% of requests were faster (good for SLA)
- P99: 99% of requests were faster (captures tail latency)

## Advanced Examples

### High-Concurrency Stress Test (Streaming)

```bash
cargo run --release -- \
  --provider openai \
  --model gpt-3.5-turbo \
  --num-requests 1000 \
  --concurrency 50 \
  --scenario "D(0,0)" \
  --stream
```

### Realistic Traffic Pattern (Streaming)

```bash
# Simulates realistic user traffic with normal distribution
cargo run --release -- \
  --provider openai \
  --model gpt-4 \
  --num-requests 500 \
  --concurrency 20 \
  --scenario "N(200,50)" \
  --max-tokens 150 \
  --temperature 0.8 \
  --dataset-path customer_queries.txt \
  --stream
```

### TTFT-Focused Benchmark

```bash
# Measure Time to First Token with low concurrency for accuracy
cargo run --release -- \
  --provider openai \
  --model gpt-3.5-turbo \
  --num-requests 100 \
  --concurrency 5 \
  --scenario "D(100,100)" \
  --stream
```

### Streaming vs Non-Streaming Comparison

```bash
# Non-streaming
cargo run --release -- \
  --model gpt-3.5-turbo \
  --num-requests 50 \
  --concurrency 10

# Streaming (compare TTFT!)
cargo run --release -- \
  --model gpt-3.5-turbo \
  --num-requests 50 \
  --concurrency 10 \
  --stream
```

### Low-Latency Baseline

```bash
# Test absolute best-case latency
cargo run --release -- \
  --provider openai \
  --model gpt-3.5-turbo \
  --num-requests 10 \
  --concurrency 1 \
  --scenario "D(1000,1000)" \
  --max-tokens 50 \
  --stream
```

## Environment Variables

You can use environment variables instead of CLI flags:

```bash
export API_KEY="sk-your-key"
export RUST_LOG=info  # Enable logging (debug, info, warn, error)

cargo run --release -- --num-requests 100
```

## Logging

Enable verbose logging for debugging:

```bash
# Info level
RUST_LOG=info cargo run --release -- --num-requests 10

# Debug level (very verbose)
RUST_LOG=debug cargo run --release -- --num-requests 10
```

## Performance Tips

### 1. Use Release Mode

Always use `--release` for accurate benchmarks:
```bash
cargo build --release
./target/release/genai-bench-rs --num-requests 100
```

### 2. Adjust Concurrency

- Start with low concurrency (1-5) to measure baseline latency
- Increase gradually to find optimal throughput
- Too much concurrency may cause rate limiting or degraded performance

### 3. Scenario Selection

- **D(0,0)** - Maximum throughput test
- **D(100,100)** - Constant rate (10 req/sec)
- **N(100,20)** - Realistic variable load
- **U(50,150)** - Random traffic spike simulation

### 4. Dataset Best Practices

- Use realistic prompts from your application
- Vary prompt lengths for accurate testing
- Include edge cases (very short, very long)
- One prompt per line in text file

## Troubleshooting

### Rate Limiting

If you see many 429 errors:
- Reduce `--concurrency`
- Increase scenario delay: `--scenario "D(500,500)"`
- Check your OpenAI tier limits

### Connection Errors

If requests timeout:
- Check network connectivity
- Verify API key is correct
- Try reducing concurrency
- Increase timeout (not yet configurable, coming soon)

### Memory Issues

For very large benchmarks:
- The tool streams results efficiently
- Should handle 10K+ requests without issues
- If problems occur, break into smaller batches

## Export Formats (Phase 2 - COMPLETE ✅)

### Excel Export

Export results to a professional multi-sheet Excel workbook:

```bash
cargo run --release -- \
  --api-key $OPENAI_API_KEY \
  --num-requests 100 \
  --stream \
  --excel \
  --output-dir benchmark-results
```

Creates `benchmark-results/results.xlsx` with 3 sheets:
1. **Summary** - Aggregated metrics (averages, percentiles, totals)
2. **Raw Data** - All individual request details
3. **Percentiles** - P10, P25, P50, P75, P90, P95, P99 breakdown

### CSV Export

Export to CSV for analysis in Excel, Python, R, etc:

```bash
cargo run --release -- \
  --api-key $OPENAI_API_KEY \
  --num-requests 100 \
  --csv
```

Creates two files:
- `results/results.csv` - All request details (request_num, ttft_ms, total_time_ms, tokens, etc.)
- `results/summary.csv` - Aggregated summary statistics

### JSON Export

Export structured JSON for programmatic access:

```bash
cargo run --release -- \
  --api-key $OPENAI_API_KEY \
  --num-requests 100 \
  --json
```

Creates `results/results.json` with:
```json
{
  "summary": {
    "overall": { "total_requests": 100, "success_rate": 100.0, ... },
    "ttft": { "avg": 245.5, "p50": 240, "p95": 280, "p99": 295 },
    "total_time": { "avg": 450.2, "p50": 445, "p95": 520, "p99": 550 },
    "tokens": { "total_input": 2500, "total_output": 10000, ... }
  },
  "raw_data": [
    { "request_num": 1, "ttft_ms": 245, "total_time_ms": 450, ... },
    ...
  ]
}
```

### Complete Export Example

Export all formats at once:

```bash
cargo run --release -- \
  --api-key $OPENAI_API_KEY \
  --model gpt-3.5-turbo \
  --num-requests 200 \
  --concurrency 20 \
  --stream \
  --excel \
  --csv \
  --json \
  --output-dir results
```

## Visualization (Phase 3 - COMPLETE ✅)

### Generate Plots

Create comprehensive visualizations automatically:

```bash
cargo run --release -- \
  --api-key $OPENAI_API_KEY \
  --num-requests 100 \
  --stream \
  --plot \
  --output-dir results
```

Generates 7 PNG plots in `results/`:

1. **ttft_histogram.png** - Time to First Token distribution
   - Shows latency distribution for first token
   - Helps identify TTFT patterns and outliers

2. **total_time_histogram.png** - Total request time distribution
   - End-to-end latency distribution
   - Useful for understanding overall performance

3. **tokens_histogram.png** - Completion tokens distribution
   - Shows variability in response sizes
   - Helps understand token usage patterns

4. **throughput.png** - Token throughput over time
   - Cumulative tokens per second
   - Shows performance trends during benchmark

5. **rps.png** - Requests per second over time
   - Time-windowed RPS visualization
   - Helps identify rate limiting or throttling

6. **percentiles.png** - Latency percentiles (P10-P99)
   - Compares TTFT vs Total Time across percentiles
   - Critical for SLA validation

7. **cdf.png** - Cumulative Distribution Function
   - TTFT CDF with P50, P95, P99 markers
   - Shows what % of requests meet latency targets

### Complete Benchmark with All Outputs

Run comprehensive benchmark with all exports and visualizations:

```bash
cargo run --release -- \
  --api-key $OPENAI_API_KEY \
  --model gpt-3.5-turbo \
  --num-requests 200 \
  --concurrency 20 \
  --scenario "N(50,10)" \
  --dataset-path prompts.txt \
  --stream \
  --excel \
  --csv \
  --json \
  --plot \
  --output-dir benchmark-results \
  --verbose
```

This creates:
```
benchmark-results/
├── results.xlsx          # Excel with 3 sheets
├── results.csv           # Raw data CSV
├── summary.csv           # Summary CSV
├── results.json          # JSON export
├── ttft_histogram.png    # TTFT distribution
├── total_time_histogram.png  # Total time distribution
├── tokens_histogram.png  # Token distribution
├── throughput.png        # Throughput over time
├── rps.png              # Requests per second
├── percentiles.png      # Percentile comparison
└── cdf.png              # CDF chart
```

## What's Next

### Phase 4: Advanced UI (Coming Soon)
- Live dashboard with ratatui
- Real-time metrics during benchmark
- Interactive TUI with drill-down capabilities

### Phase 5: Multi-Cloud (Lowest Priority)
- Azure OpenAI
- GCP Vertex AI
- AWS Bedrock
- Anthropic Claude

## Examples by Use Case

### SLA Validation

```bash
# Verify 95% of requests complete under 500ms
cargo run --release -- \
  --api-key $OPENAI_API_KEY \
  --model gpt-3.5-turbo \
  --num-requests 1000 \
  --concurrency 20 \
  --scenario "N(100,30)" \
  --stream \
  --excel \
  --plot
# Check P95 in Excel or percentiles.png
```

### Capacity Planning

```bash
# Find maximum sustainable concurrency
for c in 10 20 30 40 50; do
  echo "Testing concurrency: $c"
  cargo run --release -- \
    --model gpt-3.5-turbo \
    --num-requests 100 \
    --concurrency $c \
    --scenario "D(0,0)"
done
```

### Model Comparison

```bash
# Compare gpt-3.5 vs gpt-4
for model in gpt-3.5-turbo gpt-4; do
  echo "Testing $model"
  cargo run --release -- \
    --model $model \
    --num-requests 50 \
    --concurrency 5 \
    --dataset-path test_prompts.txt
done
```

## Support

For issues or questions:
- File an issue on GitHub
- Check ARCHITECTURE.md for design details
- See rust-implementation-plan-v2.md for roadmap

---

**Status:**
- ✅ Phase 1 Complete (OpenAI E2E MVP with streaming)
- ✅ Phase 2 Complete (Excel/CSV/JSON export)
- ✅ Phase 3 Complete (Plotting and visualization)

**Next:** Phase 4 - Advanced UI with ratatui (live dashboard)
