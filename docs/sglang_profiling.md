# SGLang Profiling Support

genai-bench now supports SGLang server-side profiling, allowing you to capture GPU kernel-level performance data using Perfetto traces.

## Overview

When benchmarking SGLang servers, you can enable profiling to get detailed insights into:
- Prefill vs decode phase timing
- GPU kernel execution breakdown
- Memory usage patterns
- CPU/GPU activity correlation

## Prerequisites

- SGLang server running with profiling support (v0.4.0+)
- `--api-backend sglang` specified in genai-bench

## Usage

### Basic Profiling

```bash
genai-bench benchmark \
  --api-backend sglang \
  --api-base "http://localhost:30000" \
  --api-model-name "meta-llama/Llama-3.1-8B-Instruct" \
  --model-tokenizer "meta-llama/Llama-3.1-8B-Instruct" \
  --task text-to-text \
  --max-time-per-run 5 \
  --max-requests-per-run 100 \
  --sglang-profile
```

### Advanced Profiling Options

```bash
genai-bench benchmark \
  --api-backend sglang \
  --api-base "http://localhost:30000" \
  --api-model-name "meta-llama/Llama-3.1-8B-Instruct" \
  --model-tokenizer "meta-llama/Llama-3.1-8B-Instruct" \
  --task text-to-text \
  --max-time-per-run 5 \
  --max-requests-per-run 100 \
  --sglang-profile \
  --sglang-profile-output-dir /path/to/profiles \
  --sglang-profile-steps 20 \
  --sglang-profile-by-stage \
  --sglang-profile-activities "CPU,GPU,MEM"
```

## CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--sglang-profile` | `False` | Enable SGLang server-side profiling |
| `--sglang-profile-output-dir` | `<experiment>/profiles` | Directory to save trace files |
| `--sglang-profile-steps` | `5` | Number of forward steps to profile (matches SGLang nightly) |
| `--sglang-profile-by-stage` | `False` | Profile prefill and decode separately |
| `--sglang-profile-activities` | `CPU,GPU` | Activities to profile (CPU, GPU, MEM) |

## Viewing Traces

After the benchmark completes, trace files are saved in the specified output directory. To view them:

1. Open [Perfetto UI](https://ui.perfetto.dev) in Google Chrome
2. Click "Open trace file" or drag-and-drop the `.json.gz` trace file
3. Explore the timeline view for detailed kernel analysis

### Understanding the Traces

- **Prefill traces**: Show input token processing performance
- **Decode traces**: Show output token generation performance
- **CUDA kernels**: Identify which operations are bottlenecks

## Integration with CI/CD

For nightly benchmarks, you can publish traces to a storage backend:

```bash
genai-bench benchmark \
  --api-backend sglang \
  --api-base "http://localhost:30000" \
  --api-model-name "meta-llama/Llama-3.1-8B-Instruct" \
  --model-tokenizer "meta-llama/Llama-3.1-8B-Instruct" \
  --task text-to-text \
  --max-time-per-run 5 \
  --max-requests-per-run 100 \
  --sglang-profile \
  --upload-results \
  --storage-provider github \
  --github-token $GITHUB_TOKEN \
  --github-owner sgl-project \
  --github-repo benchmark-traces
```

## Programmatic Usage

You can also use the profiler directly in Python:

```python
from genai_bench.user.sglang_user import SGLangProfiler

# Initialize profiler
profiler = SGLangProfiler(
    base_url="http://localhost:30000",
    output_dir="/tmp/profiles",
    profile_by_stage=False,
    api_key="your-api-key",  # Optional: for authenticated SGLang servers
)

# Start profiling (captures next N forward steps)
trace_path = profiler.start_profile(
    num_steps=5,
    activities=["CPU", "GPU"],
    profile_name="my_benchmark",
)

# Run your benchmark...

# Traces are saved automatically
print(f"Traces saved to: {trace_path}")
```

## Comparison with SGLang Native Benchmarking

| Feature | genai-bench + profiling | SGLang bench_one_batch_server |
|---------|------------------------|-------------------------------|
| Load testing | Yes (concurrency sweep) | No (fixed batch sizes) |
| Perfetto traces | Yes | Yes |
| Profile by stage | Yes | Yes |
| Excel reports | Yes | No |
| Traffic scenarios | Yes (D, N, U distributions) | No |
| Server lifecycle | No (external server) | Yes (auto start/stop) |

## Troubleshooting

### "Failed to start SGLang profiling"

Ensure:
1. SGLang server is running and accessible
2. Server version supports `/start_profile` endpoint (v0.4.0+)
3. Network connectivity to the server

### Large trace files

Reduce `--sglang-profile-steps` or disable MEM profiling:
```bash
--sglang-profile-steps 5 \
--sglang-profile-activities "CPU,GPU"
```

### Profiling not capturing all requests

The profiler captures a fixed number of forward steps. If your benchmark has very low concurrency, increase the steps or run longer:
```bash
--sglang-profile-steps 50 \
--max-time-per-run 10
```
