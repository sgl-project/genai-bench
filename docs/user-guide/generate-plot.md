# Generate a 2x4 Plot

You can check out `genai-bench plot --help` to find how to generate a 2x4 Plot containing:

1. Output Inference Speed (tokens/s) vs Output Throughput of Server (tokens/s)
2. TTFT (s) vs Output Throughput of Server (tokens/s)
3. Mean E2E Latency (s) per Request vs RPS
4. Error Rates by HTTP Status vs Concurrency
5. Output Inference Speed per Request (tokens/s) vs Total Throughput (Input + Output) of Server (tokens/s)
6. TTFT (s) vs Total Throughput (Input + Output) of Server (tokens/s)
7. P90 E2E Latency (s) per Request vs RPS
8. P99 E2E Latency (s) per Request vs RPS

**Note**: TTFT plots automatically use logarithmic scale for better visualization of the wide range of values. You can override this by specifying `"y_scale": "linear"` in custom plot configurations.

```shell
genai-bench plot --experiments-folder <path-to-experiment-folder> --group-key traffic_scenario
```