# Results Analysis

This guide explains how to analyze benchmark results, understand the metrics, and generate comprehensive reports.

## Understanding Benchmark Results

After running a benchmark, GenAI Bench generates detailed results that help you understand your system's performance characteristics.

### Result Structure

```
experiments/
└── your-experiment/
    ├── results.json          # Raw benchmark data
    ├── metrics.json          # Aggregated metrics
    ├── logs/                 # Detailed logs
    │   ├── benchmark.log
    │   └── errors.log
    └── plots/                # Generated visualizations
        └── performance.png
```

## Key Metrics Explained

### Single-Request Metrics

These metrics capture performance for individual requests:

#### Time Metrics

| Metric | Description | Formula | Interpretation |
|--------|-------------|---------|----------------|
| **TTFT** | Time to First Token | `time_at_first_token - start_time` | Initial response time |
| **E2E Latency** | End-to-End latency | `end_time - start_time` | Total request time |
| **TPOT** | Time Per Output Token | `(e2e_latency - TTFT) / (num_output_tokens - 1)` | Generation speed |

#### Throughput Metrics

| Metric | Description | Formula | Unit |
|--------|-------------|---------|------|
| **Input Throughput** | Input processing rate | `num_input_tokens / TTFT` | tokens/second |
| **Output Throughput** | Output generation rate | `(num_output_tokens - 1) / output_latency` | tokens/second |
| **Inference Speed** | Overall generation speed | `1 / TPOT` | tokens/second |

#### Token Metrics

| Metric | Description | Unit |
|--------|-------------|------|
| **Input Tokens** | Number of prompt tokens | tokens |
| **Output Tokens** | Number of generated tokens | tokens |
| **Total Tokens** | Total tokens processed | tokens |

### Aggregated Metrics

These metrics summarize performance across multiple requests:

#### System Performance

| Metric | Description | Formula | Unit |
|--------|-------------|---------|------|
| **Mean Input Throughput** | Average input processing | `sum(input_tokens) / run_duration` | tokens/second |
| **Mean Output Throughput** | Average output generation | `sum(output_tokens) / run_duration` | tokens/second |
| **Total Tokens Throughput** | Overall processing rate | `sum(total_tokens) / run_duration` | tokens/second |
| **Requests Per Minute** | Request processing rate | `num_completed_requests / duration * 60` | requests/minute |

#### Quality Metrics

| Metric | Description | Formula | Unit |
|--------|-------------|---------|------|
| **Error Rate** | Percentage of failed requests | `num_error_requests / total_requests` | percentage |
| **Success Rate** | Percentage of successful requests | `num_completed_requests / total_requests` | percentage |
| **Total Chars Per Hour** | Character processing capacity | `total_tokens_throughput * chars_per_token * 3600` | characters/hour |

## Generating Reports

### Excel Reports

Generate comprehensive Excel reports with all metrics:

```bash
genai-bench excel \
    --experiment-dir "./experiments/my-benchmark" \
    --output-file "detailed_report.xlsx" \
    --include-raw-data \
    --include-pricing
```

#### Excel Report Contents

1. **Summary Sheet**: High-level metrics and statistics
2. **Raw Data**: Individual request metrics
3. **Aggregated Metrics**: System-level performance
4. **Pricing Analysis**: Cost calculations
5. **Error Analysis**: Failed request details

### Custom Plots

Generate visualizations with custom configurations:

```bash
genai-bench plot \
    --experiment-dirs "./exp1,./exp2,./exp3" \
    --output-file "comparison.png" \
    --plot-config "my_config.yaml"
```

#### Plot Configuration

Create a YAML configuration file:

```yaml
# plot_config.yaml
layout:
  rows: 2
  cols: 4

plots:
  - title: "Throughput vs Concurrency"
    type: "line"
    x_axis: "num_users"
    y_axis: "mean_total_tokens_throughput"
    color_by: "server_engine"
    
  - title: "Latency vs Concurrency"
    type: "line"
    x_axis: "num_users"
    y_axis: "mean_e2e_latency"
    color_by: "server_engine"
    
  - title: "Error Rate vs Concurrency"
    type: "bar"
    x_axis: "num_users"
    y_axis: "error_rate"
    
  - title: "Token Distribution"
    type: "histogram"
    x_axis: "num_output_tokens"
    bins: 20
```

## Interpreting Results

### Performance Analysis

#### Throughput Analysis

1. **Identify Peak Throughput**: Find the maximum sustainable throughput
2. **Bottleneck Detection**: Look for performance degradation points
3. **Scaling Efficiency**: Analyze how performance scales with concurrency

#### Latency Analysis

1. **Response Time Distribution**: Understand latency percentiles
2. **TTFT vs E2E**: Compare initial vs total response times
3. **Latency Stability**: Check for latency spikes or inconsistencies

#### Error Analysis

1. **Error Patterns**: Identify common error types
2. **Error Rate Trends**: Monitor error rates under different loads
3. **Recovery Behavior**: Test system recovery after errors

### Comparative Analysis

#### Model Comparison

```bash
# Run benchmarks for different models
for model in "gpt-3.5-turbo" "gpt-4" "llama-2-7b"; do
    genai-bench benchmark \
        --api-model-name "$model" \
        --output-dir "./experiments/$model" \
        --max-time-per-run 300
done

# Generate comparison
genai-bench plot \
    --experiment-dirs "./experiments/gpt-3.5-turbo,./experiments/gpt-4,./experiments/llama-2-7b" \
    --output-file "model_comparison.png" \
    --group-by "api_model_name"
```

#### Configuration Comparison

Compare different configurations:

- **Concurrency Levels**: Test different user counts
- **Traffic Patterns**: Compare constant vs burst vs ramp
- **Model Parameters**: Test different temperature, max_tokens settings

### Performance Optimization

#### Identifying Bottlenecks

1. **CPU Bottlenecks**: High CPU usage, low throughput
2. **Memory Bottlenecks**: High memory usage, OOM errors
3. **Network Bottlenecks**: High latency, connection timeouts
4. **GPU Bottlenecks**: Low GPU utilization, long inference times

#### Optimization Strategies

1. **Batch Processing**: Increase batch sizes for better throughput
2. **Model Optimization**: Use quantized or optimized models
3. **Infrastructure Scaling**: Add more resources or instances
4. **Request Optimization**: Optimize prompt length and parameters

## Advanced Analysis

### Statistical Analysis

#### Percentile Analysis

```python
import pandas as pd

# Load results
df = pd.read_json("experiments/my-benchmark/results.json")

# Calculate percentiles
percentiles = df['e2e_latency'].quantile([0.5, 0.9, 0.95, 0.99])
print(f"P50: {percentiles[0.5]:.3f}s")
print(f"P90: {percentiles[0.9]:.3f}s")
print(f"P95: {percentiles[0.95]:.3f}s")
print(f"P99: {percentiles[0.99]:.3f}s")
```

#### Trend Analysis

```python
# Analyze trends over time
df['timestamp'] = pd.to_datetime(df['start_time'])
hourly_stats = df.groupby(df['timestamp'].dt.hour)['e2e_latency'].agg(['mean', 'std'])
```

### Custom Metrics

#### Business Metrics

Calculate business-relevant metrics:

```python
# Cost per request
cost_per_request = (input_tokens * input_cost + output_tokens * output_cost)

# Revenue per hour
revenue_per_hour = requests_per_hour * price_per_request

# ROI calculation
roi = (revenue_per_hour - cost_per_hour) / cost_per_hour
```

#### Quality Metrics

```python
# Response quality (if available)
quality_score = calculate_quality_score(responses)

# Consistency metrics
latency_consistency = df['e2e_latency'].std() / df['e2e_latency'].mean()
```

## Reporting Best Practices

### Report Structure

1. **Executive Summary**: High-level findings and recommendations
2. **Methodology**: Benchmark setup and configuration
3. **Results**: Detailed metrics and analysis
4. **Conclusions**: Key insights and next steps

### Visualization Guidelines

1. **Use Appropriate Charts**: Line charts for trends, bar charts for comparisons
2. **Include Error Bars**: Show confidence intervals where applicable
3. **Consistent Formatting**: Use consistent colors, fonts, and scales
4. **Clear Labels**: Provide descriptive titles and axis labels

### Documentation

1. **Record Configuration**: Document all benchmark parameters
2. **Note Environment**: Record hardware, software, and network conditions
3. **Track Changes**: Version control your benchmark configurations
4. **Share Context**: Provide business context for technical results

## Troubleshooting Analysis

### Common Issues

#### Missing Data
```
Error: No results found in experiment directory
```

**Solution**: Check that the benchmark completed successfully and generated results.

#### Inconsistent Results
```
Warning: High variance in latency measurements
```

**Solution**: Check for network issues, system load, or configuration problems.

#### Plot Generation Errors
```
Error: Invalid plot configuration
```

**Solution**: Validate your YAML configuration file syntax and structure.

### Getting Help

- Check the [CLI Reference](cli.md) for command options
- Review [Examples](../examples/basic-benchmarks.md) for analysis patterns
- Open an [issue](https://github.com/sgl-project/genai-bench/issues) for bugs

## Next Steps

- Learn about [Traffic Scenarios](tasks.md) for load testing
- Explore [Examples](../examples/basic-benchmarks.md) for practical analysis
- Check out [API Reference](../api/overview.md) for programmatic access 