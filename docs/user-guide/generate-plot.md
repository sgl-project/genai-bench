# Performance Visualization & Plotting

## Quick Start
You can check out `genai-bench plot --help` to find how to generate a 2x4 Plot containing:

1. Per-Request Inference Speed (tokens/s) vs Server Output Throughput (tokens/s)
2. TTFT (s) vs Server Output Throughput (tokens/s)
3. Mean E2E Latency (s) per Request vs RPS
4. Error Rates by HTTP Status vs Concurrency
5. Per-Request Inference Speed (tokens/s) vs Server Total Throughput (Input + Output) (tokens/s)
6. TTFT (s) vs Server Total Throughput (Input + Output) (tokens/s)
7. P90 E2E Latency (s) per Request vs RPS
8. P99 E2E Latency (s) per Request vs RPS

**Note**: TTFT plots automatically use logarithmic scale for better visualization of the wide range of values. You can override this by specifying `"y_scale": "linear"` in custom plot configurations.

```shell
genai-bench plot --experiments-folder <path-to-experiment-folder> --group-key traffic_scenario
```

## Advanced Plot Configuration

This section provides comprehensive configuration examples and templates for customizing GenAI Bench's flexible plotting system to meet your specific analysis needs.

### Usage

Use these configurations with the `genai-bench plot` command:

```bash
# Use a custom configuration file
genai-bench plot --experiments-folder /path/to/experiments \
                 --group-key traffic_scenario \
                 --plot-config examples/plot_configs/custom_2x2.json

# Use a built-in preset for multiple scenarios
genai-bench plot --experiments-folder /path/to/experiments \
                 --group-key traffic_scenario \
                 --preset simple_2x2

# Use multi-line preset for single scenario analysis
genai-bench plot --experiments-folder /path/to/experiments \
                 --group-key none \
                 --preset single_scenario_analysis

# List available fields with actual data from your experiment
genai-bench plot --experiments-folder /path/to/experiments \
                 --group-key traffic_scenario \
                 --list-fields

# Validate a configuration without generating plots
genai-bench plot --experiments-folder /path/to/experiments \
                 --group-key traffic_scenario \
                 --plot-config examples/plot_configs/custom_2x2.json \
                 --validate-only
```

### Available Configurations

#### [custom_2x2.json](https://github.com/sgl-project/genai-bench/blob/main/examples/plot_configs/custom_2x2.json)
A simple 2x2 grid layout focusing on key performance metrics:
- Throughput vs Mean Latency
- RPS vs P99 Latency
- Concurrency vs TTFT
- Error Rate Analysis

#### [performance_focused.json](https://github.com/sgl-project/genai-bench/blob/main/examples/plot_configs/performance_focused.json)
A comprehensive 2x3 grid for detailed performance analysis:
- Token generation speed analysis
- Time to first token trends
- Latency percentiles
- Token efficiency scatter plot
- Request success rates
- Throughput scaling

#### [multi_line_latency.json](https://github.com/sgl-project/genai-bench/blob/main/examples/plot_configs/multi_line_latency.json)
Demonstrates multi-line plotting capabilities with a 2x2 layout:
- **Latency Percentiles Comparison**: Multiple latency percentiles (mean, P90, P99) on one plot
- **TTFT Performance Analysis**: Mean and P95 TTFT comparison
- **Token Processing Speed**: Output speed vs input throughput comparison
- **Request Success Metrics**: Single-line error rate plot

#### [comprehensive_multi_line.json](https://github.com/sgl-project/genai-bench/blob/main/examples/plot_configs/comprehensive_multi_line.json)
Advanced multi-line example with 1x3 layout showcasing complex comparisons:
- **E2E Latency Distribution**: All percentiles (P25, P50, P75, P90, P99) with custom colors
- **Throughput Components**: Input, output, and total throughput comparison
- **Token Statistics**: Input, output, and total token counts as scatter plot

### Configuration Format

Plot configurations use the following JSON schema:

#### Single-Line Plots
```json
{
  "layout": {
    "rows": 2,
    "cols": 2,
    "figsize": [16, 12]  // Optional: [width, height] in inches
  },
  "plots": [
    {
      "title": "Plot Title",
      "x_field": "field.path.from.AggregatedMetrics",
      "y_field": "another.field.path",           // Single field
      "x_label": "Custom X Label",              // Optional
      "y_label": "Custom Y Label",              // Optional
      "plot_type": "line",                      // line, scatter, or bar
      "position": [0, 0]                        // [row, col] in grid
    }
  ]
}
```

#### Multi-Line Plots
```json
{
  "plots": [
    {
      "title": "Multi-Line Comparison",
      "x_field": "requests_per_second",
      "y_fields": [                             // Multiple fields on same plot
        {
          "field": "stats.e2e_latency.mean",
          "label": "Mean Latency",              // Optional custom label
          "color": "blue",                      // Optional custom color
          "linestyle": "-"                      // Optional: '-', '--', '-.', ':'
        },
        {
          "field": "stats.e2e_latency.p90",
          "label": "P90 Latency",
          "color": "red",
          "linestyle": "--"
        }
      ],
      "x_label": "RPS",
      "y_label": "Latency (s)",
      "plot_type": "line",
      "position": [0, 0]
    }
  ]
}
```

#### Key Features

- **Single vs Multi-Line**: Use `y_field` for single line, `y_fields` for multiple lines
- **Custom Styling**: Each line can have custom color, linestyle, and label
- **Flexible Layout**: Any NxM grid layout from 1x1 to 5x6
- **Plot Types**: `line`, `scatter`, `bar` (multi-line bar creates grouped bars)
- **Automatic Legends**: Multi-line plots automatically generate legends

#### When to Use Multi-Line Plots

✅ **GOOD Use Cases:**
- **Single scenario analysis**: Use `--group-key ""` (empty string) for one traffic scenario
- **Deep metric comparison**: Comparing mean, P90, P99 latency on same plot
- **Performance analysis**: Related metrics on the same scale

❌ **AVOID Multi-Line Plots When:**
- **Multiple scenarios**: `--group-key traffic_scenario` with multiple scenarios
- **Multiple server versions**: `--group-key server_version`
- **Any grouping**: Multi-line + grouping creates cluttered, hard-to-read plots

The system will automatically convert multi-line plots to single-line plots when it detects multiple groups/scenarios for better visualization.

#### Usage Patterns

```bash
# ✅ GOOD: Multi-line for single scenario analysis
genai-bench plot --preset single_scenario_analysis --group-key ""

# ✅ GOOD: Single-line for multiple scenarios
genai-bench plot --preset 2x4_default --group-key traffic_scenario

# ⚠️ AUTO-CONVERTED: Multi-line + grouping → single-line
genai-bench plot --preset multi_line_latency --group-key traffic_scenario
```

### Available Fields

Run `genai-bench plot --experiments-folder /path/to/experiments --group-key traffic_scenario --list-fields` to see all available field paths with actual data from your experiments.

Common field paths include:

#### Direct Metrics
- `num_concurrency` - Concurrency level
- `requests_per_second` - RPS
- `error_rate` - Error rate
- `mean_output_throughput_tokens_per_s` - Server output throughput
- `mean_total_tokens_throughput_tokens_per_s` - Total throughput
- `run_duration` - Duration of the run

#### Statistical Fields
Access statistics using `stats.{metric}.{statistic}`:

**Metrics:** ttft, tpot, e2e_latency, output_latency, output_inference_speed, num_input_tokens, num_output_tokens, total_tokens, input_throughput, output_throughput

**Statistics:** min, max, mean, stddev, sum, p25, p50, p75, p90, p95, p99

**Examples:**
- `stats.ttft.mean` - Mean time to first token
- `stats.e2e_latency.p99` - 99th percentile end-to-end latency
- `stats.output_inference_speed.mean` - Mean output inference speed

### Built-in Presets

#### 2x4_default
The original 2x4 layout with all 8 standard plots. This maintains backwards compatibility with the existing plotting system.

#### simple_2x2
A simplified 2x2 layout with the most important metrics for quick analysis.

### Creating Custom Configurations

1. Start with an example configuration
2. Modify the layout dimensions and plot specifications
3. Use `--list-fields` to find available metrics
4. Use `--validate-only` to test your configuration
5. Generate plots with your custom config

### Tips

- Use descriptive titles for your plots
- Choose appropriate plot types (line for trends, scatter for relationships, bar for comparisons)
- Ensure field paths are valid using `--validate-only`
- Consider your audience when selecting metrics to display
- Use figsize to adjust the output image dimensions
