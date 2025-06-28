# Basic Benchmarks

This guide provides practical examples of basic benchmarks you can run with GenAI Bench to get started quickly.

## Quick Start Examples

### Example 1: Simple Text-to-Text Benchmark

Run a basic chat completion benchmark:

```bash
genai-bench benchmark \
    --api-backend openai \
    --api-base "http://localhost:8082" \
    --api-key "your-api-key" \
    --api-model-name "llama-2-7b" \
    --task text-to-text \
    --num-users 2 \
    --max-time-per-run 60 \
    --dataset-name "sonnet.txt"
```

**What this does:**
- Connects to a local vLLM server
- Runs a 2-user benchmark for 60 seconds
- Uses the built-in sonnet dataset
- Generates text completions

### Example 2: Embeddings Benchmark

Test embedding generation performance:

```bash
genai-bench benchmark \
    --api-backend openai \
    --api-base "http://localhost:8082" \
    --api-key "your-api-key" \
    --api-model-name "text-embedding-ada-002" \
    --task text-to-embeddings \
    --num-users 5 \
    --max-time-per-run 120 \
    --dataset-name "sonnet.txt"
```

**What this does:**
- Tests embedding generation
- Uses 5 concurrent users
- Runs for 2 minutes
- Measures embedding latency and throughput

### Example 3: Vision Benchmark

Test image-text understanding:

```bash
genai-bench benchmark \
    --api-backend openai \
    --api-base "http://localhost:8082" \
    --api-key "your-api-key" \
    --api-model-name "llava-1.5-7b" \
    --task image-text-to-text \
    --num-users 1 \
    --max-time-per-run 180 \
    --dataset-name "vision_dataset"
```

**What this does:**
- Tests vision-language model
- Uses single user (vision models are typically slower)
- Runs for 3 minutes
- Processes images with text prompts

## Production-Ready Examples

### Example 4: Load Testing

Test system performance under load:

```bash
genai-bench benchmark \
    --api-backend openai \
    --api-base "https://your-production-api.com" \
    --api-key "your-production-key" \
    --api-model-name "gpt-4" \
    --task text-to-text \
    --num-users 20 \
    --spawn-rate 5 \
    --max-time-per-run 600 \
    --max-requests-per-run 2000 \
    --dataset-name "production_prompts.txt" \
    --ui \
    --output-dir "./production-benchmark"
```

**What this does:**
- Tests production API endpoint
- Gradually ramps up to 20 users
- Runs for 10 minutes
- Limits to 2000 requests
- Enables UI dashboard
- Saves results to specific directory

### Example 5: Model Comparison

Compare multiple models:

```bash
# Benchmark Model A
genai-bench benchmark \
    --api-backend openai \
    --api-base "http://localhost:8082" \
    --api-key "your-key" \
    --api-model-name "llama-2-7b" \
    --task text-to-text \
    --num-users 10 \
    --max-time-per-run 300 \
    --output-dir "./experiments/llama-2-7b"

# Benchmark Model B
genai-bench benchmark \
    --api-backend openai \
    --api-base "http://localhost:8082" \
    --api-key "your-key" \
    --api-model-name "mistral-7b" \
    --task text-to-text \
    --num-users 10 \
    --max-time-per-run 300 \
    --output-dir "./experiments/mistral-7b"

# Generate comparison
genai-bench plot \
    --experiment-dirs "./experiments/llama-2-7b,./experiments/mistral-7b" \
    --output-file "model_comparison.png" \
    --group-by "api_model_name"
```

### Example 6: Traffic Pattern Testing

Test different traffic scenarios:

```bash
# Constant load
genai-bench benchmark \
    --api-backend openai \
    --api-base "http://localhost:8082" \
    --api-key "your-key" \
    --api-model-name "llama-2-7b" \
    --task text-to-text \
    --traffic-scenario constant \
    --num-users 10 \
    --max-time-per-run 300 \
    --output-dir "./experiments/constant"

# Burst traffic
genai-bench benchmark \
    --api-backend openai \
    --api-base "http://localhost:8082" \
    --api-key "your-key" \
    --api-model-name "llama-2-7b" \
    --task text-to-text \
    --traffic-scenario burst \
    --num-users 50 \
    --spawn-rate 10 \
    --max-time-per-run 300 \
    --output-dir "./experiments/burst"

# Ramp-up traffic
genai-bench benchmark \
    --api-backend openai \
    --api-base "http://localhost:8082" \
    --api-key "your-key" \
    --api-model-name "llama-2-7b" \
    --task text-to-text \
    --traffic-scenario ramp \
    --num-users 20 \
    --spawn-rate 1 \
    --max-time-per-run 300 \
    --output-dir "./experiments/ramp"
```

## API-Specific Examples

### Example 7: OpenAI API

Test against OpenAI's hosted API:

```bash
genai-bench benchmark \
    --api-backend openai \
    --api-base "https://api.openai.com/v1" \
    --api-key "sk-your-openai-key" \
    --api-model-name "gpt-3.5-turbo" \
    --task text-to-text \
    --num-users 5 \
    --max-time-per-run 300 \
    --temperature 0.7 \
    --max-tokens 100
```

### Example 8: Cohere API

Test against Cohere's API:

```bash
genai-bench benchmark \
    --api-backend cohere \
    --api-base "https://api.cohere.ai" \
    --api-key "your-cohere-key" \
    --api-model-name "command-r-plus" \
    --task text-to-text \
    --num-users 5 \
    --max-time-per-run 300
```

### Example 9: OCI Cohere

Test against Oracle Cloud Infrastructure Cohere:

```bash
genai-bench benchmark \
    --api-backend oci_cohere \
    --api-base "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com" \
    --api-key "your-oci-key" \
    --api-model-name "cohere.command-r-plus" \
    --task text-to-text \
    --num-users 5 \
    --max-time-per-run 300
```

## Custom Dataset Examples

### Example 10: Custom Text Dataset

Use your own prompts:

```bash
# Create custom dataset
cat > my_prompts.txt << EOF
What is machine learning?
Explain the benefits of cloud computing.
Write a short story about a robot.
How does blockchain technology work?
EOF

# Run benchmark with custom dataset
genai-bench benchmark \
    --api-backend openai \
    --api-base "http://localhost:8082" \
    --api-key "your-key" \
    --api-model-name "llama-2-7b" \
    --task text-to-text \
    --dataset-path "./my_prompts.txt" \
    --num-users 3 \
    --max-time-per-run 180
```

### Example 11: Hugging Face Dataset

Use a dataset from Hugging Face:

```bash
genai-bench benchmark \
    --api-backend openai \
    --api-base "http://localhost:8082" \
    --api-key "your-key" \
    --api-model-name "llama-2-7b" \
    --task text-to-text \
    --dataset-name "squad" \
    --dataset-split "train" \
    --dataset-column "question" \
    --num-users 5 \
    --max-time-per-run 300
```

## Advanced Examples

### Example 12: Parameter Tuning

Test different generation parameters:

```bash
# Conservative generation
genai-bench benchmark \
    --api-backend openai \
    --api-base "http://localhost:8082" \
    --api-key "your-key" \
    --api-model-name "llama-2-7b" \
    --task text-to-text \
    --temperature 0.1 \
    --max-tokens 50 \
    --top-p 0.9 \
    --num-users 5 \
    --max-time-per-run 300 \
    --output-dir "./experiments/conservative"

# Creative generation
genai-bench benchmark \
    --api-backend openai \
    --api-base "http://localhost:8082" \
    --api-key "your-key" \
    --api-model-name "llama-2-7b" \
    --task text-to-text \
    --temperature 0.9 \
    --max-tokens 100 \
    --top-p 0.95 \
    --num-users 5 \
    --max-time-per-run 300 \
    --output-dir "./experiments/creative"
```

### Example 13: Tokenizer Configuration

Use specific tokenizer for accurate token counting:

```bash
genai-bench benchmark \
    --api-backend openai \
    --api-base "http://localhost:8082" \
    --api-key "your-key" \
    --api-model-name "llama-2-7b" \
    --task text-to-text \
    --model-tokenizer "meta-llama/Llama-2-7b-chat-hf" \
    --num-users 5 \
    --max-time-per-run 300
```

### Example 14: Server Information

Include server details for analysis:

```bash
genai-bench benchmark \
    --api-backend openai \
    --api-base "http://localhost:8082" \
    --api-key "your-key" \
    --api-model-name "llama-2-7b" \
    --task text-to-text \
    --server-engine "vLLM" \
    --server-gpu-type "H100" \
    --server-model-size "7B" \
    --server-batch-size 32 \
    --num-users 10 \
    --max-time-per-run 300
```

## Analysis Examples

### Example 15: Generate Excel Report

Create detailed Excel report:

```bash
genai-bench excel \
    --experiment-dir "./experiments/my-benchmark" \
    --output-file "detailed_report.xlsx" \
    --include-raw-data \
    --include-pricing
```

### Example 16: Custom Plots

Generate custom visualizations:

```bash
# Create plot configuration
cat > plot_config.yaml << EOF
layout:
  rows: 2
  cols: 2

plots:
  - title: "Throughput vs Concurrency"
    type: "line"
    x_axis: "num_users"
    y_axis: "mean_total_tokens_throughput"
    
  - title: "Latency vs Concurrency"
    type: "line"
    x_axis: "num_users"
    y_axis: "mean_e2e_latency"
    
  - title: "Error Rate"
    type: "bar"
    x_axis: "num_users"
    y_axis: "error_rate"
    
  - title: "Token Distribution"
    type: "histogram"
    x_axis: "num_output_tokens"
    bins: 20
EOF

# Generate plots
genai-bench plot \
    --experiment-dirs "./experiments/model1,./experiments/model2" \
    --output-file "comparison.png" \
    --plot-config "plot_config.yaml" \
    --group-by "api_model_name"
```

## Troubleshooting Examples

### Example 17: Debug Mode

Run with detailed logging:

```bash
genai-bench benchmark \
    --api-backend openai \
    --api-base "http://localhost:8082" \
    --api-key "your-key" \
    --api-model-name "llama-2-7b" \
    --task text-to-text \
    --log-level DEBUG \
    --num-users 1 \
    --max-time-per-run 60
```

### Example 18: Test Connection

Test API connectivity:

```bash
# Test with minimal configuration
genai-bench benchmark \
    --api-backend openai \
    --api-base "http://localhost:8082" \
    --api-key "your-key" \
    --api-model-name "test" \
    --task text-to-text \
    --num-users 1 \
    --max-time-per-run 10 \
    --max-requests-per-run 1
```

## Best Practices

### Before Running Benchmarks

1. **Test API Connection**: Ensure your API endpoint is accessible
2. **Check Resources**: Verify sufficient CPU, memory, and network capacity
3. **Start Small**: Begin with low concurrency and short duration
4. **Monitor System**: Watch system resources during benchmarks

### During Benchmarks

1. **Use UI Dashboard**: Monitor real-time progress at `http://localhost:8089`
2. **Check Logs**: Monitor logs for errors or warnings
3. **Resource Monitoring**: Watch CPU, memory, and network usage
4. **Error Analysis**: Pay attention to error rates and types

### After Benchmarks

1. **Generate Reports**: Create Excel reports for detailed analysis
2. **Create Plots**: Visualize results for better understanding
3. **Compare Results**: Compare with previous benchmarks
4. **Document Findings**: Record configuration and results

## Next Steps

- Read the [User Guide](../user-guide/overview.md) for detailed explanations
- Learn about [Tasks and Benchmarks](../user-guide/tasks.md) for different task types
- Check out [Results Analysis](../user-guide/analysis.md) for understanding your metrics
- Explore the [CLI Reference](../user-guide/cli.md) for complete command documentation 