# Configuration Guide

This guide explains how to configure GenAI Bench for different environments, API backends, and use cases.

## Environment Configuration

### Python Environment

Ensure you have the correct Python version and dependencies:

```bash
# Check Python version
python3 --version  # Should be 3.11 or 3.12

# Install in virtual environment (recommended)
python3 -m venv genai-bench-env
source genai-bench-env/bin/activate
pip install genai-bench
```

### Environment Variables

Set these environment variables for optimal performance:

```bash
# Hugging Face token (for tokenizer downloads)
export HF_TOKEN="your-huggingface-token"

# Disable torch warnings (not needed for benchmarking)
export TRANSFORMERS_VERBOSITY=error

# Log level
export GENAI_BENCH_LOG_LEVEL=INFO

# API keys (set as needed)
export OPENAI_API_KEY="your-openai-key"
export COHERE_API_KEY="your-cohere-key"
```

## API Backend Configuration

### OpenAI-Compatible Backends

Configure for OpenAI API, vLLM, or other OpenAI-compatible services:

```bash
# Basic configuration
--api-backend openai
--api-base "https://api.openai.com/v1"
--api-key "sk-..."

# vLLM configuration
--api-backend openai
--api-base "http://localhost:8082"
--api-key "your-vllm-key"
--api-model-name "llama-2-7b"
```

#### Configuration Options

| Option | Description | Example |
|--------|-------------|---------|
| `--api-base` | API endpoint URL | `https://api.openai.com/v1` |
| `--api-key` | Authentication key | `sk-...` |
| `--api-model-name` | Model identifier | `gpt-3.5-turbo` |
| `--api-version` | API version | `2024-02-15` |

### Cohere Backend

Configure for Cohere API:

```bash
--api-backend cohere
--api-base "https://api.cohere.ai"
--api-key "your-cohere-key"
--api-model-name "command-r-plus"
```

#### Cohere-Specific Options

| Option | Description | Example |
|--------|-------------|---------|
| `--cohere-embedding-model` | Embedding model | `embed-english-v3.0` |
| `--cohere-rerank-model` | Rerank model | `rerank-english-v2.0` |

### OCI Cohere Backend

Configure for Oracle Cloud Infrastructure Cohere:

```bash
--api-backend oci_cohere
--api-base "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
--api-key "your-oci-key"
--api-model-name "cohere.command-r-plus"
```

#### OCI Configuration

1. **Set up OCI CLI**:
   ```bash
   oci setup config
   ```

2. **Configure authentication**:
   ```bash
   export OCI_CONFIG_FILE="~/.oci/config"
   export OCI_KEY_FILE="~/.oci/oci_api_key.pem"
   ```

## Model Configuration

### Model Selection

Choose the appropriate model for your task:

#### Text-to-Text Models

```bash
# OpenAI models
--api-model-name "gpt-3.5-turbo"
--api-model-name "gpt-4"
--api-model-name "gpt-4-turbo"

# vLLM models
--api-model-name "llama-2-7b"
--api-model-name "mistral-7b"
--api-model-name "codellama-7b"

# Cohere models
--api-model-name "command-r-plus"
--api-model-name "command-light"
```

#### Embedding Models

```bash
# OpenAI embeddings
--api-model-name "text-embedding-ada-002"
--api-model-name "text-embedding-3-small"

# Cohere embeddings
--api-model-name "embed-english-v3.0"
--api-model-name "embed-multilingual-v3.0"
```

#### Vision Models

```bash
# OpenAI vision models
--api-model-name "gpt-4-vision-preview"

# Other vision models
--api-model-name "llava-1.5-7b"
--api-model-name "qwen-vl-7b"
```

### Tokenizer Configuration

For accurate token counting, specify the tokenizer:

```bash
# Use model's tokenizer
--model-tokenizer "/path/to/tokenizer"

# Use Hugging Face tokenizer
--model-tokenizer "meta-llama/Llama-2-7b-chat-hf"

# Use specific tokenizer
--model-tokenizer "gpt2"
--model-tokenizer "cl100k_base"
```

## Dataset Configuration

### Built-in Datasets

Use GenAI Bench's built-in datasets:

```bash
# Text datasets
--dataset-name "sonnet.txt"
--dataset-name "qa_dataset.txt"

# Vision datasets
--dataset-name "vision_dataset"
--dataset-name "image_qa_dataset"
```

### Custom Datasets

#### Text Datasets

Create a text file with one prompt per line:

```bash
# prompts.txt
What is the capital of France?
Explain quantum computing in simple terms.
Write a short poem about spring.
```

```bash
--dataset-path "/path/to/prompts.txt"
```

#### Vision Datasets

Organize your vision dataset:

```
vision_dataset/
├── images/
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
├── prompts.txt
└── metadata.json
```

```bash
--dataset-path "/path/to/vision_dataset"
```

#### Hugging Face Datasets

Use datasets from Hugging Face:

```bash
--dataset-name "squad"
--dataset-split "train"
--dataset-column "question"
```

## Traffic Configuration

### Traffic Scenarios

Choose the appropriate traffic pattern:

#### Constant Load

```bash
--traffic-scenario constant
--num-users 10
--spawn-rate 2
```

#### Burst Traffic

```bash
--traffic-scenario burst
--num-users 50
--spawn-rate 10
--burst-duration 30
```

#### Ramp-up Traffic

```bash
--traffic-scenario ramp
--num-users 20
--spawn-rate 1
--ramp-duration 300
```

### Load Parameters

Configure load testing parameters:

```bash
# Concurrency
--num-users 10          # Number of concurrent users
--spawn-rate 2          # Users spawned per second

# Duration
--max-time-per-run 300  # Maximum run time (seconds)
--max-requests-per-run 1000  # Maximum requests per run

# Limits
--max-requests-per-user 100  # Requests per user
--max-time-per-request 60    # Timeout per request
```

## Advanced Configuration

### Request Parameters

Customize request behavior:

#### Text-to-Text Parameters

```bash
# Generation parameters
--temperature 0.7
--max-tokens 100
--top-p 0.9
--frequency-penalty 0.1
--presence-penalty 0.1

# Stop sequences
--stop-sequences "END,STOP"
```

#### Embedding Parameters

```bash
# Embedding options
--embedding-dimensions 1536
--normalize-embeddings true
--truncate-embeddings "NONE"
```

### Server Information

Provide server details for analysis:

```bash
# Server configuration
--server-engine "vLLM"
--server-gpu-type "H100"
--server-model-size "7B"
--server-batch-size 32
```

### Monitoring Configuration

Configure monitoring and logging:

```bash
# UI dashboard
--ui
--ui-host "0.0.0.0"
--ui-port 8089

# Logging
--log-level INFO
--log-file "benchmark.log"

# Output
--output-dir "./experiments"
--save-raw-data true
```

## Configuration Files

### YAML Configuration

Create a configuration file for complex setups:

```yaml
# config.yaml
api:
  backend: openai
  base: "http://localhost:8082"
  key: "your-api-key"
  model: "llama-2-7b"

task:
  type: text-to-text
  temperature: 0.7
  max_tokens: 100

traffic:
  scenario: constant
  num_users: 10
  spawn_rate: 2
  max_time: 300

dataset:
  name: "sonnet.txt"
  tokenizer: "meta-llama/Llama-2-7b-chat-hf"

monitoring:
  ui: true
  log_level: INFO
  output_dir: "./experiments"
```

Use the configuration file:

```bash
genai-bench benchmark --config config.yaml
```

### Environment-Specific Configs

Create different configurations for different environments:

#### Development Configuration

```yaml
# dev-config.yaml
api:
  backend: openai
  base: "http://localhost:8082"
  model: "llama-2-7b"

traffic:
  num_users: 2
  max_time: 60

monitoring:
  log_level: DEBUG
```

#### Production Configuration

```yaml
# prod-config.yaml
api:
  backend: openai
  base: "https://api.openai.com/v1"
  model: "gpt-4"

traffic:
  num_users: 50
  max_time: 1800

monitoring:
  ui: false
  log_level: INFO
```

## Security Configuration

### API Key Management

Secure your API keys:

```bash
# Use environment variables (recommended)
export OPENAI_API_KEY="your-key"
export COHERE_API_KEY="your-key"

# Use key files
--api-key-file "/path/to/api-key.txt"

# Use key management services
--api-key-vault "azure-keyvault://your-vault"
```

### Network Security

Configure network settings:

```bash
# Proxy configuration
--proxy "http://proxy.company.com:8080"
--proxy-auth "user:pass"

# SSL verification
--verify-ssl true
--ca-cert "/path/to/ca-cert.pem"

# Timeout settings
--connect-timeout 30
--read-timeout 60
```

## Performance Tuning

### Resource Configuration

Optimize for your hardware:

```bash
# Memory settings
--max-memory "8GB"
--memory-fraction 0.8

# CPU settings
--num-workers 4
--worker-class "gevent"

# Network settings
--connection-pool-size 100
--max-connections 1000
```

### Benchmark Optimization

```bash
# Sampling optimization
--sample-size 1000
--sample-strategy "random"

# Metrics collection
--metrics-interval 1
--detailed-metrics true

# Result processing
--compress-results true
--save-intermediate true
```

## Troubleshooting Configuration

### Common Issues

#### API Connection Issues

```bash
# Test API connection
curl -X POST "http://localhost:8082/v1/chat/completions" \
  -H "Authorization: Bearer your-key" \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[{"role":"user","content":"test"}]}'
```

#### Tokenizer Issues

```bash
# Test tokenizer
python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
print(tokenizer.encode('test'))
"
```

#### Dataset Issues

```bash
# Validate dataset
genai-bench validate-dataset --dataset-path "/path/to/dataset"
```

### Configuration Validation

```bash
# Validate configuration
genai-bench validate-config --config config.yaml

# Test configuration
genai-bench test-config --config config.yaml
```

## Best Practices

### Configuration Management

1. **Use Version Control**: Track configuration changes
2. **Environment Separation**: Use different configs for dev/prod
3. **Sensitive Data**: Never commit API keys to version control
4. **Documentation**: Document configuration decisions

### Performance Optimization

1. **Start Simple**: Begin with basic configurations
2. **Gradual Scaling**: Increase complexity step by step
3. **Monitor Resources**: Watch CPU, memory, and network usage
4. **Test Thoroughly**: Validate configurations before production use

### Security Considerations

1. **API Key Rotation**: Regularly rotate API keys
2. **Network Security**: Use secure connections and firewalls
3. **Access Control**: Limit access to configuration files
4. **Audit Logging**: Log configuration changes

## Next Steps

- Read the [Quick Start Guide](quick-start.md) to run your first benchmark
- Explore [Tasks and Benchmarks](user-guide/tasks.md) for specific scenarios
- Check out [Examples](examples/basic-benchmarks.md) for practical configurations 