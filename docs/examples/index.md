# Examples

This section provides practical examples and configurations for GenAI Bench.

## Quick Examples

### OpenAI GPT-4 Benchmark

```bash
genai-bench benchmark \
  --api-backend openai \
  --api-base https://api.openai.com/v1 \
  --api-key $OPENAI_API_KEY \
  --api-model-name gpt-4 \
  --model-tokenizer gpt2 \
  --task text-to-text \
  --max-requests-per-run 1000 \
  --max-time-per-run 10
```

### AWS Bedrock Claude Benchmark

```bash
genai-bench benchmark \
  --api-backend aws-bedrock \
  --api-base https://bedrock-runtime.us-east-1.amazonaws.com \
  --aws-profile default \
  --aws-region us-east-1 \
  --api-model-name anthropic.claude-3-sonnet-20240229-v1:0 \
  --model-tokenizer Anthropic/claude-3-sonnet \
  --task text-to-text \
  --max-requests-per-run 500 \
  --max-time-per-run 10
```

### Multi-Modal Benchmark

```bash
genai-bench benchmark \
  --api-backend gcp-vertex \
  --api-base https://us-central1-aiplatform.googleapis.com \
  --gcp-project-id my-project \
  --gcp-location us-central1 \
  --gcp-credentials-path /path/to/service-account.json \
  --api-model-name gemini-1.5-pro-vision \
  --model-tokenizer google/gemini \
  --task image-text-to-text \
  --dataset-path /path/to/images \
  --max-requests-per-run 100 \
  --max-time-per-run 10
```

### Embedding Benchmark with Batch Sizes

```bash
genai-bench benchmark \
  --api-backend openai \
  --api-base https://api.openai.com/v1 \
  --api-key $OPENAI_API_KEY \
  --api-model-name text-embedding-3-large \
  --model-tokenizer cl100k_base \
  --task text-to-embeddings \
  --batch-size 1 --batch-size 8 --batch-size 32 --batch-size 64 \
  --max-requests-per-run 2000 \
  --max-time-per-run 10
```

## Traffic Scenarios

GenAI Bench supports various traffic patterns:

### Text Generation Scenarios
- `D(100,100)` - Deterministic: 100 input tokens, 100 output tokens
- `N(480,240)/(300,150)` - Normal distribution
- `U(50,100)/(200,250)` - Uniform distribution

### Embedding Scenarios
- `E(64)` - 64 tokens per document
- `E(512)` - 512 tokens per document
- `E(1024)` - 1024 tokens per document

### Vision Scenarios
- `I(512,512)` - 512x512 pixel images
- `I(1024,512)` - 1024x512 pixel images
- `I(2048,2048)` - 2048x2048 pixel images

## Contributing Examples

Have a useful configuration or example? We welcome contributions! Please submit a pull request with your example following our [contribution guidelines](../development/contributing.md).