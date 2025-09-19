# Multi-Cloud Quick Reference

This is a quick reference guide for common multi-cloud scenarios with genai-bench. For detailed information, see the [comprehensive guide](multi-cloud-auth-storage.md).

> **Note**: For OpenAI, SGLang and vLLM backends, both `--api-key` and `--model-api-key` are supported for backward compatibility.

## OpenAI Benchmarking

### Basic Usage

```bash
genai-bench benchmark \
  --api-backend openai \
  --api-base https://api.openai.com/v1 \
  --api-key sk-... \
  --api-model-name gpt-4 \
  --model-tokenizer gpt2 \
  --task text-to-text \
  --max-requests-per-run 100 \
  --max-time-per-run 10
```

### With Environment Variable

```bash
export MODEL_API_KEY=sk-...
genai-bench benchmark \
  --api-backend openai \
  --api-base https://api.openai.com/v1 \
  --api-model-name gpt-4 \
  --model-tokenizer gpt2 \
  --task text-to-text \
  --max-requests-per-run 100 \
  --max-time-per-run 10
```

## AWS Bedrock Benchmarking

### Using AWS Profile

```bash
genai-bench benchmark \
  --api-backend aws-bedrock \
  --api-base https://bedrock-runtime.us-east-1.amazonaws.com \
  --aws-profile default \
  --aws-region us-east-1 \
  --api-model-name anthropic.claude-3-sonnet-20240229-v1:0 \
  --model-tokenizer Anthropic/claude-3-sonnet \
  --task text-to-text \
  --max-requests-per-run 100 \
  --max-time-per-run 10
```

### Using IAM Credentials

```bash
genai-bench benchmark \
  --api-backend aws-bedrock \
  --api-base https://bedrock-runtime.us-east-1.amazonaws.com \
  --aws-access-key-id AKIAIOSFODNN7EXAMPLE \
  --aws-secret-access-key wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY \
  --aws-region us-east-1 \
  --api-model-name amazon.titan-text-express-v1 \
  --model-tokenizer amazon/titan \
  --task text-to-text \
  --max-requests-per-run 100 \
  --max-time-per-run 10
```

## Azure OpenAI Benchmarking

### Using API Key

```bash
genai-bench benchmark \
  --api-backend azure-openai \
  --api-base https://myresource.openai.azure.com \
  --azure-endpoint https://myresource.openai.azure.com \
  --azure-deployment my-gpt-4-deployment \
  --model-api-key YOUR_API_KEY \
  --api-model-name gpt-4 \
  --model-tokenizer gpt2 \
  --task text-to-text \
  --max-requests-per-run 100 \
  --max-time-per-run 10
```

## GCP Vertex AI Benchmarking

### Using Service Account

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
genai-bench benchmark \
  --api-backend gcp-vertex \
  --api-base https://us-central1-aiplatform.googleapis.com \
  --gcp-project-id my-project-123 \
  --gcp-location us-central1 \
  --api-model-name gemini-1.5-pro \
  --model-tokenizer google/gemini \
  --task text-to-text \
  --max-requests-per-run 100 \
  --max-time-per-run 10
```

## Storage Examples

### Upload to OCI Object Storage

```bash
genai-bench benchmark \
  ... \
  --upload-results \
  --storage-provider oci \
  --storage-bucket my-benchmarks \
  --storage-prefix experiments/2024 \
  --namespace my-namespace
```

### Upload to AWS S3

```bash
genai-bench benchmark \
  ... \
  --upload-results \
  --storage-provider aws \
  --storage-bucket my-benchmarks \
  --storage-prefix experiments/2024 \
  --storage-aws-profile default
```

### Upload to Azure Blob

```bash
genai-bench benchmark \
  ... \
  --upload-results \
  --storage-provider azure \
  --storage-bucket my-container \
  --storage-azure-account-name myaccount \
  --storage-azure-account-key YOUR_ACCOUNT_KEY
```

### Upload to GCP Cloud Storage

```bash
genai-bench benchmark \
  ... \
  --upload-results \
  --storage-provider gcp \
  --storage-bucket my-benchmarks \
  --storage-gcp-project-id my-project \
  --storage-gcp-credentials-path /path/to/service-account.json
```

## Cross-Cloud Examples

### Benchmark OpenAI, Store in S3

```bash
export MODEL_API_KEY=sk-...
export AWS_PROFILE=default

genai-bench benchmark \
  --api-backend openai \
  --api-base https://api.openai.com/v1 \
  --api-model-name gpt-4 \
  --model-tokenizer gpt2 \
  --task text-to-text \
  --max-requests-per-run 100 \
  --max-time-per-run 10 \
  --upload-results \
  --storage-provider aws \
  --storage-bucket openai-benchmarks \
  --storage-aws-region us-east-1
```

### Benchmark Bedrock, Store in Azure

```bash
export AWS_PROFILE=bedrock-user
export AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=..."

genai-bench benchmark \
  --api-backend aws-bedrock \
  --api-base https://bedrock-runtime.us-east-1.amazonaws.com \
  --aws-region us-east-1 \
  --api-model-name anthropic.claude-3-sonnet-20240229-v1:0 \
  --model-tokenizer Anthropic/claude-3-sonnet \
  --task text-to-text \
  --max-requests-per-run 100 \
  --max-time-per-run 10 \
  --upload-results \
  --storage-provider azure \
  --storage-bucket bedrock-benchmarks
```

### Benchmark Azure OpenAI, Store in GCP

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/storage-sa.json

genai-bench benchmark \
  --api-backend azure-openai \
  --api-base https://myresource.openai.azure.com \
  --azure-endpoint https://myresource.openai.azure.com \
  --azure-deployment my-deployment \
  --model-api-key YOUR_API_KEY
  --api-model-name gpt-4 \
  --model-tokenizer gpt2 \
  --task text-to-text \
  --max-requests-per-run 100 \
  --max-time-per-run 10 \
  --upload-results \
  --storage-provider gcp \
  --storage-bucket azure-benchmarks \
  --storage-gcp-project-id my-project
```

## Multi-Modal Tasks

### Image-text-to-Text Benchmarking

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
  --max-requests-per-run 50 \
  --max-time-per-run 10
```

### Text-to-Embeddings Benchmarking

```bash
genai-bench benchmark \
  --api-backend openai \
  --api-base https://api.openai.com/v1 \
  --api-key sk-... \
  --api-model-name text-embedding-3-large \
  --model-tokenizer cl100k_base \
  --task text-to-embeddings \
  --batch-size 1 --batch-size 8 --batch-size 32 \
  --max-requests-per-run 1000 \
  --max-time-per-run 10
```

## Environment Variable Reference

### Model Authentication

```bash
# OpenAI
export MODEL_API_KEY=sk-...

# AWS
export AWS_ACCESS_KEY_ID=AKIA...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=us-east-1
export AWS_PROFILE=default

# Azure
export AZURE_OPENAI_ENDPOINT=https://myresource.openai.azure.com
export AZURE_OPENAI_API_VERSION=2024-02-01
export AZURE_AD_TOKEN=...

# GCP
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
export GCP_PROJECT_ID=my-project
export GCP_LOCATION=us-central1
```

### Storage Authentication

```bash
# Azure Storage
export AZURE_STORAGE_ACCOUNT_NAME=myaccount
export AZURE_STORAGE_ACCOUNT_KEY=...
export AZURE_STORAGE_CONNECTION_STRING=...

# GitHub
export GITHUB_TOKEN=ghp_...
export GITHUB_OWNER=myorg
export GITHUB_REPO=benchmarks
```

### General

```bash
# HuggingFace (for downloading tokenizers)
export HF_TOKEN=hf_...
```
