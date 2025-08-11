# Multi-Cloud Authentication and Storage Guide

genai-bench now supports comprehensive multi-cloud authentication for both model endpoints and storage services. This guide covers how to configure and use authentication for various cloud providers.

## Table of Contents
- [Overview](#overview)
- [Model Provider Authentication](#model-provider-authentication)
  - [OpenAI](#openai)
  - [OCI Cohere](#oci-cohere)
  - [OCI GenAI](#oci-genai)
  - [AWS Bedrock](#aws-bedrock)
  - [Azure OpenAI](#azure-openai)
  - [GCP Vertex AI](#gcp-vertex-ai)
  - [SGLang / vLLM](#sglang-or-vllm)
- [Storage Provider Authentication](#storage-provider-authentication)
  - [OCI Object Storage](#oci-object-storage)
  - [AWS S3](#aws-s3)
  - [Azure Blob Storage](#azure-blob-storage)
  - [GCP Cloud Storage](#gcp-cloud-storage)
  - [GitHub Releases](#github-releases)
- [Command Examples](#command-examples)
- [Environment Variables](#environment-variables)
- [Best Practices](#best-practices)

## Overview

genai-bench separates authentication into two categories:
1. **Model Authentication**: For accessing LLM endpoints
2. **Storage Authentication**: For uploading benchmark results

This separation allows you to benchmark models from one provider while storing results in another provider's storage service.

## Model Provider Authentication

### OpenAI

OpenAI uses API key authentication.

**Required parameters:**

- `--api-backend openai`
- `--api-key` or `--model-api-key`: Your OpenAI API key

**Example:**
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

**Environment variable alternative:**
```bash
export MODEL_API_KEY=sk-...
genai-bench benchmark --api-backend openai ...
```

### OCI Cohere

OCI supports multiple authentication methods.

**Authentication types:**

- `user_principal`: Default, uses OCI config file
- `instance_principal`: For compute instances
- `security_token`: For delegation tokens
- `instance_obo_user`: Instance principal with user delegation

**Required parameters:**

- `--api-backend oci-cohere` or `--api-backend cohere`
- `--auth`: Authentication type (default: user_principal)

**User Principal example:**
```bash
genai-bench benchmark \
  --api-backend oci-cohere \
  --api-base https://inference.generativeai.us-chicago-1.oci.oraclecloud.com \
  --auth user_principal \
  --config-file ~/.oci/config \
  --profile DEFAULT \
  --api-model-name cohere.command-r-plus \
  --model-tokenizer Cohere/command-r-plus \
  --task text-to-text \
  --max-requests-per-run 100 \
  --max-time-per-run 10
```

**Instance Principal example:**
```bash
genai-bench benchmark \
  --api-backend oci-cohere \
  --api-base https://inference.generativeai.us-chicago-1.oci.oraclecloud.com \
  --auth instance_principal \
  --region us-chicago-1 \
  --api-model-name cohere.command-r-plus \
  --model-tokenizer Cohere/command-r-plus \
  --task text-to-text \
  --max-requests-per-run 100 \
  --max-time-per-run 10
```

### OCI GenAI

OCI GenAI provides access to Pretrained Foundational Models ([Available Models](https://docs.oracle.com/en-us/iaas/Content/generative-ai/pretrained-models.htm)) through Oracle Cloud Infrastructure's Generative AI service. It uses the same authentication methods as OCI Cohere.

**Authentication types:**

- `user_principal`: Default, uses OCI config file
- `instance_principal`: For compute instances
- `security_token`: For delegation tokens
- `instance_obo_user`: Instance principal with user delegation

**Required parameters:**

- `--api-backend oci-genai`
- `--auth`: Authentication type (default: user_principal)
- `--additional-request-params`: Must include `compartmentId` and `servingType`

**Supported tasks:**

- `text-to-text`: Chat completion

**Serving modes:**

- `ON_DEMAND`: Uses model_id for on-demand inference
- `DEDICATED`: Uses endpointId for dedicated endpoints

**User Principal example (Grok):**
```bash
genai-bench benchmark \
  --api-backend oci-genai \
  --api-base https://inference.generativeai.us-ashburn-1.oci.oraclecloud.com \
  --auth user_principal \
  --config-file ~/.oci/config \
  --profile DEFAULT \
  --api-model-name xai.grok-3-mini-fast \
  --model-tokenizer gpt2 \
  --task text-to-text \
  --additional-request-params '{"compartmentId": "ocid1.compartment.oc1..example", "servingType": "ON_DEMAND"}' \
  --max-requests-per-run 100 \
  --max-time-per-run 10
```

**Security Token example (Grok):**
```bash
genai-bench benchmark \
  --api-backend oci-genai \
  --api-base https://inference.generativeai.us-ashburn-1.oci.oraclecloud.com \
  --auth security_token \
  --config-file ~/.oci/config \
  --profile DEFAULT \
  --api-model-name xai.grok-3-mini-fast \
  --model-tokenizer gpt2 \
  --task text-to-text \
  --additional-request-params '{"compartmentId": "ocid1.compartment.oc1..example", "servingType": "ON_DEMAND"}' \
  --max-requests-per-run 100 \
  --max-time-per-run 10
```

**Dedicated Endpoint example:**
```bash
genai-bench benchmark \
  --api-backend oci-genai \
  --api-base https://inference.generativeai.us-ashburn-1.oci.oraclecloud.com \
  --auth user_principal \
  --api-model-name xai.grok-3-mini-fast \
  --model-tokenizer gpt2 \
  --task text-to-text \
  --additional-request-params '{"compartmentId": "ocid1.compartment.oc1..example", "servingType": "DEDICATED", "endpointId": "ocid1.endpoint.oc1..example"}' \
  --max-requests-per-run 100 \
  --max-time-per-run 10
```
**Note:** for Dedicated model, the `--api-model-name` is just a placeholder, the model depends on the the endpointId you provided

**Advanced features:**
```bash
# With system message and chat history
genai-bench benchmark \
  --api-backend oci-genai \
  --api-base https://inference.generativeai.us-ashburn-1.oci.oraclecloud.com \
  --auth user_principal \
  --api-model-name xai.grok-3-mini-fast \
  --model-tokenizer gpt2 \
  --task text-to-text \
  --additional-request-params '{"compartmentId": "ocid1.compartment.oc1..example", "servingType": "ON_DEMAND", "system_message": "You are a helpful assistant.", "temperature": 0.7, "top_p": 0.9}' \
  --max-requests-per-run 100 \
  --max-time-per-run 10
```

**Note:** The OCI GenAI service requires access to the specific models you want to benchmark. Ensure your tenancy has the necessary service limits and permissions configured.

### AWS Bedrock

AWS Bedrock supports IAM credentials and profiles.

**Authentication methods:**
1. **IAM Credentials**: Access key ID and secret access key
2. **AWS Profile**: Named profile from credentials file
3. **Environment variables**: AWS SDK default behavior

**Required parameters:**

- `--api-backend aws-bedrock`
- `--aws-region`: AWS region for Bedrock

**IAM Credentials example:**
```bash
genai-bench benchmark \
  --api-backend aws-bedrock \
  --api-base https://bedrock-runtime.us-east-1.amazonaws.com \
  --aws-access-key-id AKIAIOSFODNN7EXAMPLE \
  --aws-secret-access-key wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY \
  --aws-region us-east-1 \
  --api-model-name anthropic.claude-3-sonnet-20240229-v1:0 \
  --model-tokenizer Anthropic/claude-3-sonnet \
  --task text-to-text \
  --max-requests-per-run 100 \
  --max-time-per-run 10
```

**AWS Profile example:**
```bash
genai-bench benchmark \
  --api-backend aws-bedrock \
  --api-base https://bedrock-runtime.us-west-2.amazonaws.com \
  --aws-profile production \
  --aws-region us-west-2 \
  --api-model-name amazon.titan-text-express-v1 \
  --model-tokenizer amazon/titan \
  --task text-to-text \
  --max-requests-per-run 100 \
  --max-time-per-run 10
```

**Environment variables:**
```bash
export AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
export AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
export AWS_DEFAULT_REGION=us-east-1

genai-bench benchmark --api-backend aws-bedrock ...
```

### Azure OpenAI

Azure OpenAI supports API key and Azure AD authentication.

**Authentication methods:**
1. **API Key**: Traditional API key authentication
2. **Azure AD**: Azure Active Directory token

**Required parameters:**

- `--api-backend azure-openai`
- `--azure-endpoint`: Your Azure OpenAI endpoint
- `--azure-deployment`: Your deployment name
- `--azure-api-version`: API version (default: 2024-02-01)

**API Key example:**
```bash
genai-bench benchmark \
  --api-backend azure-openai \
  --api-base https://myresource.openai.azure.com \
  --azure-endpoint https://myresource.openai.azure.com \
  --azure-deployment my-gpt-4-deployment \
  --model-api-key YOUR_AZURE_API_KEY \
  --api-model-name gpt-4 \
  --model-tokenizer gpt2 \
  --task text-to-text \
  --max-requests-per-run 100 \
  --max-time-per-run 10
```

**Azure AD example:**
```bash
genai-bench benchmark \
  --api-backend azure-openai \
  --api-base https://myresource.openai.azure.com \
  --azure-endpoint https://myresource.openai.azure.com \
  --azure-deployment my-gpt-4-deployment \
  --azure-ad-token YOUR_AAD_TOKEN \
  --api-model-name gpt-4 \
  --model-tokenizer gpt2 \
  --task text-to-text \
  --max-requests-per-run 100 \
  --max-time-per-run 10
```

### GCP Vertex AI

GCP Vertex AI supports service account and API key authentication.

**Authentication methods:**
1. **Service Account**: JSON key file
2. **API Key**: For certain Vertex AI services
3. **Application Default Credentials**: GCP SDK default

**Required parameters:**

- `--api-backend gcp-vertex`
- `--gcp-project-id`: Your GCP project ID
- `--gcp-location`: GCP region (default: us-central1)

**Service Account example:**
```bash
genai-bench benchmark \
  --api-backend gcp-vertex \
  --api-base https://us-central1-aiplatform.googleapis.com \
  --gcp-project-id my-project-123 \
  --gcp-location us-central1 \
  --gcp-credentials-path /path/to/service-account.json \
  --api-model-name gemini-1.5-pro \
  --model-tokenizer google/gemini \
  --task text-to-text \
  --max-requests-per-run 100 \
  --max-time-per-run 10
```

**Environment variable:**
```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
export GCP_PROJECT_ID=my-project-123

genai-bench benchmark --api-backend gcp-vertex ...
```

### SGLang or vLLM

vLLM and SGLang use OpenAI-compatible APIs with optional authentication.

**Required parameters:**

- `--api-backend sglang` or `--api-backend vllm`
- `--api-base`: Your server endpoint
- `--api-key` or `--model-api-key`: Optional API key if authentication is enabled

**Example:**
```bash
genai-bench benchmark \
  --api-backend sglang \
  --api-base http://localhost:8000 \
  --api-key optional-key \
  --api-model-name meta-llama/Llama-2-7b-hf \
  --model-tokenizer meta-llama/Llama-2-7b-hf \
  --task text-to-text \
  --max-requests-per-run 100 \
  --max-time-per-run 10
```

## Storage Provider Authentication

Storage authentication is configured separately from model authentication, allowing you to store results in any supported storage service.

### Common Storage Parameters

All storage providers share these common parameters:

- `--upload-results`: Flag to enable result upload
- `--storage-provider`: Storage provider type (oci, aws, azure, gcp, github)
- `--storage-bucket`: Bucket/container name
- `--storage-prefix`: Optional prefix for uploaded objects

### OCI Object Storage

**Authentication types:**
Same as OCI model authentication (user_principal, instance_principal, etc.)

**Example:**
```bash
genai-bench benchmark \
  ... \
  --upload-results \
  --storage-provider oci \
  --storage-bucket my-benchmark-results \
  --storage-prefix experiments/2024 \
  --storage-auth-type user_principal \
  --namespace my-namespace
```

### AWS S3

**Authentication methods:**
1. **IAM Credentials**
2. **AWS Profile**
3. **Environment variables**

**IAM Credentials example:**
```bash
genai-bench benchmark \
  ... \
  --upload-results \
  --storage-provider aws \
  --storage-bucket my-benchmark-results \
  --storage-prefix experiments/2024 \
  --storage-aws-access-key-id AKIAIOSFODNN7EXAMPLE \
  --storage-aws-secret-access-key wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY \
  --storage-aws-region us-east-1
```

**AWS Profile example:**
```bash
genai-bench benchmark \
  ... \
  --upload-results \
  --storage-provider aws \
  --storage-bucket my-benchmark-results \
  --storage-prefix experiments/2024 \
  --storage-aws-profile production \
  --storage-aws-region us-west-2
```

### Azure Blob Storage

**Authentication methods:**
1. **Storage Account Key**
2. **Connection String**
3. **SAS Token**
4. **Azure AD**

**Account Key example:**
```bash
genai-bench benchmark \
  ... \
  --upload-results \
  --storage-provider azure \
  --storage-bucket my-container \
  --storage-prefix experiments/2024 \
  --storage-azure-account-name mystorageaccount \
  --storage-azure-account-key YOUR_ACCOUNT_KEY
```

**Connection String example:**
```bash
genai-bench benchmark \
  ... \
  --upload-results \
  --storage-provider azure \
  --storage-bucket my-container \
  --storage-azure-connection-string "DefaultEndpointsProtocol=https;AccountName=..."
```

### GCP Cloud Storage

**Authentication methods:**
1. **Service Account**
2. **Application Default Credentials**

**Example:**
```bash
genai-bench benchmark \
  ... \
  --upload-results \
  --storage-provider gcp \
  --storage-bucket my-benchmark-results \
  --storage-prefix experiments/2024 \
  --storage-gcp-project-id my-project-123 \
  --storage-gcp-credentials-path /path/to/service-account.json
```

### GitHub Releases

GitHub storage uploads results as release artifacts.

**Required parameters:**

- `--github-token`: Personal access token with repo permissions
- `--github-owner`: Repository owner (user or organization)
- `--github-repo`: Repository name

**Example:**
```bash
genai-bench benchmark \
  ... \
  --upload-results \
  --storage-provider github \
  --github-token ghp_xxxxxxxxxxxxxxxxxxxx \
  --github-owner myorg \
  --github-repo benchmark-results
```

## Command Examples

### Cross-Cloud Benchmarking

**Benchmark OpenAI and store in AWS S3:**
```bash
genai-bench benchmark \
  --api-backend openai \
  --api-base https://api.openai.com/v1 \
  --api-key sk-... \
  --api-model-name gpt-4 \
  --model-tokenizer gpt2 \
  --task text-to-text \
  --max-requests-per-run 100 \
  --max-time-per-run 10 \
  --upload-results \
  --storage-provider aws \
  --storage-bucket my-benchmarks \
  --storage-aws-profile default \
  --storage-aws-region us-east-1
```

**Benchmark AWS Bedrock and store in Azure Blob:**
```bash
genai-bench benchmark \
  --api-backend aws-bedrock \
  --api-base https://bedrock-runtime.us-east-1.amazonaws.com \
  --aws-profile bedrock-user \
  --aws-region us-east-1 \
  --api-model-name anthropic.claude-3-sonnet-20240229-v1:0 \
  --model-tokenizer Anthropic/claude-3-sonnet \
  --task text-to-text \
  --max-requests-per-run 100 \
  --max-time-per-run 10 \
  --upload-results \
  --storage-provider azure \
  --storage-bucket benchmarks \
  --storage-azure-connection-string "DefaultEndpointsProtocol=..."
```

**Benchmark OCI GenAI (Grok models) and store in OCI Object storage**
```bash
genai-bench benchmark \
  --api-backend oci-genai \
  --api-base https://inference.generativeai.us-ashburn-1.oci.oraclecloud.com \
  --auth security_token \
  --config-file ~/.oci/config \
  --profile DEFAULT \
  --api-model-name xai.grok-3-mini-fast \
  --model-tokenizer gpt2 \
  --task text-to-text \
  --additional-request-params '{"compartmentId": "ocid1.compartment.oc1..example", "servingType": "ON_DEMAND"}' \
  --max-requests-per-run 100 \
  --max-time-per-run 10 \
  --upload-results \
  --storage-provider oci \
  --storage-bucket oci-genai-benchmarks \
  --namespace my-namespace
```

### Multi-Modal Tasks

**Image-text-to-text with GCP Vertex AI:**
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
  --dataset-path /path/to/image/dataset \
  --max-requests-per-run 50 \
  --max-time-per-run 10
```

## Environment Variables

genai-bench supports environment variables for sensitive credentials:

### Model Authentication

- `MODEL_API_KEY`: API key for OpenAI, Azure OpenAI, or GCP
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`: AWS credentials
- `AWS_PROFILE`, `AWS_DEFAULT_REGION`: AWS configuration
- `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT`, `AZURE_OPENAI_API_VERSION`: Azure configuration
- `AZURE_AD_TOKEN`: Azure AD authentication token
- `GCP_PROJECT_ID`, `GCP_LOCATION`: GCP configuration
- `GOOGLE_APPLICATION_CREDENTIALS`: Path to GCP service account JSON

### Storage Authentication
- `AZURE_STORAGE_ACCOUNT_NAME`, `AZURE_STORAGE_ACCOUNT_KEY`: Azure storage credentials
- `AZURE_STORAGE_CONNECTION_STRING`, `AZURE_STORAGE_SAS_TOKEN`: Azure storage alternatives
- `GITHUB_TOKEN`, `GITHUB_OWNER`, `GITHUB_REPO`: GitHub configuration

### General
- `HF_TOKEN`: HuggingFace token for downloading tokenizers

## Best Practices

### Security
1. **Never commit credentials**: Use environment variables or secure credential stores
2. **Use least privilege**: Grant only necessary permissions for benchmarking
3. **Rotate credentials regularly**: Update API keys and tokens periodically
4. **Use service accounts**: Prefer service accounts over personal credentials for automation

### Performance
1. **Choose nearby regions**: Select cloud regions close to your location for lower latency
2. **Batch operations**: Use appropriate batch sizes for embedding tasks
3. **Monitor costs**: Be aware of API pricing and set appropriate limits

### Organization
1. **Use consistent naming**: Adopt a naming convention for storage prefixes
2. **Separate environments**: Use different buckets/prefixes for dev/test/prod
3. **Tag resources**: Use cloud provider tags for cost tracking and organization

### Important Notes

1. **Task-specific behavior**:
    - For `text-to-embeddings` and `text-to-rerank` tasks, the iteration type automatically switches to `batch_size`
    - For other tasks, `num_concurrency` iteration is used
    - This is handled automatically by the CLI

2. **OCI GenAI requirements**:
    - Only supports `text-to-text` task (chat completion) for Grok models
    - Requires `compartmentId` in `additional-request-params`
    - Supports both `ON_DEMAND` and `DEDICATED` serving types
    - For dedicated endpoints, `endpointId` is required

3. **Image format requirements**:
    - Image inputs are expected to be in JPEG format for multi-modal tasks
    - Base64 encoding is handled automatically

4. **Token counting**:
    - Different providers may use different tokenization methods
    - Token estimates for embeddings tasks may vary by provider

### Troubleshooting
1. **Check credentials**: Verify authentication credentials are correct
2. **Verify permissions**: Ensure accounts have necessary permissions
3. **Check regions**: Confirm services are available in selected regions
4. **Review quotas**: Check API quotas and rate limits
5. **Enable logging**: Use verbose logging for debugging authentication issues

## Migration from Legacy CLI

If you're migrating from the legacy OCI-only CLI:

**Old command:**
```bash
genai-bench benchmark \
  --api-backend oci-cohere \
  --bucket my-bucket \
  --prefix my-prefix \
  ...
```

**New command:**
```bash
genai-bench benchmark \
  --api-backend oci-cohere \
  --storage-bucket my-bucket \
  --storage-prefix my-prefix \
  --storage-provider oci \
  ...
```

The main changes are:

- `--bucket` → `--storage-bucket`
- `--prefix` → `--storage-prefix`
- Add `--storage-provider oci` (though OCI is the default for backward compatibility)
