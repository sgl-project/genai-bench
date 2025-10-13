# Together AI Integration

This document describes the Together AI backend integration for genai-bench.

## Overview

The Together AI backend has been fully integrated into genai-bench, allowing you to benchmark models hosted on Together AI's platform.

## Features

- **Chat Completions**: Support for text-to-text and image-text-to-text tasks
- **Embeddings**: Support for text-to-embeddings tasks
- **Streaming**: Full support for streaming responses
- **Authentication**: API key-based authentication

## Usage

### Basic Usage

```bash
genai-bench benchmark \
    --api-backend together \
    --api-base https://api.together.xyz \
    --api-key YOUR_TOGETHER_API_KEY \
    --api-model-name meta-llama/Llama-2-7b-chat-hf \
    --task text-to-text \
    --num-concurrency 1,2,4,8 \
    --batch-size 1,2,4 \
    --dataset-path /path/to/your/dataset.json
```

### Environment Variables

You can also set the API key via environment variable:

```bash
export TOGETHER_API_KEY=your_api_key_here
genai-bench benchmark \
    --api-backend together \
    --api-base https://api.together.xyz \
    --api-model-name meta-llama/Llama-2-7b-chat-hf \
    --task text-to-text \
    # ... other options
```

### Supported Models

Together AI supports a wide range of models. Some popular options include:

- `meta-llama/Llama-2-7b-chat-hf`
- `meta-llama/Llama-2-13b-chat-hf`
- `meta-llama/Llama-2-70b-chat-hf`
- `mistralai/Mistral-7B-Instruct-v0.1`
- `togethercomputer/RedPajama-INCITE-Chat-3B-v1`
- And many more...

### Supported Tasks

- `text-to-text`: Standard chat completions
- `image-text-to-text`: Multimodal chat with images
- `text-to-embeddings`: Text embedding generation

## Implementation Details

### Files Added/Modified

1. **User Implementation**: `genai_bench/user/together_user.py`
   - Implements `TogetherUser` class extending `BaseUser`
   - Supports chat completions and embeddings
   - Handles streaming responses

2. **Authentication**: `genai_bench/auth/together/`
   - `auth.py`: Basic Together AI authentication
   - `model_auth_adapter.py`: Adapter for model authentication

3. **CLI Integration**:
   - Added "together" to supported backends in `option_groups.py`
   - Added together backend handling in `cli.py`
   - Added TogetherUser to validation mapping

### API Compatibility

The Together AI backend uses OpenAI-compatible API endpoints:
- Chat completions: `/v1/chat/completions`
- Embeddings: `/v1/embeddings`

This ensures compatibility with existing benchmarking scenarios and metrics collection.

## Example Commands

### Text-to-Text Benchmarking

```bash
genai-bench benchmark \
    --api-backend together \
    --api-base https://api.together.xyz \
    --api-key $TOGETHER_API_KEY \
    --api-model-name meta-llama/Llama-2-7b-chat-hf \
    --task text-to-text \
    --num-concurrency 1,2,4,8,16 \
    --batch-size 1,2,4,8 \
    --dataset-path examples/dataset_configs/huggingface_simple.json
```

### Embeddings Benchmarking

```bash
genai-bench benchmark \
    --api-backend together \
    --api-base https://api.together.xyz \
    --api-key $TOGETHER_API_KEY \
    --api-model-name togethercomputer/RedPajama-INCITE-Chat-3B-v1 \
    --task text-to-embeddings \
    --num-concurrency 1,2,4,8 \
    --batch-size 1,2,4,8 \
    --dataset-path examples/dataset_configs/huggingface_simple.json
```

### Multimodal Benchmarking

```bash
genai-bench benchmark \
    --api-backend together \
    --api-base https://api.together.xyz \
    --api-key $TOGETHER_API_KEY \
    --api-model-name meta-llama/Llama-2-7b-chat-hf \
    --task image-text-to-text \
    --num-concurrency 1,2,4 \
    --batch-size 1,2 \
    --dataset-path examples/dataset_configs/config_llava-bench-in-the-wild.json
```

## Notes

- The Together AI backend requires a valid API key from [Together AI](https://together.ai)
- All standard genai-bench features are supported (metrics collection, reporting, etc.)
- The implementation follows the same patterns as other backends for consistency
- Streaming responses are fully supported for accurate latency measurements
