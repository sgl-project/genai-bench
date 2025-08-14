# Baseten Support

GenAI Bench provides support for Baseten model endpoints, including multiple request and response formats.

## Overview

The Baseten backend can handle both OpenAI-compatible chat format and a simple prompt format. This allows benchmarking of both instruct-tuned and base models. 

## Key Features

### Dual Request Format Support

**OpenAI-Compatible Format (Default)**
- Uses `{"messages": [{"role": "user", "content": "..."}]}` structure
- Compatible with instruct-tuned models
- Supports image content for vision models

**Simple Prompt Format**
- Uses `{"prompt": "..."}` structure
- Suitable for non-instruct models
- Enabled via `{"use_prompt_format": true}` in `additional_request_params`

### Streaming Control

- Supports both streaming and non-streaming responses
- Uses global `--disable-streaming` flag (consistent with other backends)
- Automatically filters out `stream` parameter from `additional_request_params`

### Response Format Flexibility

- Handles OpenAI-compatible JSON responses
- Non-OpenAI format: Automatically detects and parses various JSON field names (`text`, `output`, `response`, `generated_text`) or plain text responses

## Usage Examples

### Basic OpenAI-Compatible Format

```bash
genai-bench benchmark \
  --api-backend baseten \
  --api-base "your-endpoint-url" \
  --api-key "your-baseten-api-key" \
  --api-model-name "Qwen3-30B-A3B-Instruct-2507-FP8" \
  --model-tokenizer "Qwen/Qwen3-30B-A3B-Instruct-2507" \
  --task text-to-text \
  --max-requests-per-run 200 \
  --num-concurrency 8 \
  --max-time-per-run 600 \
  --additional-request-params '{"temperature": 0.7}'
```

### Simple Prompt Format for Non-Instruct Models

```bash
genai-bench benchmark \
  --api-backend baseten \
  --api-base "your-endpoint-url" \
  --api-key "your-baseten-api-key" \
  --api-model-name "Mistral-7B-v0.1" \
  --model-tokenizer "mistralai/Mistral-7B-v0.1" \
  --task text-to-text \
  --additional-request-params '{"use_prompt_format": true, "temperature": 0.7}' \
  --num-concurrency 1 \
  --traffic-scenario "N(100,100)/(100,100)"
```

### Non-Streaming Mode

```bash
genai-bench benchmark \
  --api-backend baseten \
  --api-base "your-endpoint-url" \
  --api-key "your-baseten-api-key" \
  --api-model-name "test-model" \
  --model-tokenizer "test/tokenizer" \
  --task text-to-text \
  --disable-streaming \
  --additional-request-params '{"use_prompt_format": true, "temperature": 0.7}'
```

### Image-to-Text Benchmarking

```bash
genai-bench benchmark \
  --api-backend baseten \
  --api-base "your-endpoint-url" \
  --api-key "your-baseten-api-key" \
  --api-model-name "vision-model" \
  --model-tokenizer "vision/tokenizer" \
  --task image-text-to-text \
  --dataset-path /path/to/images \
  --max-requests-per-run 50 \
  --max-time-per-run 10
```

## Request Format Details

### OpenAI-Compatible Format Payload

```json
{
  "model": "model-name",
  "messages": [
    {
      "role": "user",
      "content": "Hello, world!"
    }
  ],
  "max_tokens": 100,
  "temperature": 0.7,
  "ignore_eos": true,
  "stream": true,
  "stream_options": {
    "include_usage": true
  }
}
```

### Simple Prompt Format Payload

```json
{
  "prompt": "Hello, world!",
  "max_tokens": 100,
  "temperature": 0.7,
  "stream": true
}
```

### Image Content Format

```json
{
  "model": "vision-model",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Describe this image"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,base64_image_data"
          }
        }
      ]
    }
  ],
  "max_tokens": 100,
  "temperature": 0.7,
  "stream": true
}
```

## Response Handling

### OpenAI-Compatible Response

```json
{
  "choices": [
    {
      "message": {
        "content": "This is the generated response"
      }
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 50,
    "total_tokens": 60
  }
}
```

### Simple Text Response

```
This is a plain text response
```

### JSON Response with Alternative Fields

```json
{
  "text": "Response from text field"
}
```

```json
{
  "output": "Response from output field"
}
```

```json
{
  "response": "Response from response field"
}
```

## Parameter Filtering

The Baseten backend automatically filters out certain parameters from `additional_request_params`:

- `stream`: Always controlled by global `--disable-streaming` flag
- `use_prompt_format`: Used internally for format selection, not sent to API

Other parameters (like `temperature`, `top_p`, etc.) are passed through to the API.

## Error Handling

The backend provides robust error handling for:

- Network connection issues
- HTTP error responses
- Malformed JSON responses
- Plain text responses
- Missing or invalid authentication

## Environment Variables

You can use environment variables for authentication:

```bash
export MODEL_API_KEY=your-baseten-api-key
```

## Supported Tasks

- **text-to-text**: Text generation with both formats
- **image-text-to-text**: Vision tasks with OpenAI format
- **text-to-embeddings**: Embedding generation (if supported by model)

## Best Practices

1. **Use OpenAI format for instruct models**: Models like Qwen-Instruct, Llama-Instruct, etc.
2. **Use prompt format for base models**: Models like Mistral-7B-v0.1, base Llama models, etc.
3. **Set appropriate temperature**: Avoid `temperature: 0.0` which may cause model server errors
4. **Test with small scenarios first**: Use `--num-concurrency 1` and small traffic scenarios for initial testing
5. **Monitor logs**: Watch for warnings about token estimation and response parsing

## Troubleshooting

### Common Issues

1. **Model Endpoint**: Verify the full URL is correct and the model is deployed and running

2. **Authentication Issues**: Ensure your API key is correct and has proper permissions

3. **Temperature Error**: If you see `ValueError: temperature (=0.0) has to be a strictly positive float`, set `temperature: 0.7` in `additional_request_params`

4. **Response Parsing Errors**: The backend automatically handles various response formats, but check logs for parsing warnings


### Debug Mode

Enable debug logging to see detailed request/response information:

```bash
export LOG_LEVEL=DEBUG
genai-bench benchmark --api-backend baseten ...
```

## Related Documentation

- [Multi-Cloud Quick Reference](multi-cloud-quick-reference.md)
- [Multi-Cloud Authentication & Storage](multi-cloud-auth-storage.md)
- [Run Benchmark Guide](run-benchmark.md) 