# Custom API Backends Guide

This guide explains how to create and use custom API backends with genai-bench, allowing you to extend the tool to support any API endpoint without modifying the core codebase.

## Overview

You can define your own API backend by creating a Python file with a class that inherits from `BaseUser`, then point genai-bench to that file using the `--custom-backend` parameter. The easiest way to get started is to copy an existing built-in backend (e.g., `genai_bench/user/openai_user.py`) and modify it for your needs.

## Quick Start

```bash
genai-bench benchmark \
    --api-backend custom \
    --custom-backend /path/to/your_custom_backend.py \
    --api-base <your-api-endpoint> \
    --api-model-name <your-model-id> \
    --task text-to-text \
    --model-tokenizer <your-tokenizer>
```

## Creating a Custom Backend

### Minimum Requirements

Your custom backend class must:

1. **Inherit from `BaseUser`**
   ```python
   from genai_bench.user.base_user import BaseUser

   class MyCustomUser(BaseUser):
       pass
   ```

2. **Set the `BACKEND_NAME` class attribute**
   ```python
   BACKEND_NAME = "my-custom-backend"
   ```

3. **Define `supported_tasks` mapping**
   ```python
   supported_tasks = {
       "text-to-text": "chat",
       "text-to-embeddings": "embeddings",
   }
   ```

   The keys are task types (as used in `--task`), and values are method names.

4. **Implement task methods with the `@task` decorator**
   ```python
   from locust import task

   @task
   def chat(self):
       # Get request from sampler
       user_request = self.sample()

       # Make API call
       # ...

       # Collect metrics
       self.collect_metrics(user_response, "/my-backend/chat")
   ```

5. **Implement `on_start()` for initialization**
   ```python
   def on_start(self):
       """Initialize API client, auth, etc."""
       # Set up your API client here
       pass
   ```

### Complete Example Structure

```python
from locust import task
from genai_bench.user.base_user import BaseUser
from genai_bench.protocol import UserChatRequest, UserChatResponse
from genai_bench.logging import init_logger

logger = init_logger(__name__)

class MyCustomUser(BaseUser):
    """My custom API backend."""

    BACKEND_NAME = "my-custom-backend"

    supported_tasks = {
        "text-to-text": "chat",
    }

    def on_start(self):
        """Initialize client."""
        # Initialize your API client
        self.client = ...

    @task
    def chat(self):
        """Handle chat requests."""
        # Get request
        user_request = self.sample()

        # Validate request type
        if not isinstance(user_request, UserChatRequest):
            raise AttributeError(f"Expected UserChatRequest, got {type(user_request)}")

        # Make API call
        try:
            response = self._make_request(user_request)
            self.collect_metrics(response, "/my-backend/chat")
        except Exception as e:
            logger.exception(f"Request failed: {e}")
            error_response = UserResponse(status_code=500, error_message=str(e))
            self.collect_metrics(error_response, "/my-backend/chat")

    def _make_request(self, request: UserChatRequest) -> UserChatResponse:
        """Make the actual API request."""
        # Implement your API logic here
        pass
```

## Request and Response Types

### Request Types

- **`UserChatRequest`**: For text-to-text and image-text-to-text tasks
  - `prompt`: The input prompt
  - `model`: Model identifier
  - `max_tokens`: Maximum tokens to generate
  - `num_prefill_tokens`: Number of tokens in the prompt
  - `additional_request_params`: Dict of extra parameters

- **`UserEmbeddingRequest`**: For text-to-embeddings tasks
  - `documents`: List of texts to embed
  - `model`: Model identifier
  - `num_prefill_tokens`: Total tokens in documents

### Response Types

- **`UserChatResponse`**: For chat completions
  ```python
  UserChatResponse(
      status_code=200,
      generated_text="Generated response",
      tokens_received=150,
      time_at_first_token=0.1,  # Time when first token arrived
      num_prefill_tokens=50,
      start_time=start_time,
      end_time=end_time,
  )
  ```

- **`UserResponse`**: For simple responses or errors
  ```python
  UserResponse(
      status_code=200,
      start_time=start_time,
      end_time=end_time,
      num_prefill_tokens=50,
  )
  ```

## Key Methods and Patterns

### Getting Requests

Use `self.sample()` to get the next request:

```python
user_request = self.sample()
```

The sampler automatically generates appropriate requests based on the traffic scenario.

### Collecting Metrics

Always call `self.collect_metrics()` to report results:

```python
self.collect_metrics(user_response, "/endpoint/name")
```

This handles both successful and error responses.

### Authentication

Custom backends handle their own authentication in `on_start()`:

```python
def on_start(self):
    # Option 1: Use environment variables
    api_key = os.environ.get("MY_API_KEY")

    # Option 2: Use CLI parameters via self.environment
    # (requires custom CLI options)

    # Option 3: Load from config file
    with open("config.json") as f:
        config = json.load(f)

    self.client = MyAPIClient(api_key=api_key)
```

### Streaming Responses

For streaming APIs, parse the stream and track timing:

```python
import time

start_time = time.monotonic()
time_at_first_token = None
generated_text = ""

for chunk in stream:
    content = parse_chunk(chunk)
    if content and not time_at_first_token:
        time_at_first_token = time.monotonic()
    generated_text += content

end_time = time.monotonic()
```

### Error Handling

Always wrap API calls in try-except:

```python
try:
    response = self._make_request(user_request)
    self.collect_metrics(response, "/endpoint")
except Exception as e:
    logger.exception(f"Request failed: {e}")
    error_response = UserResponse(
        status_code=500,
        error_message=str(e),
    )
    self.collect_metrics(error_response, "/endpoint")
```

## Example: AWS SageMaker Backend

See [custom_sagemaker_backend.py](custom_sagemaker_backend.py) for a complete example implementing a custom backend for AWS SageMaker endpoints.

Key features demonstrated:
- AWS SDK initialization
- Streaming response parsing
- OpenAI-compatible format handling
- Token counting fallback
- Error handling

## Tips and Best Practices

1. **Start from an existing backend**: Copy `genai_bench/user/openai_user.py` and modify it for your needs. See also `examples/minimal_custom_backend.py` for a stripped-down starting point.

2. **Test incrementally**: Start with a simple non-streaming implementation, then add streaming.

3. **Use logging**: The logger is your friend for debugging:
   ```python
   logger.info("Making request...")
   logger.debug(f"Request body: {body}")
   logger.exception("Error occurred")
   ```

4. **Handle token counting**: If your API doesn't return token counts, estimate them:
   ```python
   tokens = self.environment.sampler.get_token_length(
       text, add_special_tokens=False
   )
   ```

5. **Support additional parameters**: Read from `request.additional_request_params`:
   ```python
   temperature = request.additional_request_params.get("temperature", 1.0)
   ```

6. **Document your backend**: Add a docstring explaining what API it's for and any special requirements.

## Advanced: Class Name Specification

If your file has multiple classes, specify which one to use by appending the class name with a colon:

```bash
genai-bench benchmark \
    --api-backend custom \
    --custom-backend /path/to/your_backend.py:MyUser \
    --api-base <your-api-endpoint> \
    ...
```

Example file with multiple classes:
```python
# In your_backend.py
class MyUser(BaseUser):
    BACKEND_NAME = "my-backend"
    # ...

class AnotherUser(BaseUser):
    BACKEND_NAME = "another-backend"
    # ...
```

Without the class name suffix (`:MyUser`), the loader will automatically detect a BaseUser subclass if there's only one. With multiple classes, you must specify which one to use.

## Troubleshooting

### "No BaseUser subclass found"
- Ensure your class inherits from `BaseUser`
- Check that the file is valid Python
- Verify import statements are correct

### "Task 'X' is not supported"
- Add the task to `supported_tasks` dict
- Implement the corresponding method
- Use the `@task` decorator

### "Auth provider not set"
- Custom backends don't use the standard auth provider
- Implement your own auth in `on_start()`

### "Module not found" errors
- Install required dependencies: `pip install <package>`
- Document dependencies in your backend's docstring

## Getting Help

- Start from a built-in backend: `genai_bench/user/openai_user.py`
- Check the minimal example: `examples/minimal_custom_backend.py`
- Review other built-in backends in `genai_bench/user/`
- See the BaseUser interface in `genai_bench/user/base_user.py`
- Open an issue on GitHub for support
