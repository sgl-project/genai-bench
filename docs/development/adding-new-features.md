# Adding New Features

This guide covers how to add new features to GenAI Bench, including model providers, storage providers, and tasks.

## Adding a New Model Provider

1. Create auth provider in `genai_bench/auth/`
2. Create user class in `genai_bench/user/`
3. Update `UnifiedAuthFactory`
4. Add validation in `cli/validation.py`
5. Write tests

## Adding a New Storage Provider

1. Create storage auth in `genai_bench/auth/`
2. Create storage implementation in `genai_bench/storage/`
3. Update `StorageFactory`
4. Write tests

## Adding a New Task

This guide explains how to add support for a new task in `genai-bench`. Follow the steps below to ensure consistency and compatibility with the existing codebase.

### 1. Define the Request and Response in `protocol.py`

#### Steps

1. Add relevant fields to the appropriate request/response data classes in [`protocol.py`](https://github.com/sgl-project/genai-bench/blob/main/genai_bench/protocol.py)
2. If the new task involves a new input-output modality, create a new request/response class.
3. Use existing request/response classes (`UserChatRequest`, `UserEmbeddingRequest`, `UserImageChatRequest`, etc.) if they suffice.

#### Example

```python
class UserTextToImageRequest(UserRequest):
    """Represents a request for generating images from text."""
    prompt: str
    num_images: int = Field(..., description="Number of images to generate.")
    image_resolution: Tuple[int, int] = Field(..., description="Resolution of the generated images.")
```

### 2. Update or Create a Sampler

#### 2.1 If Input Modality Is Supported by an Existing Sampler

1. Check if the current [`TextSampler`](https://github.com/sgl-project/genai-bench/blob/main/genai_bench/sampling/text_sampler.py) or [`ImageSampler`](https://github.com/sgl-project/genai-bench/blob/main/genai_bench/sampling/image_sampler.py) supports the input-modality.
2. Add request creation logic in the relevant `TextSampler` or `ImageSampler` class.
3. Refactor the sampler's `_create_request` method to support the new task.
4. **Tip:** Avoid adding long `if-else` chains for new tasks. Utilize helper methods or design a request creator pattern if needed.

#### 2.2 If Input Modality Is Not Supported

1. Create a new sampler class inheriting from [`BaseSampler`](https://github.com/sgl-project/genai-bench/blob/main/genai_bench/sampling/base_sampler.py).
2. Define the `sample` method to generate requests for the new task.
3. Refer to `TextSampler` and `ImageSampler` for implementation patterns.
4. Add utility functions for data preprocessing or validation specific to the new modality if necessary.

#### Example for a New Sampler

```python
class AudioSampler(Sampler):
    input_modality = "audio"
    supported_tasks = {"audio-to-text", "audio-to-embeddings"}

    def sample(self, scenario: Scenario) -> UserRequest:
        # Validate scenario
        self._validate_scenario(scenario)

        if self.output_modality == "text":
            return self._create_audio_to_text_request(scenario)
        elif self.output_modality == "embeddings":
            return self._create_audio_to_embeddings_request(scenario)
        else:
            raise ValueError(f"Unsupported output_modality: {self.output_modality}")
```

### 3. Add Task Support in the User Class

Each `User` corresponds to one API backend, such as [`OpenAIUser`](https://github.com/sgl-project/genai-bench/blob/main/genai_bench/user/openai_user.py) for OpenAI. Users can have multiple tasks, each corresponding to an endpoint.

#### Steps

1. Add the new task to the `supported_tasks` dictionary in the relevant `User` class.
2. Map the new task to its corresponding function name in the dictionary.
3. Implement the new function in the `User` class for handling the task logic.
4. If the new task uses an existing endpoint, refactor the function to support both tasks without duplicating logic.
5. **Important:** Avoid creating multiple functions for tasks that use the same endpoint.

#### Example

```python
class OpenAIUser(BaseUser):
    supported_tasks = {
        "text-to-text": "chat",
        "image-text-to-text": "chat",
        "text-to-embeddings": "embeddings",
        "audio-to-text": "audio_to_text",  # New task added
    }

    def audio_to_text(self):
        # Implement the logic for audio-to-text task
        endpoint = "/v1/audio/transcriptions"
        user_request = self.sample()

        # Add payload and send request
        payload = {"audio": user_request.audio_file}
        self.send_request(False, endpoint, payload, self.parse_audio_response)
```

### 4. Add Unit Tests

#### Steps

1. Add tests for the new task in the appropriate test files.
2. Include tests for:
    - Request creation in the sampler.
    - Task validation in the `User` class.
    - End-to-end workflow using the new task.

### 5. Update Documentation

#### Steps

1. Add the new task to the list of supported tasks in the [Task Definition guide](../getting-started/task-definition.md).
2. Provide sample commands and explain any required configuration changes.
3. Mention the new task in this contributing guide for future developers.
