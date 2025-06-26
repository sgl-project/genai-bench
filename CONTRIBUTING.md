# Contribution Guideline

Welcome and thank you for your interest in contributing to genai-bench.

## Coding Style Guide

genai-bench uses python 3.11, and we adhere to [Google Python style guide](https://google.github.io/styleguide/pyguide.html).

We use `make format` to format our code using `isort` and `ruff`. The detailed configuration can be found in
[pyproject.toml](pyproject.toml).

## Pull Requests

When submitting a pull request, please:

1. `git pull origin main` to make sure your code has been rebased on top of the latest commit on the main branch.
2. Ensure code is properly formatted and passed every lint checks by running `make check`.
3. Add new test cases to stay robust and correct. In the case of a bug fix, the tests should fail without your code
changes. For new features, try to cover as many variants as reasonably possible. You can use `make test` to check the
test coverage. Or use `make test_changed` to test the test coverage for your own branch. We enforce to keep a test
coverage about 90%.

### PR Template

It is required to classify your PR and make the commit message concise and useful. Prefix the PR title appropriately
to indicate the type of change. Please use one of the following:

`[Bugfix]` for bug fixes.

`[Core]` for core backend changes. This includes build, version upgrade, changes in user and sampling.

`[Metrics]` for changes made to metrics.

`[Frontend]` for UI dashboard and CLI entrypoint related changes.

`[Docs]` for changes related to documentation.

`[CI/Tests]` for unittests and integration tests.

`[Report]` for changes in generating plots and excel reports.

`[Misc]` for PRs that do not fit the above categories. Please use this sparingly.

Open source community also recommends to keep the commit message title within 52 chars and each line in message content
within 72 chars.

### Code Reviews

All submissions, including submissions by project members, require a code review.
To make the review process as smooth as possible, please:

1. Keep your changes as concise as possible.
   If your pull request involves multiple unrelated changes, consider splitting it into separate pull requests.
2. Respond to all comments within a reasonable time frame.
   If a comment isn't clear,
   or you disagree with a suggestion, feel free to ask for clarification or discuss the suggestion.
3. Provide constructive feedback and meaningful comments. Focus on specific improvements
   and suggestions that can enhance the code quality or functionality. Remember to
   acknowledge and respect the work the author has already put into the submission.


## Setup Development Environment

### `make`

genai-bench utilizes `make` for a lot of useful commands.

If your laptop doesn't have `GNU make` installed, (check this by typing `make --version` in your terminal),
you can ask our GenerativeAI's chatbot about how to install it in your system.

### `uv`

Install uv with `make uv` or install it from the [official website](https://docs.astral.sh/uv/).
If installing from the website, create a project venv with `uv venv -p python3.11`.

Once you have `make` and `uv` installed, you can follow the command below to build genai-bench wheel:

```shell
# check out commands genai-bench supports
make help
#activate virtual env managed by uv
source .venv/bin/activate
# install dependencies
make install
```

You can utilize wheel to install genai-bench.

```shell
# build a .whl under genai-bench/dist
make build
# send the wheel to your remote machine if applies
rsync --delete -avz ~/genai-bench/dist/<.wheel> <remote-user>@<remote-ip>:<dest-addr>
```

On your remote machine, you can simply use the `pip` to install genai-bench.

```shell
pip install <dest-addr>/<.wheel>
```

# Development Guide: Adding a New Task in `genai-bench`

This guide explains how to add support for a new task in `genai-bench`. Follow the steps below to ensure consistency and compatibility with the existing codebase.

---

## 1. Define the Request and Response in `protocol.py`

### Steps

1. Add relevant fields to the appropriate request/response data classes in [`protocol.py`](genai_bench/protocol.py)
2. If the new task involves a new input-output modality, create a new request/response class.
3. Use existing request/response classes (`UserChatRequest`, `UserEmbeddingRequest`, `UserImageChatRequest`, etc.) if they suffice.

### Example

```python
class UserTextToImageRequest(UserRequest):
    """Represents a request for generating images from text."""
    prompt: str
    num_images: int = Field(..., description="Number of images to generate.")
    image_resolution: Tuple[int, int] = Field(..., description="Resolution of the generated images.")
```

---

## 2. Update or Create a Sampler

### 2.1 If Input Modality Is Supported by an Existing Sampler

1. Check if the current [`TextSampler`](genai_bench/sampling/text_sampler.py) or [`ImageSampler`](genai_bench/sampling/image_sampler.py) supports the input-modality.
2. Add request creation logic in the relevant `TextSampler` or `ImageSampler` class.
3. Refactor the sampler's `_create_request` method to support the new task.
4. **Tip:** Avoid adding long `if-else` chains for new tasks. Utilize helper methods or design a request creator pattern if needed.

### 2.2 If Input Modality Is Not Supported

1. Create a new sampler class inheriting from [`BaseSampler`](genai_bench/sampling/base_sampler.py).
2. Define the `sample` method to generate requests for the new task.
3. Refer to `TextSampler` and `ImageSampler` for implementation patterns.
4. Add utility functions for data preprocessing or validation specific to the new modality if necessary.

### Example for a New Sampler

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

---

## 3. Add Task Support in the User Class

Each `User` corresponds to one API backend, such as [`OpenAIUser`](genai_bench/user/openai_user.py) for OpenAI. Users can have multiple tasks, each corresponding to an endpoint.

### Steps

1. Add the new task to the `supported_tasks` dictionary in the relevant `User` class.
2. Map the new task to its corresponding function name in the dictionary.
3. Implement the new function in the `User` class for handling the task logic.
4. If the new task uses an existing endpoint, refactor the function to support both tasks without duplicating logic.
5. **Important:** Avoid creating multiple functions for tasks that use the same endpoint.

### Example

```python
class OpenAIUser(BaseUser):
    supported_tasks = {
        "text-to-text": "chat",
        "image-to-text": "chat",
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

---

## 4. Add Unit Tests

### Steps

1. Add tests for the new task in the appropriate test files.
2. Include tests for:
   - Request creation in the sampler.
   - Task validation in the `User` class.
   - End-to-end workflow using the new task.

---

## 5. Update Documentation

### Steps

1. Add the new task to the list of supported tasks in [`USER_GUIDE.md#task-definition`](USER_GUIDE.md#task-definition).
2. Provide sample commands and explain any required configuration changes.
3. Mention the new task in `CONTRIBUTING.md` for future developers.
