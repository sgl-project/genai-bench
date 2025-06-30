# Task Definition

Tasks in `genai-bench` define the type of benchmark you want to run, based on the input modality (e.g., text, image) and output modality (e.g., text, embeddings). Tasks are specified using the `--task` option in the `genai-bench benchmark` command.

Each task follows the pattern:

```bash
<input_modality>-to-<output_modality>
```

Here are the currently supported tasks:

**NOTE**: Task compatibility may vary depending on the API format.

| Task Name             | Description                                                                                 |
|-----------------------|---------------------------------------------------------------------------------------------|
| `text-to-text`        | Benchmarks generating text output from text input, such as chat or QA tasks.                |
| `text-to-embeddings`  | Benchmarks generating embeddings from text input, often for semantic search.                |
| `image-text-to-text`  | Benchmarks generating text from images and text prompts, such as visual question answering. |
| `image-to-embeddings` | Benchmarks generating embeddings from images, often for image similarity.                   |

### How Tasks Work

* **Input Modality:** Defines the type of input data the task operates on, such as text or images.

* **Output Modality:** Defines the type of output the task generates, such as text or embeddings.

When you specify a task, the appropriate sampler (`TextSampler` or `ImageSampler`) and request type (`UserChatRequest`, `UserEmbeddingRequest`, etc.) are automatically selected based on the input and output modalities.

### Example Task Usage

* For a **text-to-text** task (e.g., generating a response to a text prompt, typical chat completions):

    ```bash
    genai-bench benchmark --task text-to-text ...
    ```

* For an **image-to-text** task (e.g., generating a response for an image and text interleave message):

    ```bash
    genai-bench benchmark --task image-to-text ...
    ```

* For an **image-to-embeddings** task (e.g., generating embeddings for similarity search):

    ```bash
    genai-bench benchmark --task text-to-embeddings ...
    ```