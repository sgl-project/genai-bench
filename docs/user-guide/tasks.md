# Tasks and Benchmarks

This guide explains the different task types supported by GenAI Bench and how to configure them for various benchmarking scenarios.

## Task Overview

Tasks in GenAI Bench define the type of benchmark you want to run, based on the input modality (e.g., text, image) and output modality (e.g., text, embeddings). Each task follows the pattern:

```bash
<input_modality>-to-<output_modality>
```

## Supported Tasks

### Text-to-Text (`text-to-text`)

Generate text responses from text input, such as chat completions or question answering.

#### Use Cases
- **Chat Applications**: Conversational AI, customer support
- **Question Answering**: Knowledge base queries, educational content
- **Text Generation**: Content creation, summarization
- **Code Generation**: Programming assistance, code completion

#### Configuration Example

```bash
genai-bench benchmark \
    --api-backend openai \
    --api-base "http://localhost:8082" \
    --api-key "your-api-key" \
    --api-model-name "llama-2-7b" \
    --task text-to-text \
    --dataset-name "sonnet.txt" \
    --max-time-per-run 300
```

#### Request Format

```json
{
  "messages": [
    {"role": "user", "content": "What is the capital of France?"}
  ],
  "max_tokens": 100,
  "temperature": 0.7
}
```

### Text-to-Embeddings (`text-to-embeddings`)

Generate embeddings from text input for semantic search and similarity tasks.

#### Use Cases
- **Semantic Search**: Document retrieval, content discovery
- **Similarity Matching**: Duplicate detection, clustering
- **Recommendation Systems**: Content recommendations
- **Information Retrieval**: Search engines, knowledge graphs

#### Configuration Example

```bash
genai-bench benchmark \
    --api-backend openai \
    --api-base "http://localhost:8082" \
    --api-key "your-api-key" \
    --api-model-name "text-embedding-ada-002" \
    --task text-to-embeddings \
    --dataset-name "sonnet.txt" \
    --max-time-per-run 300
```

#### Request Format

```json
{
  "input": "This is a sample text for embedding generation",
  "model": "text-embedding-ada-002"
}
```

### Image-Text-to-Text (`image-text-to-text`)

Generate text responses from images combined with text prompts.

#### Use Cases
- **Visual Question Answering**: Image-based Q&A
- **Image Captioning**: Descriptive text generation
- **Visual Analysis**: Object detection, scene understanding
- **Multimodal Chat**: Interactive image-based conversations

#### Configuration Example

```bash
genai-bench benchmark \
    --api-backend openai \
    --api-base "http://localhost:8082" \
    --api-key "your-api-key" \
    --api-model-name "gpt-4-vision" \
    --task image-text-to-text \
    --dataset-name "vision-dataset" \
    --max-time-per-run 300
```

#### Request Format

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "What do you see in this image?"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
      ]
    }
  ],
  "max_tokens": 150
}
```

### Image-to-Embeddings (`image-to-embeddings`)

Generate embeddings from images for image similarity and search tasks.

#### Use Cases
- **Image Similarity**: Duplicate image detection
- **Visual Search**: Find similar images
- **Content Moderation**: Inappropriate content detection
- **Image Clustering**: Organize image collections

#### Configuration Example

```bash
genai-bench benchmark \
    --api-backend openai \
    --api-base "http://localhost:8082" \
    --api-key "your-api-key" \
    --api-model-name "clip-vit-base-patch32" \
    --task image-to-embeddings \
    --dataset-name "image-dataset" \
    --max-time-per-run 300
```

#### Request Format

```json
{
  "input": "data:image/jpeg;base64,...",
  "model": "clip-vit-base-patch32"
}
```

## Task Compatibility

### API Backend Support

| Task | OpenAI | Cohere | OCI Cohere |
|------|--------|--------|------------|
| `text-to-text` | ✅ | ✅ | ✅ |
| `text-to-embeddings` | ✅ | ✅ | ✅ |
| `image-text-to-text` | ✅ | ✅ | ✅ |
| `image-to-embeddings` | ✅ | ✅ | ✅ |

### Model Requirements

#### Text-to-Text Models
- **Chat Models**: GPT-3.5-turbo, GPT-4, Llama-2, etc.
- **Completion Models**: text-davinci-003, etc.

#### Embedding Models
- **Text Embeddings**: text-embedding-ada-002, Cohere embeddings
- **Image Embeddings**: CLIP, DALL-E embeddings

#### Vision Models
- **Multimodal**: GPT-4V, Claude 3 Vision, etc.
- **Image Understanding**: Models with vision capabilities

## Dataset Configuration

### Built-in Datasets

GenAI Bench provides several built-in datasets:

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

For vision tasks, organize your dataset:

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

### Hugging Face Datasets

Use datasets from Hugging Face:

```bash
--dataset-name "squad"
--dataset-split "train"
--dataset-column "question"
```

## Advanced Task Configuration

### Tokenizer Configuration

For accurate token counting, specify the tokenizer:

```bash
# Use model's tokenizer
--model-tokenizer "/path/to/tokenizer"

# Use Hugging Face tokenizer
--model-tokenizer "meta-llama/Llama-2-7b-chat-hf"

# Use specific tokenizer
--model-tokenizer "gpt2"
```

### Request Parameters

Customize request parameters for different tasks:

#### Text-to-Text Parameters

```bash
# Temperature (creativity)
--temperature 0.7

# Maximum tokens
--max-tokens 100

# Top-p sampling
--top-p 0.9

# Frequency penalty
--frequency-penalty 0.1

# Presence penalty
--presence-penalty 0.1
```

#### Embedding Parameters

```bash
# Embedding dimensions
--embedding-dimensions 1536

# Normalize embeddings
--normalize-embeddings true
```

### Task-Specific Metrics

Different tasks collect different metrics:

#### Text-to-Text Metrics
- **TTFT**: Time to first token
- **TPOT**: Time per output token
- **Output throughput**: Tokens per second
- **Response quality**: Perplexity, accuracy

#### Embedding Metrics
- **Embedding latency**: Time to generate embeddings
- **Embedding quality**: Cosine similarity, clustering quality
- **Batch processing**: Throughput for multiple inputs

#### Vision Metrics
- **Image processing time**: Time to process images
- **Multimodal latency**: Combined text + image processing
- **Vision accuracy**: Object detection, caption quality

## Best Practices

### Task Selection

1. **Match Your Use Case**: Choose the task that matches your application
2. **Consider Model Capabilities**: Ensure your model supports the task
3. **Test Compatibility**: Verify API backend support

### Dataset Preparation

1. **Representative Data**: Use data similar to your production workload
2. **Appropriate Size**: Balance between coverage and benchmark duration
3. **Quality Control**: Ensure data quality and consistency

### Benchmark Configuration

1. **Start Simple**: Begin with basic configurations
2. **Gradual Scaling**: Increase complexity step by step
3. **Consistent Parameters**: Use same settings for comparisons

### Result Interpretation

1. **Task-Specific Metrics**: Focus on relevant metrics for your task
2. **Context Matters**: Consider your specific use case requirements
3. **Baseline Comparison**: Compare against known good performance

## Troubleshooting

### Common Issues

#### Task Not Supported
```
Error: Task 'unsupported-task' is not supported by backend 'openai'
```

**Solution**: Check task compatibility with your API backend.

#### Model Incompatibility
```
Error: Model 'text-model' does not support vision tasks
```

**Solution**: Use a model that supports your task type.

#### Dataset Issues
```
Error: Dataset format not compatible with task 'image-text-to-text'
```

**Solution**: Ensure your dataset matches the task requirements.

### Getting Help

- Check the [CLI Reference](cli.md) for detailed command options
- Review [Examples](../examples/basic-benchmarks.md) for practical scenarios
- Open an [issue](https://github.com/sgl-project/genai-bench/issues) for bugs

## Next Steps

- Learn about [Results Analysis](analysis.md) for understanding metrics
- Explore [Examples](../examples/basic-benchmarks.md) for practical use cases
- Check out the [CLI Reference](cli.md) for complete command documentation 