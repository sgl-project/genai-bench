# User Guide

This guide covers everything you need to know to effectively use GenAI Bench for benchmarking LLM endpoints.

## What You'll Learn

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } **Running Benchmarks**

    ---

    Learn how to run benchmarks against various LLM endpoints

    [:octicons-arrow-right-24: Run Benchmark](run-benchmark.md)

-   :material-cloud:{ .lg .middle } **Multi-Cloud Setup**

    ---

    Configure authentication for AWS, Azure, GCP, OCI, Baseten, and more

    [:octicons-arrow-right-24: Multi-Cloud Guide](multi-cloud-auth-storage.md)

-   :material-cog:{ .lg .middle } **Baseten Support**

    ---

    Learn about dual format support and flexible response handling

    [:octicons-arrow-right-24: Baseten Guide](baseten-support.md)

-   :material-docker:{ .lg .middle } **Docker Deployment**

    ---

    Run GenAI Bench in containerized environments

    [:octicons-arrow-right-24: Docker Guide](run-benchmark-using-docker.md)

-   :material-file-excel:{ .lg .middle } **Excel Reports**

    ---

    Generate comprehensive Excel reports from benchmark results

    [:octicons-arrow-right-24: Excel Reports](generate-excel-sheet.md)

</div>

## Common Workflows

### Basic Benchmarking

1. **Choose your model provider** - OpenAI, AWS Bedrock, Azure OpenAI, etc.
2. **Configure authentication** - API keys, IAM roles, or service accounts
3. **Run the benchmark** - Specify task type and parameters
4. **Analyze results** - View real-time dashboard or generate reports

### Cross-Cloud Benchmarking

Benchmark models from one provider while storing results in another:

```bash
# Benchmark OpenAI, store in AWS S3
genai-bench benchmark \
  --api-backend openai \
  --api-key $OPENAI_KEY \
  --upload-results \
  --storage-provider aws \
  --storage-bucket my-results
```

### Multi-Modal Tasks

Support for text, embeddings, and vision tasks:

- `text-to-text` - Chat and completion tasks
- `text-to-embeddings` - Embedding generation
- `image-text-to-text` - Vision-language tasks
- `text-to-rerank` - Document reranking

## Need Help?

- Check the [Quick Reference](multi-cloud-quick-reference.md) for common commands
- Review [Command Guidelines](../getting-started/command-guidelines.md) for detailed options
- See [Troubleshooting](multi-cloud-auth-storage.md#troubleshooting) for common issues