# Uploading Benchmark Results to Cloud Storage

> **Note**: For comprehensive multi-cloud storage authentication and configuration, see the [Multi-Cloud Authentication & Storage Guide](multi-cloud-auth-storage.md).

## OCI Object Storage (Legacy)
GenAI Bench supports uploading benchmark results directly to OCI Object Storage. This feature is useful for:

- Storing benchmark results in a centralized location
- Sharing results with team members
- Maintaining a historical record of benchmarks
- Analyzing results across different runs

To enable result uploading, use the following options with the `benchmark` command:

```bash
genai-bench benchmark \
    --api-base "http://localhost:8082" \
    --api-key "your-openai-api-key" \
    --api-model-name "vllm-model" \
    --model-tokenizer "/mnt/data/models/Meta-Llama-3.1-70B-Instruct" \
    --task text-to-text \
    --max-time-per-run 15 \
    --max-requests-per-run 300 \
    --server-engine "vLLM" \
    --server-gpu-type "H100" \
    --server-version "v0.6.0" \
    --server-gpu-count 4 \
    --upload-results \
    --storage-bucket "your-bucket-name" \
    --storage-provider oci
```
By default, GenAI Bench uses OCI User Principal for authentication and authorization.
The default namespace is the current tenancy, and the default region is the current region in which the client is positioned.
You can override the namespace and region using the `--namespace` and `--region` options, respectively.
Alternatively, you can change the authentication and authorization mechanism using the `--auth` option.
The default object prefix is empty, but you can specify a prefix using the `--storage-prefix` option.

## Multi-Cloud Storage Support

GenAI Bench now supports multiple cloud storage providers:

- **AWS S3**: Use `--storage-provider aws`
- **Azure Blob Storage**: Use `--storage-provider azure`
- **GCP Cloud Storage**: Use `--storage-provider gcp`
- **GitHub Releases**: Use `--storage-provider github`

For detailed configuration and authentication options for each provider, please refer to the [Multi-Cloud Authentication & Storage Guide](multi-cloud-auth-storage.md).