import click
import oci

from genai_bench.cli.validation import (
    DEFAULT_BATCH_SIZES,
    DEFAULT_NUM_CONCURRENCIES,
    set_model_from_tokenizer,
    validate_additional_request_params,
    validate_api_backend,
    validate_api_key,
    validate_dataset_config,
    validate_dataset_path_callback,
    validate_iteration_params,
    validate_object_storage_options,
    validate_task,
    validate_traffic_scenario_callback,
    validate_warmup_cooldown_ratio_options,
)


# Group API-related options
# NOTE: when adding new options, please add them at the top of the func, as
# the decorator works in reversed order
def api_options(func):
    func = click.option(
        "--additional-request-params",
        type=str,
        default=None,
        callback=validate_additional_request_params,
        help="A dictionary containing additional params to be sent to the "
        "backend model server. Default: {}. Example: "
        "'{'temperature': 2.0, 'ignore_eos': false}'",
    )(func)
    func = click.option(
        "--api-model-name",
        type=str,
        required=True,
        prompt=True,
        help="The model to use for the backend server request body.",
    )(func)
    func = click.option(
        "--task",
        type=click.Choice(
            [
                "text-to-text",
                "text-to-embeddings",
                "text-to-rerank",
                "image-text-to-text",
                "image-to-embeddings",
            ],
            case_sensitive=False,
        ),
        required=True,
        prompt=True,
        callback=validate_task,
        help="The task to benchmark: it follows `<input_modality>-to-"
        "<output_modality>` pattern. Currently we support `text-to-text`,"
        " `image-text-to-text`, `text-to-embeddings, and `image-to-embeddings`.",
    )(func)
    func = click.option(
        "--api-key",
        type=str,
        required=False,
        prompt=False,
        callback=validate_api_key,
        help="The API key for authentication. Required for OpenAI backend only.",
    )(func)
    func = click.option(
        "--api-base",
        type=str,
        required=True,
        prompt=True,
        help="The API base URL.",
    )(func)
    func = click.option(
        "--api-backend",
        type=click.Choice(
            [
                "openai",
                "oci-cohere",
                "oci-genai",
                "cohere",
                "aws-bedrock",
                "azure-openai",
                "gcp-vertex",
                "vllm",
                "sglang",
            ],
            case_sensitive=False,
        ),
        required=True,
        prompt=True,
        callback=validate_api_backend,
        help="The API backend to use. Supports major cloud providers and "
        "open-source servers.",
    )(func)
    return func


# Model endpoint authentication options
def model_auth_options(func):
    """Model endpoint authentication options.

    Supports authentication for various model providers:
    - OpenAI: API key authentication
    - OCI: Multiple auth methods (user_principal, instance_principal, etc.)
    - AWS Bedrock: IAM credentials or profile
    - Azure OpenAI: API key or Azure AD
    - GCP Vertex: Service account or API key
    - vLLM/SGLang: Optional bearer token
    """
    # Model provider selection (determines which auth options are relevant)
    func = click.option(
        "--model-auth-type",
        type=str,
        default=None,
        help="Authentication type for the model provider. "
        "Options depend on --api-backend.",
    )(func)

    # OpenAI and general API key auth
    func = click.option(
        "--model-api-key",
        type=str,
        default=None,
        envvar="MODEL_API_KEY",
        help="API key for model authentication "
        "(OpenAI, Azure OpenAI, GCP with API key).",
    )(func)

    # AWS Bedrock auth options
    func = click.option(
        "--aws-access-key-id",
        type=str,
        default=None,
        envvar="AWS_ACCESS_KEY_ID",
        help="AWS access key ID for Bedrock authentication.",
    )(func)

    func = click.option(
        "--aws-secret-access-key",
        type=str,
        default=None,
        envvar="AWS_SECRET_ACCESS_KEY",
        help="AWS secret access key for Bedrock authentication.",
    )(func)

    func = click.option(
        "--aws-session-token",
        type=str,
        default=None,
        envvar="AWS_SESSION_TOKEN",
        help="AWS session token for temporary credentials.",
    )(func)

    func = click.option(
        "--aws-profile",
        type=str,
        default=None,
        envvar="AWS_PROFILE",
        help="AWS profile name to use from credentials file.",
    )(func)

    func = click.option(
        "--aws-region",
        type=str,
        default=None,
        envvar="AWS_DEFAULT_REGION",
        help="AWS region for Bedrock.",
    )(func)

    # Azure OpenAI auth options
    func = click.option(
        "--azure-endpoint",
        type=str,
        default=None,
        envvar="AZURE_OPENAI_ENDPOINT",
        help="Azure OpenAI endpoint URL.",
    )(func)

    func = click.option(
        "--azure-deployment",
        type=str,
        default=None,
        envvar="AZURE_OPENAI_DEPLOYMENT",
        help="Azure OpenAI deployment name.",
    )(func)

    func = click.option(
        "--azure-api-version",
        type=str,
        default="2024-02-01",
        envvar="AZURE_OPENAI_API_VERSION",
        help="Azure OpenAI API version.",
    )(func)

    func = click.option(
        "--azure-ad-token",
        type=str,
        default=None,
        envvar="AZURE_AD_TOKEN",
        help="Azure AD token for authentication.",
    )(func)

    # GCP Vertex AI auth options
    func = click.option(
        "--gcp-project-id",
        type=str,
        default=None,
        envvar="GCP_PROJECT_ID",
        help="GCP project ID for Vertex AI.",
    )(func)

    func = click.option(
        "--gcp-location",
        type=str,
        default="us-central1",
        envvar="GCP_LOCATION",
        help="GCP location/region for Vertex AI.",
    )(func)

    func = click.option(
        "--gcp-credentials-path",
        type=str,
        default=None,
        envvar="GOOGLE_APPLICATION_CREDENTIALS",
        help="Path to GCP service account JSON file.",
    )(func)

    return func


# OCI auth related options
def oci_auth_options(func):
    """OCI authentication options.

    Supports authentication methods:
    1. user_principal - Default authentication using config file (default)
    2. instance_principal - Used when running on OCI instances
    3. security_token - Using security token for delegation
    4. instance_obo_user - Instance principal with user delegation
    """
    func = click.option(
        "--auth",
        type=click.Choice(
            [
                "user_principal",
                "instance_principal",
                "security_token",
                "instance_obo_user",
            ]
        ),
        default="user_principal",
        help="OCI authentication type to use",
    )(func)

    func = click.option(
        "--config-file",
        type=str,
        default=oci.config.DEFAULT_LOCATION,
        help="Path to OCI config file for user_principal authentication",
    )(func)

    func = click.option(
        "--profile",
        type=str,
        default=oci.config.DEFAULT_PROFILE,
        help="OCI config profile name for user_principal authentication",
    )(func)

    func = click.option(
        "--security-token",
        type=str,
        default=None,
        help="Security token for security_token authentication",
    )(func)

    func = click.option(
        "--region",
        type=str,
        default=None,
        help="OCI region for security_token authentication",
    )(func)

    return func


# Group sampling-related options
def sampling_options(func):
    func = click.option(
        "--dataset-config",
        type=click.Path(exists=True),
        default=None,
        callback=validate_dataset_config,
        help="Path to JSON configuration file for advanced dataset options. "
        "This allows full control over dataset loading parameters.",
    )(func)
    func = click.option(
        "--dataset-image-column",
        type=str,
        default=None,
        help="Column name containing images (for multimodal datasets).",
    )(func)
    func = click.option(
        "--dataset-prompt-column",
        type=str,
        default=None,
        help="Column name containing prompts (for CSV/HuggingFace datasets). "
        "If not specified, empty prompts will be used. For advanced usage, please"
        "check out DatasetConfig.",
    )(func)
    func = click.option(
        "--dataset-path",
        type=str,
        callback=validate_dataset_path_callback,
        default=None,
        help="Dataset source: local file path, HuggingFace ID, or 'default' for "
        "built-in sonnet.txt. Examples:\n"
        "- Local file: /path/to/data.csv or /path/to/prompts.txt\n"
        "- HuggingFace: squad or meta-llama/Llama-2-7b-hf\n"
        "- Default: Leave empty to use built-in sonnet.txt",
    )(func)
    return func


# Group server-related options
def server_options(func):
    """
    Server-related options used for experiment filtering and analysis.
    These options allow users to specify GPU count, type, server version,
    and backend engine.
    """
    func = click.option(
        "--server-gpu-count",
        type=str,
        required=False,
        default=None,
        help="The number of the GPU cards serving the backend model server.",
    )(func)
    func = click.option(
        "--server-gpu-type",
        type=click.Choice(
            ["H100", "A100-80G", "A100-40G", "MI300", "A10", "H200", "B200"],
            case_sensitive=True,
        ),
        required=False,
        default=None,
        help="The type of the GPU node serving the backend model server.",
    )(func)
    func = click.option(
        "--server-version",
        type=str,
        required=False,
        default=None,
        help="The version of the backend model server.",
    )(func)
    func = click.option(
        "--server-engine",
        type=click.Choice(
            ["vLLM", "SGLang", "TGI", "cohere-TensorRT", "cohere-vLLM", "LlamaCPP"],
            case_sensitive=True,
        ),
        required=False,
        default=None,
        help="The backend engine of the backend model server.",
    )(func)
    return func


# Group experiment-related options
def experiment_options(func):
    func = click.option(
        "--experiment-folder-name",
        type=str,
        default=None,
        help="The name of the folder to save the experiment results. "
        "Defaults to a generated name in the format: "
        "<api_backend>_<server_engine>_<server_version>_<task>_<model>_<timestamp>. "  # noqa: E501
        "If server_engine or server_version is not provided, it will be "
        "omitted from the name.",
    )(func)
    func = click.option(
        "--experiment-base-dir",
        type=str,
        default=None,
        help="Base local directory for storing experiment results. "
        "It can be either absolute or relative.",
    )(func)
    func = click.option(
        "--log-dir",
        type=str,
        default=None,
        help=(
            "Directory for storing log files "
            "(genai_bench.log and genai_bench_worker_*.log). "
            "If not specified, logs will be written to the current working directory."
        ),
    )(func)
    func = click.option(
        "--max-requests-per-run",
        type=int,
        required=True,
        prompt=True,
        help="The maximum number of requests per experiment run. "
        "One experiment run will exit if max_requests_per_run is "
        "reached. Recommendation: at least 1 * max(`num_concurrency`). ",
    )(func)
    func = click.option(
        "--max-time-per-run",
        type=int,
        required=True,
        prompt=True,
        help="The max duration per experiment run. Unit: minute. "
        "One experiment run will exit if max_time_per_run is "
        "reached. ",
    )(func)
    func = click.option(
        "--warmup-ratio",
        type=click.FloatRange(0.0, 1.0, min_open=False, max_open=True),
        default=None,
        help=(
            "The ratio of the benchmark to run as a warmup "
            "and not include in the final results."
        ),
    )(func)
    func = click.option(
        "--cooldown-ratio",
        type=click.FloatRange(0.0, 1.0, min_open=False, max_open=True),
        default=None,
        callback=validate_warmup_cooldown_ratio_options,
        help=(
            "The ratio of the benchmark to run as a cooldown "
            "and not include in the final results."
        ),
    )(func)
    func = click.option(
        "--traffic-scenario",
        type=str,
        multiple=True,
        default=None,
        callback=validate_traffic_scenario_callback,
        help="""
                A list of scenario strings defining the scenarios for sampling input and output tokens.

                Supported distributions are:

                \b
                1. **Deterministic (D)**: Constant token count.
                   - Format: D(num_input_tokens,num_output_tokens)
                   - Example: D(100,200)

                \b
                2. **Normal (N)**: Normally distributed token counts.
                   - Format: N(mean_input_tokens,stddev_input_tokens)/
                   (mean_output_tokens,stddev_output_tokens)
                   - Example: N(100,20)/(200,30)

                \b
                3. **Uniform (U)**: Uniformly distributed token counts.
                   - Formats: U(min_input_tokens,max_input_tokens)/
                   (min_output_tokens,max_output_tokens)
                              U(max_input_tokens,max_output_tokens)
                   - Examples: U(50,100)/(200,250)
                               U(100,200)

                \b
                Supported modalities are:

                \b
                1. **Image (I)**: Image, i.e. JPEG as input.
                   - Format: I(num_input_dimension_width,num_input_dimension_height)
                   - Example: I(256,256)

                \b
                Supported embeddings are:

                \b
                1.  **Embeddings (E)**: Document as input.
                   - Format: E(max_tokens_per_document)
                   - Examples: E(1024)

                Supported scenarios are:

                \b
                1.  **ReRank (R)**: Documents and query is input.
                   - Format: R(max_tokens_per_document, max_tokens_per_query)
                   - Examples: R(1024, 128)

                \b
                Default scenarios vary by task:
                For chat:
                1. Fusion: N(480,240)/(300,150)
                2. Chatbot/Dialog: D(100,100)
                3. Generation Heavy: D(100,1000)
                4. Typical RAG: D(2000,200)
                5. Heavier RAG: D(7800,200)

                \b
                For embeddings:
                1. E(64)
                2. E(128)
                3. E(256)
                4. E(512)
                5. E(1024)

                \b
                For re-rank:
                1. R(64,100)
                2. R(128,100)
                3. R(256,100)
                4. R(512,100)
                5. R(1024,100)
                6. R(2048,100)
                7. R(4096,100)

                \b
                For image multi-modality:
                1. I(512,512)
                2. I(1024,512)
                3. I(2048,2048)

                \b
                Example to input multiple values:
                --traffic-scenario "N(480,240)/(300,150)" \\
                --traffic-scenario "D(16000,100)" \\
                --traffic-scenario "D(7800,200)"
            """,  # noqa: E501
    )(func)
    func = click.option(
        "--batch-size",
        type=click.INT,
        multiple=True,
        is_eager=True,
        default=DEFAULT_BATCH_SIZES,
        help="""
            List of batch sizes to use for tasks. Only supported for
            text-to-embeddings or text-to-rerank tasks for now. Default batch sizes
            [1, 2, 4, 8, 24, 48, 96].

            \b
            Example:
            --batch-size 1 --batch-size 8 \\
            --batch-size 16 --batch-size 32
            """,
    )(func)
    func = click.option(
        "--num-concurrency",
        type=click.INT,
        multiple=True,
        is_eager=True,
        default=DEFAULT_NUM_CONCURRENCIES,
        help="""
                List of concurrency levels to run the experiment with.

                \b
                Example to input multiple values:
                --num-concurrency 1 --num-concurrency 2 \\
                --num-concurrency 4 --num-concurrency 8 \\
                --num-concurrency 16 --num-concurrency 32
             """,
    )(func)
    func = click.option(
        "--iteration-type",
        type=click.Choice(["num_concurrency", "batch_size"], case_sensitive=False),
        default="num_concurrency",
        callback=validate_iteration_params,
        help="Type of iteration to use for the experiment. "
        "Note: batch_size is auto-selected for text-to-embeddings tasks, "
        "num_concurrency for others.",
    )(func)
    func = click.option(
        "--model",
        type=str,
        default=None,
        callback=set_model_from_tokenizer,
        help="The model to benchmark. If not set, will be parsed from "
        "--model-tokenizer.",
    )(func)
    func = click.option(
        "--model-tokenizer",
        type=str,
        required=True,
        prompt=True,
        help="The tokenizer to use. Should be a Huggingface loadable tokenizer "
        "or a local path. IMPORTANT: it should match the tokenizer the model "
        "server uses.",
    )(func)
    func = click.option(
        "--metrics-time-unit",
        type=click.Choice(["s", "ms"], case_sensitive=False),
        default="s",
        help="Time unit for latency metrics display and export. "
        "Options: 's' (seconds), 'ms' (milliseconds). Default: s",
    )(func)
    return func


def distributed_locust_options(func):
    func = click.option(
        "--master-port",
        type=int,
        default=5557,
        required=False,
        help="The port for the master process when running with multiple workers.",
    )(func)
    func = click.option(
        "--num-workers",
        type=int,
        default=0,
        required=False,
        help="Number of worker processes to spawn for the load test. "
        "By default it runs as a single process.",
    )(func)
    func = click.option(
        "--spawn-rate",
        type=int,
        default=None,
        required=False,
        help="Number of users to spawn per second. Defaults to concurrency value. "
        "Use lower values (e.g., 10-50) for LLM workloads to prevent worker overload.",
    )(func)
    return func


# Storage provider authentication options
def storage_auth_options(func):
    """Storage provider authentication options.

    Supports various storage providers:
    - OCI Object Storage
    - AWS S3
    - Azure Blob Storage
    - GCP Cloud Storage
    - GitHub Releases/Artifacts
    """
    # Storage provider selection
    func = click.option(
        "--storage-provider",
        type=click.Choice(
            ["oci", "aws", "azure", "gcp", "github"], case_sensitive=False
        ),
        default="oci",
        help="Storage provider for uploading results. "
        "Defaults to OCI for backward compatibility.",
    )(func)

    # Generic storage options
    func = click.option(
        "--storage-bucket",
        type=str,
        is_eager=True,
        help="Bucket/container name for storage provider.",
    )(func)

    func = click.option(
        "--storage-prefix",
        type=str,
        default="",
        help="Prefix for uploaded objects in the bucket.",
    )(func)

    # Storage auth type
    func = click.option(
        "--storage-auth-type",
        type=str,
        default=None,
        help="Authentication type for storage provider. "
        "Options depend on --storage-provider.",
    )(func)

    # AWS S3 storage auth options
    func = click.option(
        "--storage-aws-access-key-id",
        type=str,
        default=None,
        envvar="AWS_ACCESS_KEY_ID",
        help="AWS access key ID for S3 storage.",
    )(func)

    func = click.option(
        "--storage-aws-secret-access-key",
        type=str,
        default=None,
        envvar="AWS_SECRET_ACCESS_KEY",
        help="AWS secret access key for S3 storage.",
    )(func)

    func = click.option(
        "--storage-aws-session-token",
        type=str,
        default=None,
        envvar="AWS_SESSION_TOKEN",
        help="AWS session token for S3 storage.",
    )(func)

    func = click.option(
        "--storage-aws-region",
        type=str,
        default=None,
        envvar="AWS_DEFAULT_REGION",
        help="AWS region for S3 storage.",
    )(func)

    func = click.option(
        "--storage-aws-profile",
        type=str,
        default=None,
        envvar="AWS_PROFILE",
        help="AWS profile for S3 storage.",
    )(func)

    # Azure Blob storage auth options
    func = click.option(
        "--storage-azure-account-name",
        type=str,
        default=None,
        envvar="AZURE_STORAGE_ACCOUNT_NAME",
        help="Azure storage account name.",
    )(func)

    func = click.option(
        "--storage-azure-account-key",
        type=str,
        default=None,
        envvar="AZURE_STORAGE_ACCOUNT_KEY",
        help="Azure storage account key.",
    )(func)

    func = click.option(
        "--storage-azure-connection-string",
        type=str,
        default=None,
        envvar="AZURE_STORAGE_CONNECTION_STRING",
        help="Azure storage connection string (overrides account name/key).",
    )(func)

    func = click.option(
        "--storage-azure-sas-token",
        type=str,
        default=None,
        envvar="AZURE_STORAGE_SAS_TOKEN",
        help="Azure storage SAS token.",
    )(func)

    # GCP Cloud Storage auth options
    func = click.option(
        "--storage-gcp-project-id",
        type=str,
        default=None,
        envvar="GCP_PROJECT_ID",
        help="GCP project ID for Cloud Storage.",
    )(func)

    func = click.option(
        "--storage-gcp-credentials-path",
        type=str,
        default=None,
        envvar="GOOGLE_APPLICATION_CREDENTIALS",
        help="Path to GCP service account JSON for Cloud Storage.",
    )(func)

    # GitHub storage options
    func = click.option(
        "--github-token",
        type=str,
        default=None,
        envvar="GITHUB_TOKEN",
        help="GitHub personal access token.",
    )(func)

    func = click.option(
        "--github-owner",
        type=str,
        default=None,
        envvar="GITHUB_OWNER",
        help="GitHub repository owner (user or org).",
    )(func)

    func = click.option(
        "--github-repo",
        type=str,
        default=None,
        envvar="GITHUB_REPO",
        help="GitHub repository name.",
    )(func)

    return func


# Group object storage upload options
def object_storage_options(func):
    """Object Storage related options for uploading benchmark results."""
    func = click.option(
        "--namespace",
        type=str,
        help="OCI Object Storage namespace (OCI-specific)",
    )(func)
    func = click.option(
        "--upload-results",
        is_flag=True,
        default=False,
        callback=validate_object_storage_options,
        help="Whether to upload benchmark results to storage",
    )(func)
    return func


# SGLang profiling options
def sglang_profile_options(func):
    """SGLang server-side profiling options.

    These options enable Perfetto trace profiling on SGLang servers,
    capturing GPU kernel-level performance data for detailed analysis.
    Only applicable when --api-backend is 'sglang'.
    """
    func = click.option(
        "--sglang-profile",
        is_flag=True,
        default=False,
        help="Enable SGLang server-side profiling. Generates Perfetto traces "
        "for GPU kernel-level performance analysis. Only works with "
        "--api-backend sglang.",
    )(func)
    func = click.option(
        "--sglang-profile-output-dir",
        type=str,
        default=None,
        help="Directory to save SGLang profile traces. "
        "Defaults to '<experiment-folder>/profiles'.",
    )(func)
    func = click.option(
        "--sglang-profile-steps",
        type=int,
        default=5,
        help="Number of forward steps to profile. More steps = more data "
        "but larger trace files. Default: 5 (matches SGLang nightly tests).",
    )(func)
    func = click.option(
        "--sglang-profile-by-stage",
        is_flag=True,
        default=True,
        help="Profile prefill and decode stages separately. "
        "Enables detailed analysis of each inference phase. Default: True.",
    )(func)
    func = click.option(
        "--sglang-profile-activities",
        type=str,
        default="CPU,GPU",
        help="Comma-separated list of activities to profile. "
        "Options: CPU, GPU, MEM. Default: 'CPU,GPU'.",
    )(func)
    return func
