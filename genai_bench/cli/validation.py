import json
import os
from pathlib import Path

import click
from huggingface_hub.utils import HfHubHTTPError
from transformers import AutoTokenizer

from genai_bench.data.config import DatasetConfig
from genai_bench.logging import init_logger
from genai_bench.scenarios.base import Scenario
from genai_bench.user.aws_bedrock_user import AWSBedrockUser
from genai_bench.user.azure_openai_user import AzureOpenAIUser
from genai_bench.user.cohere_user import CohereUser
from genai_bench.user.gcp_vertex_user import GCPVertexUser
from genai_bench.user.oci_cohere_user import OCICohereUser
from genai_bench.user.oci_genai_user import OCIGenAIUser
from genai_bench.user.openai_user import OpenAIUser

logger = init_logger(__name__)

API_BACKEND_USER_MAP = {
    OpenAIUser.BACKEND_NAME: OpenAIUser,
    OCICohereUser.BACKEND_NAME: OCICohereUser,
    OCIGenAIUser.BACKEND_NAME: OCIGenAIUser,
    CohereUser.BACKEND_NAME: CohereUser,
    AWSBedrockUser.BACKEND_NAME: AWSBedrockUser,
    AzureOpenAIUser.BACKEND_NAME: AzureOpenAIUser,
    GCPVertexUser.BACKEND_NAME: GCPVertexUser,
    "vllm": OpenAIUser,  # vLLM uses OpenAI-compatible API
    "sglang": OpenAIUser,  # SGLang uses OpenAI-compatible API
}

DEFAULT_NUM_CONCURRENCIES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 24, 48, 96]

DEFAULT_SCENARIOS_FOR_CHAT = [
    "N(480,240)/(300,150)",
    "D(100,100)",
    "D(100,1000)",
    "D(2000,200)",
    "D(7800,200)",
]


DEFAULT_SCENARIOS_FOR_VISION = [
    "I(512,512)",
    "I(1024,512)",
    "I(2048,2048)",
]

DEFAULT_SCENARIOS_FOR_EMBEDDING = [
    "E(64)",
    "E(128)",
    "E(256)",
    "E(512)",
    "E(1024)",
]

DEFAULT_SCENARIOS_FOR_RERANK = [
    "R(64,100)",
    "R(128,100)",
    "R(256,100)",
    "R(512,100)",
    "R(1024,100)",
    "R(2096,100)",
    "R(4096,100)",
]

DEFAULT_SCENARIOS_BY_TASK = {
    "text-to-text": DEFAULT_SCENARIOS_FOR_CHAT,
    "text-to-rerank": DEFAULT_SCENARIOS_FOR_RERANK,
    "image-text-to-text": DEFAULT_SCENARIOS_FOR_VISION,
    "text-to-embeddings": DEFAULT_SCENARIOS_FOR_EMBEDDING,
    "image-to-embeddings": DEFAULT_SCENARIOS_FOR_VISION,
    # add other tasks and default scenarios as needed
}


# -------------------------
# General Validation Functions
# -------------------------
def validate_dataset_path_callback(ctx, param, value):
    """Validate the dataset path."""
    task = ctx.params.get("task")
    if not task:
        raise click.BadParameter(
            "The '--task' option is required but was not provided."
        )

    input_modality, output_modality = task.split("-to-")
    if "image" in input_modality and value is None:
        # Check if dataset_config is provided as alternative
        dataset_config = ctx.params.get("dataset_config")
        if dataset_config is None:
            raise click.BadParameter(
                '--dataset-path is required when --task includes "image" input '
                "modality and --dataset-config is not provided."
            )
        else:
            logger.warning(
                "Using dataset configuration file for image task. "
                "Ensure your config file specifies the correct image dataset source."
            )
    return value


def validate_dataset_config(ctx, param, value):
    """Validate dataset configuration file."""
    if value is None:
        return None

    try:
        _ = DatasetConfig.from_file(value)
        return value
    except Exception as e:
        raise click.BadParameter(f"Invalid dataset configuration file: {str(e)}") from e


def validate_scenario_callback(scenario):
    """Validate an individual traffic scenario."""
    try:
        Scenario.validate(scenario)
        return scenario
    except ValueError as e:
        raise click.BadParameter(str(e)) from e


def validate_traffic_scenario_callback(ctx, param, value):
    """Validate and assign traffic scenarios based on the task."""
    task = ctx.params.get("task")
    if not task:
        raise click.BadParameter(
            "The '--task' option is required but was not provided."
        )
    if value:
        return [validate_scenario_callback(v) for v in value]
    # If user provided a dataset and no scenario, prefer dataset mode (no data shaping)
    if ctx.params.get("dataset_path") or ctx.params.get("dataset_config"):
        return ["dataset"]
    if task not in DEFAULT_SCENARIOS_BY_TASK:
        raise click.BadParameter(
            f"No default traffic scenarios defined for task '{task}'"
        )

    logger.info(f"Using default traffic scenarios for task {task}")
    return DEFAULT_SCENARIOS_BY_TASK[task]


def validate_tokenizer(model_tokenizer):
    """Validate and load the tokenizer, either locally or from HuggingFace.

    The function now tries an **anonymous** download first to support public
    repositories without requiring an ``HF_TOKEN``. Authentication is only
    enforced when the Hugging Face Hub returns *401* or *403* errors.
    """
    if isinstance(model_tokenizer, str) and Path(model_tokenizer).exists():
        return AutoTokenizer.from_pretrained(model_tokenizer)
    hf_token = os.environ.get("HF_TOKEN")

    try:
        return AutoTokenizer.from_pretrained(model_tokenizer, token=hf_token)
    except (HfHubHTTPError, OSError) as e:
        is_auth_error = False

        if isinstance(e, OSError):
            if (
                "gated repo" in str(e)
                or "private repository" in str(e)
                and "make sure to pass a token" in str(e)
            ):
                is_auth_error = True
        elif isinstance(e, HfHubHTTPError) and e.response is not None:
            is_auth_error = e.response.status_code in {401, 403, 404}

        if is_auth_error and hf_token is None:
            raise click.BadParameter(
                "Hugging Face requires authentication for this tokenizer. "
                "Please export HF_TOKEN with a valid access token and retry."
            ) from e

        # Propagate all other errors unchanged
        raise


def validate_iteration_params(ctx, param, value) -> str:
    """
    Validate and determine iteration parameters based on task type.

    Args:
        ctx: Click context
        param: Click parameter
        value: Current parameter value (iteration_type)

    Returns:
        str: The validated iteration_type
    """
    task = ctx.params.get("task")
    num_concurrency = ctx.params.get("num_concurrency", [])
    batch_size = ctx.params.get("batch_size", [])

    # For text-to-embeddings tasks, always use batch_size iteration
    if task == "text-to-embeddings" or task == "text-to-rerank":
        if value != "batch_size":
            click.echo("Note: Using batch_size iteration for text-to-embeddings task")
        batch_size = batch_size or DEFAULT_BATCH_SIZES
        value = "batch_size"
        num_concurrency = [1]

    # For all other tasks, use num_concurrency iteration
    else:
        if value != "num_concurrency":
            click.echo(f"Note: Using num_concurrency iteration for {task} task")
        num_concurrency = num_concurrency or DEFAULT_NUM_CONCURRENCIES
        value = "num_concurrency"
        batch_size = [1]

    # Update context with validated values
    ctx.params.update(
        {
            "iteration_type": value,
            "batch_size": batch_size,
            "num_concurrency": num_concurrency,
        }
    )

    return value


# -------------------------
# CLI Validation Functions
# -------------------------


def validate_api_backend(ctx, param, value):
    """
    Validate the selected API backend and set the corresponding user class.
    """
    api_backend = value.lower()
    user_class = API_BACKEND_USER_MAP.get(api_backend)
    if not user_class:
        raise click.BadParameter(f"{value} is not a supported API backend.")

    if ctx.obj is None:
        ctx.obj = {}
    ctx.obj["user_class"] = user_class

    return api_backend


def validate_api_key(ctx, param, value):
    """Validate API key based on backend."""
    api_backend = ctx.params.get("api_backend")

    if not api_backend:
        raise click.BadParameter("api_backend must be specified before api_key")

    # Backends that require API key
    api_key_required = [OpenAIUser.BACKEND_NAME, "vllm", "sglang"]

    # Backends that don't use traditional API key
    no_api_key = [
        OCICohereUser.BACKEND_NAME,
        OCIGenAIUser.BACKEND_NAME,
        CohereUser.BACKEND_NAME,
        AWSBedrockUser.BACKEND_NAME,
        GCPVertexUser.BACKEND_NAME,
    ]

    # Azure OpenAI can use API key or Azure AD
    if api_backend in api_key_required:
        if not value:
            raise click.BadParameter(f"API key is required for {api_backend} backend")
    elif api_backend == AzureOpenAIUser.BACKEND_NAME:
        # Azure can use API key or Azure AD - validated in model auth options
        pass
    elif api_backend in no_api_key:
        if value:
            click.echo(
                f"Warning: API key is not used for {api_backend} backend "
                f"as it uses cloud-specific authentication",
                err=True,
            )
        return None

    return value


def validate_task(ctx, param, value):
    """
    Validate the selected task and ensure compatibility with the API backend.
    """
    task = value.lower()

    # Ensure API backend has been validated
    user_class = ctx.obj.get("user_class") if ctx.obj else None
    if not user_class:
        raise click.BadParameter(
            "API backend is not set. Please provide a valid --api-backend "
            "before --task."
        )

    # Check task compatibility
    if not user_class.is_task_supported(task):
        supported_tasks = ", ".join(user_class.supported_tasks.keys())
        raise click.BadParameter(
            f"Task '{task}' is not supported by the selected API backend. "
            f"Supported tasks are: {supported_tasks}."
        )

    # Store the user task function in the context
    ctx.obj["user_task"] = getattr(user_class, user_class.supported_tasks[task])

    return task


def set_model_from_tokenizer(ctx, param, value):
    """
    Infer the model name from the tokenizer if not explicitly provided.
    """
    model_tokenizer = ctx.params.get("model_tokenizer")
    return value or model_tokenizer.split("/")[-1]


def validate_additional_request_params(ctx, param, value):
    """Validate and parse additional request parameters in JSON format."""
    if value is None:
        return {}  # Return an empty dictionary to be parsed in Sampler
    try:
        value = json.loads(value)
        if "temperature" in value and value["temperature"] > 1.5:
            warning_msg = (
                f"You have set temperature {value['temperature']} too high. "
                f"This may cause higher chars_to_token ratio in the metrics "
                f"and result in higher total_chars_per_hour."
            )
            logger.warning(warning_msg)
            click.secho(warning_msg, fg="yellow", bold=True)
            if click.confirm("Do you want to re-enter a temperature?", default=False):
                value["temperature"] = click.prompt(
                    "Please enter a new temperature,",
                    default=0,
                )
        if "ignore_eos" in value and not value["ignore_eos"]:
            warning_msg = (
                "You have set ignore_eos to False. This will cause inaccurate "
                "num_output_tokens because the generation can stop before "
                "reaching the num_output_tokens you passed. Unless you are "
                "doing correctness test, it is recommended to set ignore_eos "
                "to True."
            )
            logger.warning(warning_msg)
            click.secho(warning_msg, fg="yellow", bold=True)
            if click.confirm(
                "Do you want to reset the ignore_eos to True?", default=False
            ):
                value["ignore_eos"] = True

        return value
    except json.JSONDecodeError as e:
        raise click.BadParameter(
            f"Invalid JSON string in `--additional-request-params`: {str(e)}"
        ) from e


def validate_filter_criteria(ctx, param, value):
    """Parse and validate the filter criteria provided by the user."""
    if value is None:
        return {}
    try:
        return json.loads(value)
    except json.JSONDecodeError as e:
        raise click.BadParameter(
            f"Invalid JSON string in `--filter-criteria`: {str(e)}"
        ) from e


def validate_object_storage_options(ctx, param, value):
    """Validate object storage options."""
    if (
        param.name == "upload_results"
        and value
        and not ctx.params.get("storage_bucket")
    ):
        raise click.UsageError(
            "You must provide a storage bucket name (--storage-bucket) "
            "when uploading results."
        )
    return value


def validate_warmup_cooldown_ratio_options(ctx, param, value):
    """Validate warmup and cooldown options."""
    warmup_ratio = ctx.params.get("warmup_ratio", 0.0)
    cooldown_ratio = value or 0.0
    if warmup_ratio + cooldown_ratio >= 1.0:
        raise click.BadParameter(
            f"warmup_ratio({warmup_ratio}) + cooldown_ratio({cooldown_ratio}) "
            f"must be < 1.0.",
            param_hint=["--warmup-ratio", "--cooldown-ratio"],
        )
    return value
