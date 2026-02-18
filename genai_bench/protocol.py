from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, conint

LiveMetricsData = Dict[str, List[float] | Dict[str, float]]


class UserRequest(BaseModel):
    """
    A class to encapsulate the basic request details from user tasks.
    """

    model: str = Field(..., description="Model Name")
    additional_request_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional parameters for the request.",
    )


class UserChatRequest(UserRequest):
    """
    A class to encapsulate the details related to chat request tasks.
    """

    prompt: str = Field(..., description="Prompt to send to the LLM API server.")
    num_prefill_tokens: int | None = Field(
        ..., description="Number of tokens in the prompt."
    )
    max_tokens: int | None = Field(
        ..., description="Number of maximum tokens expected in the generation."
    )


class UserImageChatRequest(UserChatRequest):
    """
    Represents a request that combines image and chat modalities, used for tasks
    where image input is processed alongside textual prompts.

    This class is an extension of `UserChatRequest` because vision-based tasks
    (e.g., visual question answering) often use the same endpoint as chat-based
    tasks
    and follow similar input/output structures.
    """

    image_content: List[str] = Field(
        ...,
        description="Base64 encoding image(s) to send to the LLM API server.",
    )
    num_images: int = Field(..., description="Number of images.")


class UserVideoChatRequest(UserChatRequest):
    """
    Represents a request that combines video and chat modalities, used for tasks
    where video input is processed alongside textual prompts.

    This class is an extension of `UserChatRequest` because video-based tasks
    (e.g., video question answering, video understanding) use the same endpoint
    as chat-based tasks and follow similar input/output structures.
    """

    video_content: List[str] = Field(
        ...,
        description="Video URL list (base64 data URLs or HTTP/HTTPS URLs).",
    )


class UserEmbeddingRequest(UserRequest):
    """
    A class to encapsulate the details related to embedding request tasks.
    """

    documents: List[str] = Field(
        ..., description="Documents to send to the LLM API server."
    )
    num_prefill_tokens: Optional[int] = Field(
        None,
        description="Number of tokens in the whole request, should be "
        "equal to num_document * token_per_document.",
    )


class UserReRankRequest(UserRequest):
    """
    A class to encapsulate the details related to rerank request tasks.
    """

    documents: List[str] = Field(..., description="Documents to rerank by the model.")
    query: str = Field(
        ..., description="Query as per which the documents are re-ranked."
    )
    num_prefill_tokens: Optional[int] = Field(
        None,
        description="Number of tokens in the whole request, should be equal to "
        "num_document * (token_per_document + max_tokens_per_query) ",
    )


class UserImageEmbeddingRequest(UserEmbeddingRequest):
    """
    Represents a request that combines image and embedding modalities, used
    for tasks where image input is processed to generate embeddings.

    This class extends `UserEmbeddingRequest` to support vision-based tasks
    (e.g., image-to-embeddings) by including image-specific input attributes
    while maintaining compatibility with embedding-based workflows.
    """

    image_content: List[str] = Field(
        ...,
        description="Base64 encoding image(s) to send to the LLM API server.",
    )
    num_images: int = Field(..., description="Number of images.")


class UserResponse(BaseModel):
    """
    A class to encapsulate the most common response details from user tasks.
    """

    # Common fields for all tasks i.e. Chat, Embedding and Vision
    status_code: int = Field(..., description="The HTTP status code of the response.")
    time_at_first_token: Optional[float] = Field(
        default=None,
        description="The time at which the first token was received.",
    )
    start_time: Optional[float] = Field(
        default=None, description="The time at which the request was initiated."
    )
    end_time: Optional[float] = Field(
        default=None, description="The time at which the request was completed."
    )
    error_message: Optional[str] = Field(
        default=None, description="The error message if the request failed."
    )
    num_prefill_tokens: Optional[int] = Field(
        default=None,
        description=(
            "For chat, it represents the number of prompt tokens."
            "For vision, it only represents the number of text tokens."
            "For embeddings, this would represent the number of tokens across "
            "all documents."
        ),
    )


class UserChatResponse(UserResponse):
    """
    A class to encapsulate the most common response details from chat tasks.
    """

    generated_text: Optional[str] = Field(
        default="", description="The text generated by the API, if any."
    )
    tokens_received: Optional[int] = Field(
        default=0,
        description="The number of tokens received in the response.",
    )
    reasoning_tokens: Optional[int] = Field(
        default=0,
        description="Number of reasoning tokens.",
    )


class APIAuthConfig(BaseModel):
    """Authentication configuration for different API backends."""


class OpenAIAuthConfig(APIAuthConfig):
    """OpenAI Style Api_key based authentication."""

    api_key: Optional[str] = None


class OCIAuthConfig(APIAuthConfig):
    """AuthConfig for OCI Services"""

    config_file: Optional[str] = None
    profile_name: Optional[str] = None


class ExperimentMetadata(BaseModel):
    """A class to encapsulate the metadata of an experiment."""

    cmd: str = Field(..., description="Exact command for the current experiment.")
    benchmark_version: str = Field(
        ..., description="The current version of genai-bench."
    )
    api_backend: str = Field(
        ...,
        description="The API backend to use.",
    )
    auth_config: Dict[str, Any] = Field(
        default={},
        description="Authentication configuration for the API backend.",
    )
    api_model_name: str = Field(
        ..., description="The model to use for the backend server request body."
    )
    server_model_tokenizer: Optional[str] = Field(
        None, description="The tokenizer used for token sampling."
    )
    model: str = Field(..., description="The model to benchmark.")
    task: str = Field(..., description="The task to benchmark.")
    num_concurrency: List[conint(ge=1)] = Field(  # type: ignore[valid-type]
        ..., description="The number of concurrent requests."
    )
    batch_size: Optional[List[int]] = Field(
        None, description="The batch sizes for embedding tasks."
    )
    iteration_type: Literal["num_concurrency", "batch_size"] = Field(
        "num_concurrency", description="Type of iteration used in the experiment."
    )
    traffic_scenario: List[str] = Field(default_factory=list)
    additional_request_params: Dict[str, Any] = Field(default_factory=dict)
    server_engine: Optional[str] = None
    server_version: Optional[str] = None
    server_gpu_type: Optional[str] = None
    server_gpu_count: Optional[str] = None
    max_time_per_run_s: int = Field(
        ...,
        description="The maximum number of seconds to run for each scenario "
        "and num_concurrency.",
    )
    max_requests_per_run: int = Field(
        ...,
        description="The maximum number of requests per experiment run.",
    )
    experiment_folder_name: str = Field(
        ...,
        description="The name of the folder to save the experiment results.",
    )
    metrics_time_unit: str = Field(
        default="s",
        description="Time unit for latency metrics display and export (s or ms).",
    )
    dataset_path: Optional[str] = None
