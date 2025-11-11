"""Azure OpenAI user implementation."""

from locust import task

import json
import time
from typing import Any, Callable, Dict, List, Optional

import httpx

from genai_bench.auth.auth_provider import AuthProvider
from genai_bench.logging import init_logger
from genai_bench.protocol import (
    UserChatRequest,
    UserChatResponse,
    UserEmbeddingRequest,
    UserImageChatRequest,
    UserResponse,
)
from genai_bench.user.base_user import BaseUser

logger = init_logger(__name__)


class AzureOpenAIUser(BaseUser):
    """Azure OpenAI API user implementation."""

    BACKEND_NAME = "azure-openai"

    # Task mapping
    supported_tasks = {
        "text-to-text": "chat",
        "text-to-embeddings": "embeddings",
        "image-text-to-text": "chat",  # Same method handles both text and image
    }

    host: Optional[str] = None
    auth_provider: Optional[AuthProvider] = None
    api_version: Optional[str] = None
    deployment_name: Optional[str] = None
    headers: Dict[str, str] = {}
    _http_client: Optional[httpx.Client] = None

    def on_start(self):
        """Initialize Azure OpenAI client using auth provider."""
        if not self.auth_provider:
            raise ValueError("Auth provider not set for Azure OpenAI")

        # Get auth headers
        self.headers = self.auth_provider.get_headers()
        self.headers["Content-Type"] = "application/json"

        # Get config
        config = self.auth_provider.get_config()
        self.api_version = config.get("api_version", "2024-02-01")

        # Azure deployment name should come from the model field
        # but can be overridden in config
        self.deployment_name = config.get("azure_deployment", None)

        # Update host if provided in config
        if config.get("azure_endpoint"):
            self.host = config["azure_endpoint"].rstrip("/")

        # Initialize httpx client for proper streaming support
        self._http_client = httpx.Client(timeout=None)

        super().on_start()

    def on_stop(self):
        """Cleanup httpx client when stopping"""
        if self._http_client:
            self._http_client.close()
        super().on_stop()

    @task
    def chat(self):
        """Perform a chat request to Azure OpenAI."""
        # Get request using sample method
        user_request = self.sample()

        if not isinstance(user_request, UserChatRequest):
            raise AttributeError(
                "user_request should be of type UserChatRequest for "
                f"AzureOpenAIUser.chat, got {type(user_request)}"
            )

        # Use deployment name from config or fall back to model name
        deployment = self.deployment_name or user_request.model

        # Build URL
        endpoint = (
            f"/openai/deployments/{deployment}/chat/completions?"
            f"api-version={self.api_version}"
        )

        # Prepare request body
        request_body = self._prepare_chat_request(user_request)

        # Send request
        self.send_request(
            user_request.additional_request_params.get("stream", True),
            endpoint,
            request_body,
            self.parse_chat_response,
            user_request.num_prefill_tokens,
        )

    @task
    def embeddings(self):
        """Perform an embeddings request to Azure OpenAI."""
        # Get request using sample method
        user_request = self.sample()

        if not isinstance(user_request, UserEmbeddingRequest):
            raise AttributeError(
                "user_request should be of type UserEmbeddingRequest for "
                f"AzureOpenAIUser.embeddings, got {type(user_request)}"
            )

        # Use deployment name from config or fall back to model name
        deployment = self.deployment_name or user_request.model

        # Build URL
        endpoint = (
            f"/openai/deployments/{deployment}/embeddings?"
            f"api-version={self.api_version}"
        )

        # Prepare request body
        request_body = {
            "input": user_request.documents,
            "encoding_format": user_request.additional_request_params.get(
                "encoding_format", "float"
            ),
            **user_request.additional_request_params,
        }

        # Send request
        self.send_request(
            False,
            endpoint,
            request_body,
            self.parse_embedding_response,
        )

    def send_request(
        self,
        stream: bool,
        endpoint: str,
        payload: Dict[str, Any],
        parse_strategy: Callable[..., UserResponse],
        num_prefill_tokens: Optional[int] = None,
    ) -> UserResponse:
        """
        Sends a POST request using httpx, handling both streaming and
        non-streaming responses.

        Args:
            endpoint (str): The API endpoint.
            payload (Dict[str, Any]): The JSON payload for the request.
            stream (bool): Whether to stream the response.
            parse_strategy (Callable): The function to parse the response.
            num_prefill_tokens (Optional[int]): The num of tokens in the
                prefill/prompt. Only need for streaming requests.

        Returns:
            UserResponse: A response object containing status and metrics data.
        """
        try:
            start_time = time.monotonic()

            if stream:
                # Use streaming for chat completions
                with self._http_client.stream(
                    "POST",
                    f"{self.host}{endpoint}",
                    json=payload,
                    headers=self.headers,
                ) as response:
                    non_stream_post_end_time = time.monotonic()

                    if response.status_code == 200:
                        metrics_response = parse_strategy(
                            response,
                            start_time,
                            num_prefill_tokens,
                            non_stream_post_end_time,
                        )
                    else:
                        metrics_response = UserResponse(
                            status_code=response.status_code,
                            error_message=response.text,
                        )
            else:
                # Non-streaming request (embeddings)
                response = self._http_client.post(
                    f"{self.host}{endpoint}",
                    json=payload,
                    headers=self.headers,
                )
                non_stream_post_end_time = time.monotonic()

                if response.status_code == 200:
                    metrics_response = parse_strategy(
                        response,
                        start_time,
                        num_prefill_tokens,
                        non_stream_post_end_time,
                    )
                else:
                    metrics_response = UserResponse(
                        status_code=response.status_code,
                        error_message=response.text,
                    )

        except httpx.ConnectError as e:
            metrics_response = UserResponse(
                status_code=503, error_message=f"Connection error: {e}"
            )
        except httpx.TimeoutException as e:
            metrics_response = UserResponse(
                status_code=408, error_message=f"Request timed out: {e}"
            )
        except httpx.HTTPError as e:
            metrics_response = UserResponse(
                status_code=500,  # Assign a generic 500
                error_message=str(e),
            )

        self.collect_metrics(metrics_response, endpoint)
        return metrics_response

    def parse_chat_response(
        self,
        response: httpx.Response,
        start_time: float,
        num_prefill_tokens: int,
        _: float,
    ) -> UserResponse:
        """
        Parses a streaming response.

        Args:
            response (Response): The response object.
            start_time (float): The time when the request was started.
            num_prefill_tokens (int): The num of tokens in the prefill/prompt.
            _ (float): Placeholder for an unused var, to keep parse_*_response
                have the same interface.

        Returns:
            UserChatResponse: A response object with metrics and generated text.
        """
        stream_chunk_prefix = "data: "
        end_chunk = b"[DONE]"

        generated_text = ""
        tokens_received = 0
        time_at_first_token = None
        finish_reason = None
        previous_data = None
        num_prompt_tokens = None

        for chunk in response.iter_lines(chunk_size=None):
            # Caution: Adding logs here can make debug mode unusable.
            chunk = chunk.strip()

            if not chunk:
                continue

            chunk = chunk[len(stream_chunk_prefix) :]
            if chunk == end_chunk:
                break
            data = json.loads(chunk)

            # Handle streaming error response
            if data.get("error") is not None:
                return UserResponse(
                    status_code=data["error"].get("code", -1),
                    error_message=data["error"].get(
                        "message", "Unknown error, please check server logs"
                    ),
                )

            # Process the streaming data similar to OpenAI
            if (
                not data["choices"]
                and finish_reason
                and "usage" in data
                and data["usage"]
            ):
                num_prefill_tokens, num_prompt_tokens, tokens_received = (
                    self._get_usage_info(data, num_prefill_tokens)
                )
                break

            try:
                delta = data["choices"][0]["delta"]
                content, usage = (
                    delta.get("content", None),
                    delta.get("usage", None),
                )
                if usage:
                    tokens_received = usage["completion_tokens"]
                if content:
                    if not time_at_first_token:
                        if tokens_received > 1:
                            logger.warning(
                                f"The first chunk the server returned "
                                f"has >1 tokens: {tokens_received}. It will "
                                f"affect the accuracy of time_at_first_token!"
                            )
                        time_at_first_token = time.monotonic()
                    generated_text += content

                finish_reason = data["choices"][0].get("finish_reason", None)

                # Check for usage in the last chunk
                if finish_reason and "usage" in data and data["usage"]:
                    num_prefill_tokens, num_prompt_tokens, tokens_received = (
                        self._get_usage_info(data, num_prefill_tokens)
                    )
                    break

            except (IndexError, KeyError) as e:
                logger.warning(
                    f"Error processing chunk: {e}, data: {data}, "
                    f"previous_data: {previous_data}, "
                    f"finish_reason: {finish_reason}, skipping"
                )

            previous_data = data

        end_time = time.monotonic()
        logger.debug(
            f"Generated text: {generated_text} \n"
            f"Time at first token: {time_at_first_token} \n"
            f"Finish reason: {finish_reason}\n"
            f"Prompt Tokens: {num_prompt_tokens} \n"
            f"Completion Tokens: {tokens_received}\n"
            f"Start Time: {start_time}\n"
            f"End Time: {end_time}"
        )

        if not tokens_received:
            tokens_received = self.environment.sampler.get_token_length(
                generated_text, add_special_tokens=False
            )
            logger.warning(
                "There is no usage info returned from the model "
                "server. Estimated tokens_received based on the model "
                "tokenizer."
            )

        return UserChatResponse(
            status_code=200,
            generated_text=generated_text,
            tokens_received=tokens_received,
            time_at_first_token=time_at_first_token,
            num_prefill_tokens=num_prefill_tokens,
            start_time=start_time,
            end_time=end_time,
        )

    @staticmethod
    def _get_usage_info(data, num_prefill_tokens):
        num_prompt_tokens = data["usage"]["prompt_tokens"]
        tokens_received = data["usage"]["completion_tokens"]
        # For vision task
        if num_prefill_tokens is None:
            # use num_prompt_tokens as prefill to cover image tokens
            num_prefill_tokens = num_prompt_tokens
        if abs(num_prompt_tokens - num_prefill_tokens) >= 50:
            logger.warning(
                f"Significant difference detected in prompt tokens: "
                f"The number of prompt tokens processed by the model "
                f"server ({num_prompt_tokens}) differs from the number "
                f"of prefill tokens returned by the sampler "
                f"({num_prefill_tokens}) by "
                f"{abs(num_prompt_tokens - num_prefill_tokens)} tokens."
            )
        return num_prefill_tokens, num_prompt_tokens, tokens_received

    @staticmethod
    def parse_embedding_response(
        response: httpx.Response, start_time: float, _: Optional[int], end_time: float
    ) -> UserResponse:
        """
        Parses a non-streaming response.

        Args:
            response (Response): The response object.
            start_time (float): The time when the request was started.
            _ (Optional[int]): Placeholder for an unused var, to keep
                parse_*_response have the same interface.
            end_time(float): The time when the request was finished.

        Returns:
            UserResponse: A response object with metrics.
        """
        data = response.json()
        num_prompt_tokens = data["usage"]["prompt_tokens"]

        return UserResponse(
            status_code=200,
            start_time=start_time,
            end_time=end_time,
            time_at_first_token=end_time,
            num_prefill_tokens=num_prompt_tokens,
        )

    def _prepare_chat_request(self, request: UserChatRequest) -> Dict[str, Any]:
        """Prepare chat request body.

        Args:
            request: User request

        Returns:
            Request body dict
        """
        # Build messages
        messages = []

        if isinstance(request, UserImageChatRequest) and request.image_content:
            # Multimodal request
            content: List[Dict[str, Any]] = [{"type": "text", "text": request.prompt}]
            for image in request.image_content:
                content.append({"type": "image_url", "image_url": {"url": image}})
            messages.append({"role": "user", "content": content})
        else:
            # Text-only request
            messages.append({"role": "user", "content": request.prompt})

        # Build request body
        body = {
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.additional_request_params.get("temperature", 0.0),
            "stream": request.additional_request_params.get("stream", True),
            "stream_options": {
                "include_usage": True,
            },
            **request.additional_request_params,
        }

        return body
