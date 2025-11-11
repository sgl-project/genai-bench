from locust import task

import json
import time
from typing import Any, Callable, Dict, Optional

import httpx

from genai_bench.auth.model_auth_provider import ModelAuthProvider
from genai_bench.logging import init_logger
from genai_bench.protocol import (
    UserChatRequest,
    UserChatResponse,
    UserEmbeddingRequest,
    UserImageEmbeddingRequest,
    UserResponse,
)
from genai_bench.user.base_user import BaseUser

logger = init_logger(__name__)


class CohereUser(BaseUser):
    """User class for Cohere model API."""

    BACKEND_NAME = "cohere"
    supported_tasks = {
        "text-to-text": "chat",
        "text-to-embeddings": "embeddings",
        "image-to-embeddings": "embeddings",
    }

    host: Optional[str] = None
    auth_provider: Optional[ModelAuthProvider] = None
    headers = None
    _http_client: Optional[httpx.Client] = None

    def on_start(self):
        """Initialize Cohere client on start."""
        if not self.host or not self.auth_provider:
            raise ValueError("API key and base must be set for CohereUser.")
        # TODO this is not really auth,
        #  we should fix this later by removing it \_(ツ)_/¯
        self.headers = {
            "Authorization": f"Bearer {self.auth_provider.get_credentials()}",
            "Content-Type": "application/json",
        }

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
        """Handles the chat task by sending a streaming request to Cohere's chat API."""
        endpoint = "/v2/chat"
        user_request = self.sample()

        if not isinstance(user_request, UserChatRequest):
            raise AttributeError(
                f"user_request should be of type UserChatRequest for CohereUser.chat, "
                f"got {type(user_request)}"
            )

        payload = {
            "model": user_request.model,
            "messages": [
                {
                    "role": "user",
                    "content": user_request.prompt,
                }
            ],
            "stream": True,
            **user_request.additional_request_params,
        }

        return self.send_request(
            True,
            endpoint,
            payload,
            self.parse_chat_response,
            user_request.num_prefill_tokens,
        )

    @task
    def embeddings(self):
        """Handles the embeddings task by sending a request to Cohere's embed API."""
        endpoint = "/v2/embed"
        user_request = self.sample()

        if not isinstance(user_request, UserEmbeddingRequest):
            raise AttributeError(
                f"user_request should be of type UserEmbeddingRequest for "
                f"CohereUser.embeddings, got {type(user_request)}"
            )

        if user_request.documents and not user_request.num_prefill_tokens:
            logger.warning(
                "Number of prefill tokens is missing or 0. Please double check."
            )

        inputs = self.get_inputs(user_request)

        payload = {
            "model": user_request.model,
            **inputs,
            "embedding_types": user_request.additional_request_params.get(
                "embedding_types", ["float"]
            ),
            "truncate": user_request.additional_request_params.get("truncate", "END"),
        }

        return self.send_request(
            False,
            endpoint,
            payload,
            self.parse_embedding_response,
            user_request.num_prefill_tokens or len(inputs),
        )

    def _process_response(
        self,
        response: httpx.Response,
        start_time: float,
        num_prefill_tokens: Optional[int],
        parse_strategy: Callable[..., UserResponse],
        url: str,
    ) -> UserResponse:
        """
        Process response and generate metrics.

        Args:
            response: The HTTP response object.
            start_time: The time when the request was started.
            num_prefill_tokens: The num of tokens in the prefill/prompt.
            parse_strategy: The function to parse the response.
            url: The request URL for logging.

        Returns:
            UserResponse: A response object containing status and metrics data.
        """
        response.raise_for_status()
        logger.debug(
            f"Request to {url} succeeded with status code " f"{response.status_code}."
        )
        non_stream_post_end_time = time.monotonic()

        if response.status_code == 200:
            return parse_strategy(
                response,
                start_time,
                num_prefill_tokens,
                non_stream_post_end_time,
            )
        else:
            return UserResponse(
                status_code=response.status_code,
                error_message=response.text,
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
        Sends a POST request, handling both streaming and non-streaming responses.

        Args:
            stream (bool): Whether to stream the response.
            endpoint (str): The API endpoint.
            payload (Dict[str, Any]): The JSON payload for the request.
            parse_strategy (Callable[[Response, float], UserResponse]): Function to
                parse response.
            num_prefill_tokens (Optional[int]): Number of tokens in the prefill/prompt.

        Returns:
            UserResponse: A response object containing status and metrics data.
        """
        start_time = time.monotonic()
        url = f"{self.host}{endpoint}"

        try:
            if stream:
                # Use httpx streaming for chat completions
                with self._http_client.stream(
                    "POST",
                    url,
                    json=payload,
                    headers=self.headers,
                ) as response:
                    metrics_response = self._process_response(
                        response, start_time, num_prefill_tokens, parse_strategy, url
                    )
            else:
                # Non-streaming request for embeddings
                response = self._http_client.post(
                    url,
                    json=payload,
                    headers=self.headers,
                )
                try:
                    metrics_response = self._process_response(
                        response, start_time, num_prefill_tokens, parse_strategy, url
                    )
                finally:
                    response.close()
        except httpx.HTTPError as e:
            metrics_response = UserResponse(
                status_code=500,
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
    ) -> UserChatResponse:
        """
        Parses a streaming chat response for Cohere API.

        Args:
            response (httpx.Response): The response object.
            start_time (float): The time when the request was started.
            num_prefill_tokens (int): Number of tokens in the prefill/prompt.
            _(float): The time when the request was finished.

        Returns:
            UserResponse: A response object with metrics.
        """
        generated_text = ""
        tokens_received = 0
        time_at_first_token = None
        usage = None

        for line in response.iter_lines():
            if not line:
                continue

            output_line = line.strip()
            if "data: " not in output_line:
                continue

            if output_line.startswith("event: "):
                # Separate 'event: ...' and 'data: ...'
                parts = output_line.split("\n", 1)
                if len(parts) > 1 and parts[1].startswith("data: "):
                    output_line = parts[1]
                else:
                    continue

            data_str = output_line[len("data: ") :]
            if data_str == "[DONE]":
                break

            try:
                data = json.loads(data_str)

                # Handle content-delta events which contain the actual text
                if data.get("type") == "content-delta":
                    content = (
                        data.get("delta", {})
                        .get("message", {})
                        .get("content", {})
                        .get("text", "")
                    )
                    if content:
                        generated_text += content
                        if not time_at_first_token:
                            time_at_first_token = time.monotonic()

                # Handle message-end event which contains token usage information
                elif data.get("type") == "message-end":
                    usage = data.get("delta", {}).get("usage", {})
                    if usage:
                        tokens_received = usage.get("tokens", {}).get(
                            "output_tokens", 0
                        )
                        num_input_tokens = usage.get("tokens", {}).get(
                            "input_tokens", num_prefill_tokens
                        )
                        if abs(num_input_tokens - num_prefill_tokens) >= 50:
                            logger.warning(
                                f"Significant difference detected in input tokens: "
                                f"The number of input tokens processed by the model "
                                f"server ({num_input_tokens}) differs from the number "
                                f"of prefill tokens returned by the sampler "
                                f"({num_prefill_tokens}) by "
                                f"{abs(num_input_tokens - num_prefill_tokens)} tokens."
                            )

            except json.JSONDecodeError:
                logger.warning(f"Failed to decode JSON line: {output_line}")
                return UserChatResponse(
                    status_code=-1,
                    error_message=f"Failed to decode JSON line: {output_line}",
                )

        end_time = time.monotonic()

        if not tokens_received and usage is None:
            logger.warning(
                "🚨🚨🚨 No usage info returned from the model server. "
                "Estimating tokens_received based on the model tokenizer."
            )
            tokens_received = self.environment.sampler.get_token_length(
                generated_text, add_special_tokens=False
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

    def parse_embedding_response(
        self,
        _: httpx.Response,
        start_time: float,
        num_prefill_tokens: int,
        end_time: float,
    ) -> UserResponse:
        """
        Parses a non-streaming response for Cohere API.

        Args:
            _ (httpx.Response): The response object.
            start_time (float): The time when the request was started.
            num_prefill_tokens (int): Number of tokens in the prefill/prompt.
            end_time(float): The time when the request was finished.

        Returns:
            UserResponse: A response object with metrics.
        """
        return UserResponse(
            status_code=200,
            num_prefill_tokens=num_prefill_tokens,
            start_time=start_time,
            end_time=end_time,
            time_at_first_token=end_time,
        )

    def get_inputs(self, user_request) -> Dict[str, Any]:
        if isinstance(user_request, UserImageEmbeddingRequest):
            num_sampled_images = len(user_request.image_content)
            if num_sampled_images > 1:
                raise ValueError(
                    f"OCI-Cohere Image embedding supports only 1 "
                    f"image but, the value provided in traffic"
                    f"scenario is requesting {num_sampled_images}"
                )
            return {
                "images": user_request.image_content,
                "input_type": "IMAGE",
            }
        return {"texts": user_request.documents, "input_type": "CLUSTERING"}
