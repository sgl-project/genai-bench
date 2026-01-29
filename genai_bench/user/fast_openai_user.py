"""Customized user for OpenAI backends using FastHttpUser."""

from locust import task

import json
import random
import time
from typing import Any, Callable, Dict, Optional

from requests import Response

from genai_bench.auth.model_auth_provider import ModelAuthProvider
from genai_bench.logging import init_logger
from genai_bench.protocol import (
    UserChatRequest,
    UserChatResponse,
    UserEmbeddingRequest,
    UserImageChatRequest,
    UserResponse,
)
from genai_bench.user.fast_http_user import BaseFastUser

logger = init_logger(__name__)


class FastOpenAIUser(BaseFastUser):
    BACKEND_NAME = "fast-openai"
    supported_tasks = {
        "text-to-text": "chat",
        "image-text-to-text": "chat",
        "text-to-embeddings": "embeddings",
    }

    host: Optional[str] = None
    auth_provider: Optional[ModelAuthProvider] = None
    headers = None

    def on_start(self):
        if not self.host or not self.auth_provider:
            raise ValueError("API key and base must be set for FastOpenAIUser.")
        auth_headers = self.auth_provider.get_headers()
        self.headers = {
            **auth_headers,
            "Content-Type": "application/json",
        }
        self.api_backend = getattr(self, "api_backend", self.BACKEND_NAME)
        super().on_start()

    @task
    def chat(self):
        endpoint = "/v1/chat/completions"
        user_request = self.sample()

        if not isinstance(user_request, UserChatRequest):
            raise AttributeError(
                f"user_request should be of type "
                f"UserChatRequest for FastOpenAIUser.chat, got "
                f"{type(user_request)}"
            )

        if isinstance(user_request, UserImageChatRequest):
            text_content = [{"type": "text", "text": user_request.prompt}]
            image_content = [
                {
                    "type": "image_url",
                    "image_url": {"url": image},
                }
                for image in user_request.image_content
            ]
            content = text_content + image_content
        else:
            # Backward compatibility for vLLM versions prior to v0.5.1.
            # OpenAI API used a different text prompt format before
            # multi-modality model support.
            content = user_request.prompt

        payload = {
            "model": user_request.model,
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
            "max_tokens": user_request.max_tokens,
            "temperature": user_request.additional_request_params.get(
                "temperature", 0.0
            ),
            "stream": True,
            "stream_options": {
                "include_usage": True,
            },
            **user_request.additional_request_params,
        }

        # Conditionally add ignore_eos for vLLM and SGLang backends
        if self.api_backend in ["vllm", "sglang"]:
            payload.setdefault("ignore_eos", bool(user_request.max_tokens))
        else:
            # Remove ignore_eos for OpenAI backend, as it is not supported
            payload.pop("ignore_eos", None)

        self.send_request(
            True,
            endpoint,
            payload,
            self.parse_chat_response,
            user_request.num_prefill_tokens,
        )

    @task
    def embeddings(self):
        endpoint = "/v1/embeddings"

        user_request = self.sample()

        if not isinstance(user_request, UserEmbeddingRequest):
            raise AttributeError(
                f"user_request should be of type "
                f"UserEmbeddingRequest for FastOpenAIUser."
                f"embeddings, got {type(user_request)}"
            )

        random.shuffle(user_request.documents)
        payload = {
            "model": user_request.model,
            "input": user_request.documents,
            "encoding_format": user_request.additional_request_params.get(
                "encoding_format", "float"
            ),
            **user_request.additional_request_params,
        }
        self.send_request(False, endpoint, payload, self.parse_embedding_response)

    def send_request(
        self,
        stream: bool,
        endpoint: str,
        payload: Dict[str, Any],
        parse_strategy: Callable[..., UserResponse],
        num_prefill_tokens: Optional[int] = None,
    ) -> UserResponse:
        """
        Sends a POST request, handling both streaming and non-streaming
        responses.

        Args:
            endpoint (str): The API endpoint.
            payload (Dict[str, Any]): The JSON payload for the request.
            stream (bool): Whether to stream the response.
            parse_strategy (Callable[[Response, float], UserResponse]):
                The function to parse the response.
            num_prefill_tokens (Optional[int]): The num of tokens in the
                prefill/prompt. Only need for streaming requests.

        Returns:
            UserResponse: A response object containing status and metrics data.
        """
        response = None

        try:
            # TTFT Measurement Timeline:
            # T0: start_time - Request initiation timestamp
            # T1: POST request sent over network
            # T2: non_stream_post_end_time - HTTP response headers received
            # T3: Iteration begins (for streaming requests)
            # T4: time_at_first_token - First content chunk arrives (streaming only)
            #
            # For streaming requests (chat):
            #   TTFT = time_at_first_token - start_time
            #   (measured in parse_chat_response)
            # For non-streaming requests (embeddings):
            #   TTFT = non_stream_post_end_time - start_time
            #   (end_time = TTFT)
            start_time = time.monotonic()

            # Use self.client.post instead of requests.post to
            # leverage FastHttpUser's connection pooling
            # Use data=json.dumps(payload) to ensure correct serialization
            # and avoid potential double-header issues
            with self.client.post(
                url=f"{self.host}{endpoint}",
                data=json.dumps(payload),
                stream=stream,
                headers=self.headers,
                # Required for FastHttpUser to handle response manually
                catch_response=True,
                name=endpoint,
            ) as response:
                # Timestamp when HTTP response headers are received.
                # For non-streaming requests: this is the end_time.
                # For streaming requests: this is passed to parse_strategy
                # but unused, as TTFT is measured when the first content
                # chunk arrives during iteration.
                non_stream_post_end_time = time.monotonic()

                if response.status_code == 200:
                    metrics_response = parse_strategy(
                        response,
                        start_time,
                        num_prefill_tokens,
                        non_stream_post_end_time,
                    )
                    # Check if stream contains errors (e.g., data.get("error"))
                    if metrics_response.status_code == 200:
                        response.success()
                    else:
                        response.failure(
                            f"Stream error: {metrics_response.error_message}"
                        )
                else:
                    metrics_response = UserResponse(
                        status_code=response.status_code,
                        error_message=response.text,
                    )
                    response.failure(f"HTTP {response.status_code}: {response.text}")
        except Exception as e:
            # FastHttpUser raises different exceptions,
            # catch generic Exception for simplicity
            # or import specific exceptions from geventhttpclient if needed
            metrics_response = UserResponse(
                status_code=500, error_message=f"Request failed: {str(e)}"
            )
        finally:
            # FastHttpUser context manager handles closing
            # but explicit close doesn't hurt
            pass

        # FastHttpUser already emits Locust request events; suppress duplicate fire
        self.collect_metrics(metrics_response, endpoint, fire_event=False)
        return metrics_response

    def parse_chat_response(
        self,
        response: Response,
        start_time: float,
        num_prefill_tokens: int,
        _: float,
    ) -> UserResponse:
        """
        Parses a streaming chat response.

        Interface Design Note:
            This method shares the same signature as parse_embedding_response()
            to support the Strategy Pattern in send_request(). The 4th parameter
            (non_stream_post_end_time) is intentionally unused here because:

            - For streaming requests, TTFT must be measured when the first
              content chunk arrives during iteration (see line ~332)
            - The end_time is also measured after iteration completes
            - non_stream_post_end_time is captured before iteration starts,
              so it cannot represent either TTFT or end_time for streaming

            This design allows send_request() to uniformly call all parse
            strategies without needing to know their specific implementations.

        Args:
            response (Response): The streaming response object to iterate.
            start_time (float): Request initiation timestamp (T0).
            num_prefill_tokens (int): The num of tokens in the prefill/prompt.
            _ (float): Unused. Kept for interface compatibility with
                parse_embedding_response().

        Returns:
            UserChatResponse: Response with dynamically measured TTFT and end_time.
        """
        stream_chunk_prefix = "data: "

        generated_text = ""
        tokens_received = 0
        time_at_first_token = None
        finish_reason = None
        previous_data = None
        num_prompt_tokens = None

        # FastHttpUser response object is different from requests.Response
        # It doesn't have iter_lines(), so we need to implement line reading manually
        # or use response.content if not streaming (but here we are streaming)

        buffer = ""
        # Iterate over chunks from the response
        for chunk_bytes in response:
            if not chunk_bytes:
                continue

            # Decode bytes to string and add to buffer
            chunk_str = chunk_bytes.decode("utf-8", errors="ignore")
            buffer += chunk_str

            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()

                if not line:
                    continue

                if line.startswith(stream_chunk_prefix):
                    chunk_data = line[len(stream_chunk_prefix) :]
                else:
                    # Handle cases where prefix might be missing or different format
                    chunk_data = line

                if chunk_data == "[DONE]":  # Check string "[DONE]" not bytes b"[DONE]"
                    break

                try:
                    data = json.loads(chunk_data)
                except json.JSONDecodeError:
                    continue

                # Handle streaming error response as OpenAI API server handles it
                # differently. Some might return 200 first and generate error response
                # later in the chunk
                if data.get("error") is not None:
                    return UserResponse(
                        status_code=data["error"].get("code", -1),
                        error_message=data["error"].get(
                            "message", "Unknown error, please check server logs"
                        ),
                    )

                # Standard OpenAI API streams include "finish_reason"
                # in the second-to-last chunk,
                # followed by "usage" in the final chunk,
                # which does not contain "finish_reason"
                if (
                    not data["choices"]
                    and finish_reason
                    and "usage" in data
                    and data["usage"]
                ):
                    num_prefill_tokens, num_prompt_tokens, tokens_received = (
                        self._get_usage_info(data, num_prefill_tokens)
                    )
                    # Additional check for time_at_first_token when the response is
                    # too short
                    if not time_at_first_token:
                        tokens_received = data["usage"].get("completion_tokens", 0)
                        if tokens_received > 1:
                            logger.warning(
                                f"🚨🚨🚨 The first chunk the server returned "
                                f"has >1 tokens: {tokens_received}. It will "
                                f"affect the accuracy of time_at_first_token!"
                            )
                            time_at_first_token = time.monotonic()
                        # else:
                        #     raise Exception("Invalid Response") # Relaxed check
                    break

                try:
                    delta = data["choices"][0]["delta"]
                    content = delta.get("content") or delta.get("reasoning_content")
                    usage = delta.get("usage")

                    if usage:
                        tokens_received = usage["completion_tokens"]
                    if content:
                        if not time_at_first_token:
                            if tokens_received > 1:
                                logger.warning(
                                    f"🚨🚨🚨 The first chunk the server returned "
                                    f"has >1 tokens: {tokens_received}. It will "
                                    f"affect the accuracy of time_at_first_token!"
                                )
                            time_at_first_token = time.monotonic()
                        generated_text += content

                    finish_reason = data["choices"][0].get("finish_reason", None)

                    # SGLang v0.4.3 to v0.4.7 has finish_reason and usage
                    # in the last chunk
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
                "🚨🚨🚨 There is no usage info returned from the model "
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
        response: Response, start_time: float, _: Optional[int], end_time: float
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
