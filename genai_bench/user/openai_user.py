"""Customized user for OpenAI backends."""

from locust import task

import json
import random
import time
from typing import Any, Callable, Dict, Optional

import requests
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
from genai_bench.user.base_user import BaseUser

logger = init_logger(__name__)


class OpenAIUser(BaseUser):
    BACKEND_NAME = "openai"
    supported_tasks = {
        "text-to-text": "chat",
        "image-text-to-text": "chat",
        "text-to-embeddings": "embeddings",
        # Future support can be added here
    }

    host: Optional[str] = None
    auth_provider: Optional[ModelAuthProvider] = None
    headers = None
    disable_streaming: bool = False

    def on_start(self):
        if not self.host or not self.auth_provider:
            raise ValueError("API key and base must be set for OpenAIUser.")
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
                f"UserChatRequest for OpenAIUser.chat, got "
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
            "stream": not self.disable_streaming,
            **user_request.additional_request_params,
        }

        # Only add stream_options if streaming is enabled
        if not self.disable_streaming:
            payload["stream_options"] = {
                "include_usage": True,
            }

        # Conditionally add ignore_eos for vLLM and SGLang backends
        if self.api_backend in ["vllm", "sglang"]:
            payload.setdefault("ignore_eos", bool(user_request.max_tokens))
        else:
            # Remove ignore_eos for OpenAI/Baseten backends, as it is not supported
            payload.pop("ignore_eos", None)

        if self.disable_streaming:
            self.send_request(
                False,
                endpoint,
                payload,
                self.parse_non_streaming_chat_response,
                user_request.num_prefill_tokens,
            )
        else:
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
                f"UserEmbeddingRequest for OpenAIUser."
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
            start_time = time.monotonic()
            response = requests.post(
                url=f"{self.host}{endpoint}",
                json=payload,
                stream=stream,
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
        except requests.exceptions.ConnectionError as e:
            metrics_response = UserResponse(
                status_code=503, error_message=f"Connection error: {e}"
            )
        except requests.exceptions.Timeout as e:
            metrics_response = UserResponse(
                status_code=408, error_message=f"Request timed out: {e}"
            )
        except requests.exceptions.RequestException as e:
            metrics_response = UserResponse(
                status_code=500,  # Assign a generic 500
                error_message=str(e),
            )
        finally:
            if response is not None:
                response.close()

        self.collect_metrics(metrics_response, endpoint)
        return metrics_response

    def parse_chat_response(
        self,
        response: Response,
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

            # Don't set time_at_first_token here - we'll set it after processing usage

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
                    else:
                        # Use end_time as fallback instead of raising exception
                        # This handles edge cases where response format is unexpected
                        time_at_first_token = time.monotonic()
                        logger.warning(
                            f"⚠️ Response had ≤1 tokens ({tokens_received}) in usage chunk. "
                            f"Using current time as time_at_first_token fallback."
                        )
                break

            try:
                # Skip chunks with empty choices
                if not data["choices"]:
                    # Even if choices are empty, set time_at_first_token on first chunk
                    # to ensure we have a timestamp even if response format is unexpected
                    if not time_at_first_token:
                        time_at_first_token = time.monotonic()
                        logger.warning(
                            f"⚠️ Setting time_at_first_token on chunk with empty choices. "
                            f"This may indicate unusual response format. Chunk data: {data}"
                        )
                    continue

                delta = data["choices"][0]["delta"]
                content = (
                    delta.get("content")
                    or delta.get("reasoning_content")
                    or delta.get("reasoning")
                )
                usage = delta.get("usage")

                if usage:
                    tokens_received = usage["completion_tokens"]

                if not time_at_first_token:
                    if tokens_received > 1:
                        logger.warning(
                            f"🚨🚨🚨 The first chunk the server returned "
                            f"has >1 tokens: {tokens_received}. It will "
                            f"affect the accuracy of time_at_first_token!"
                        )
                    time_at_first_token = time.monotonic()

                if content:
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
                # Even when exceptions occur, try to set time_at_first_token on first chunk
                if not time_at_first_token:
                    time_at_first_token = time.monotonic()
                    logger.warning(
                        f"⚠️ Setting time_at_first_token after exception on first chunk. "
                        f"Exception: {e}, data: {data}"
                    )
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

        # Ensure time_at_first_token is never None (fallback to end_time)
        # This can happen if no content chunks were received or all chunks were skipped
        if time_at_first_token is None:
            time_at_first_token = end_time
            logger.warning(
                f"⚠️ time_at_first_token was None, using end_time ({end_time}) as fallback. "
                f"This may indicate an issue with the streaming response format or that no content chunks were received."
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

    def parse_non_streaming_chat_response(
        self,
        response: Response,
        start_time: float,
        num_prefill_tokens: int,
        _: float,
    ) -> UserResponse:
        """
        Parses a non-streaming chat response.

        Args:
            response (Response): The response object.
            start_time (float): The time when the request was started.
            num_prefill_tokens (int): The num of tokens in the prefill/prompt.
            _ (float): Placeholder for an unused var, to keep parse_*_response
                have the same interface.

        Returns:
            UserChatResponse: A response object with metrics and generated text.
        """
        data = response.json()

        # Handle error response
        if data.get("error") is not None:
            return UserResponse(
                status_code=data["error"].get("code", -1),
                error_message=data["error"].get(
                    "message", "Unknown error, please check server logs"
                ),
            )

        # Extract response content
        generated_text = data["choices"][0]["message"]["content"]
        finish_reason = data["choices"][0].get("finish_reason", None)

        # Get usage information
        num_prefill_tokens, num_prompt_tokens, tokens_received = self._get_usage_info(
            data, num_prefill_tokens
        )

        end_time = time.monotonic()

        # For non-streaming, we can't measure TTFT, so we use a small offset
        # This prevents division by zero in metrics calculation
        time_at_first_token = start_time + 0.001  # 1ms offset

        logger.debug(
            f"Generated text: {generated_text} \n"
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

        # Ensure time_at_first_token is never None (fallback to end_time)
        # This can happen if:
        # 1. No content chunks were received (e.g., only reasoning tokens in unexpected format)
        # 2. All chunks were skipped due to empty choices
        # 3. Response format is unexpected
        if time_at_first_token is None:
            time_at_first_token = end_time
            logger.warning(
                f"⚠️ time_at_first_token was None, using end_time ({end_time}) as fallback. "
                f"tokens_received: {tokens_received}, generated_text length: {len(generated_text)}. "
                f"This may indicate reasoning-only tokens or an unusual response format."
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
