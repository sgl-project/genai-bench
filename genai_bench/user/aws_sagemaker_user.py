"""AWS Bedrock user implementation."""

from locust import task

import json
import time
from typing import Any, Dict, List, Optional

from genai_bench.logging import init_logger
from genai_bench.protocol import (
    UserChatRequest,
    UserChatResponse,
    UserEmbeddingRequest,
    UserResponse,
)
from genai_bench.user.base_user import BaseUser

try:
    from botocore.eventstream import EventStream
except:
    pass

logger = init_logger(__name__)


class AWSSagemakerUser(BaseUser):
    """AWS Bedrock API user implementation."""

    BACKEND_NAME = "aws-sagemaker"

    # Task mapping
    supported_tasks = {
        "text-to-text": "chat",
        "text-to-embeddings": "embeddings",
        "image-text-to-text": "chat",  # Same method handles both text and image
    }

    host: Optional[str] = None
    auth_provider = None
    runtime_client = None

    def on_start(self):
        """Initialize AWS Bedrock client using auth provider credentials."""
        if not self.auth_provider:
            raise ValueError("Auth provider not set for AWS Bedrock")

        try:
            import boto3
            from botocore.config import Config
        except ImportError as e:
            raise ImportError(
                "boto3 is required for AWS Bedrock. "
                "Install it with: pip install boto3"
            ) from e

        # Get credentials from auth provider
        creds = self.auth_provider.get_credentials()

        # Create session with credentials
        session_kwargs = {}
        if creds.get("profile_name"):
            session_kwargs["profile_name"] = creds["profile_name"]
        else:
            if creds.get("aws_access_key_id"):
                session_kwargs["aws_access_key_id"] = creds["aws_access_key_id"]
            if creds.get("aws_secret_access_key"):
                session_kwargs["aws_secret_access_key"] = creds["aws_secret_access_key"]
            if creds.get("aws_session_token"):
                session_kwargs["aws_session_token"] = creds["aws_session_token"]

        session = boto3.Session(**session_kwargs)

        # Create Bedrock runtime client
        region = creds.get("region_name", "us-east-1")
        config = Config(region_name=region)

        self.runtime_client = session.client(
            service_name="sagemaker-runtime", config=config
        )

        logger.info(f"Initialized AWS Bedrock client in region {region}")

    @task
    def chat(self):
        """Perform a chat request to AWS Bedrock."""
        # Get request using sample method
        user_request = self.sample()

        if not isinstance(user_request, UserChatRequest):
            raise AttributeError(
                "user_request should be of type UserChatRequest for "
                f"AWSBedrockUser.chat, got {type(user_request)}"
            )

        # Prepare request body based on model
        model_id = user_request.model
        request_body = self._prepare_request_body(user_request, model_id)

        # Track timing
        start_time = time.monotonic()

        try:
            # Determine if streaming
            use_streaming = user_request.additional_request_params.get(
                "stream", True
            ) and self._supports_streaming(model_id)

            if use_streaming:
                # Streaming request
                response = self.runtime_client.invoke_endpoint_with_response_stream(
                    EndpointName=model_id,
                    Body=json.dumps(request_body),
                    ContentType="application/json",
                    Accept="application/json",
                )

            else:
                # Non-streaming request
                raise ValueError("Not supported")
                response = self.runtime_client.invoke_endpoint(
                    EndpointName=model_id,
                    Body=json.dumps(request_body),
                    ContentType="application/json",
                    Accept="application/json",
                )

            user_response = self.parse_chat_response(response["Body"], start_time, user_request.num_prefill_tokens)

            # Collect metrics
            self.collect_metrics(user_response, "/bedrock/chat")

        except Exception as e:
            logger.exception(f"AWS Bedrock request failed: {e}")
            user_response = UserResponse(
                status_code=500,
                error_message=str(e),
            )
            self.collect_metrics(user_response, "/bedrock/chat")

    @task
    def embeddings(self):
        """Perform an embeddings request to AWS Bedrock."""
        # Get request using sample method
        user_request = self.sample()

        if not isinstance(user_request, UserEmbeddingRequest):
            raise AttributeError(
                "user_request should be of type UserEmbeddingRequest for "
                f"AWSBedrockUser.embeddings, got {type(user_request)}"
            )

        # Only Titan embeddings models are supported
        model_id = user_request.model
        if "titan-embed" not in model_id.lower():
            logger.error(f"Model {model_id} does not support embeddings")
            user_response = UserResponse(
                status_code=400,
                error_message=f"Model {model_id} does not support embeddings",
            )
            self.collect_metrics(user_response, "/bedrock/embeddings")
            return

        # Prepare request body
        # For embeddings, join all documents into a single text
        input_text = " ".join(user_request.documents)
        request_body = {"inputText": input_text}

        # Track timing
        start_time = time.monotonic()

        try:
            # Invoke model
            response = self.runtime_client.invoke_model(
                modelId=model_id,
                body=json.dumps(request_body),
                contentType="application/json",
                accept="application/json",
            )

            # Parse response (embeddings are not extracted for now)
            json.loads(response["body"].read())
            end_time = time.monotonic()

            # Create response
            user_response = UserResponse(
                status_code=200,
                start_time=start_time,
                time_at_first_token=end_time,  # No streaming for embeddings
                end_time=end_time,
                num_prefill_tokens=user_request.num_prefill_tokens,
            )

            # Collect metrics
            self.collect_metrics(user_response, "/bedrock/embeddings")

        except Exception as e:
            logger.error(f"AWS Bedrock embeddings request failed: {e}")
            user_response = UserResponse(
                status_code=500,
                error_message=str(e),
            )
            self.collect_metrics(user_response, "/bedrock/embeddings")

    def _prepare_request_body(
        self, request: UserChatRequest, model_id: str
    ) -> Dict[str, Any]:
        """Prepare request body based on model type.

        Args:
            request: User request
            model_id: Model ID

        Returns:
            Request body dict
        """
        body: Dict[str, Any]
        # Generic format

        body = {
             "messages": [
                {
                    "role": "user",
                    "content": request.prompt,
                }
            ],
            "max_tokens": request.max_tokens or 1000,
            "stream": True,
            "stream_options": {
                "include_usage": True,
            },
        }

        if request.additional_request_params.get("temperature") is not None:
            body["temperature"] = request.additional_request_params["temperature"]
        if request.additional_request_params.get("top_p") is not None:
            body["top_p"] = request.additional_request_params["top_p"]

        # Add any additional params
        body.update(request.additional_request_params)

        return body

    def _supports_streaming(self, model_id: str) -> bool:
        """Check if model supports streaming.

        Args:
            model_id: Model ID

        Returns:
            True if streaming is supported
        """
        # Most models support streaming except embeddings
        return "embed" not in model_id.lower()

    def parse_chat_response(
        self,
        response: EventStream,
        start_time: float,
        num_prefill_tokens: int,
        _: float = 0,
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
        chunk = ""
        for chunk_part in response:
            # Caution: Adding logs here can make debug mode unusable.
            chunk_part = chunk_part['PayloadPart']['Bytes'].decode("utf8")
            chunk += chunk_part

            if not chunk.endswith("\n\n"):
                continue
        
            chunk = chunk[len(stream_chunk_prefix) :]
            if chunk == end_chunk:
                break
            data = json.loads(chunk)

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
                        raise Exception("Invalid Response")
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
                logger.exception(
                    f"Error processing chunk: {e}, data: {data}, "
                    f"previous_data: {previous_data}, "
                    f"finish_reason: {finish_reason}, skipping"
                )

            chunk = ""
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