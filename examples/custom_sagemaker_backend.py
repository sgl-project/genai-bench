"""Example custom backend implementation for AWS SageMaker.

This is an example of how to create a custom API backend for genai-bench.
You can use this as a template to create your own custom backends.

Usage:
    genai-bench benchmark \
        --api-backend custom \
        --custom-backend examples/custom_sagemaker_backend.py \
        --api-base <your-sagemaker-endpoint> \
        --api-model-name <your-model-id> \
        --task text-to-text \
        --model-tokenizer <your-tokenizer>

Key points for custom backends:
1. Your class must inherit from BaseUser
2. Set the BACKEND_NAME class attribute
3. Define supported_tasks mapping task names to method names
4. Implement the task methods (chat, embeddings, etc.)
5. Implement on_start() to initialize any clients/auth
6. Use self.sample() to get requests and self.collect_metrics() to report results
"""

from locust import task

import json
import time
from typing import Any, Dict, Optional

from genai_bench.logging import init_logger
from genai_bench.protocol import (
    UserChatRequest,
    UserChatResponse,
    UserResponse,
)
from genai_bench.user.base_user import BaseUser

import contextlib

with contextlib.suppress(ImportError):
    from botocore.eventstream import EventStream

logger = init_logger(__name__)


class CustomSagemakerUser(BaseUser):
    """Custom AWS SageMaker backend implementation.

    This example shows how to create a custom backend for SageMaker endpoints
    that use OpenAI-compatible streaming format.
    """

    # REQUIRED: Backend identifier
    BACKEND_NAME = "custom-sagemaker"

    # REQUIRED: Task mapping - maps task types to method names
    supported_tasks = {
        "text-to-text": "chat",
        "image-text-to-text": "chat",
    }

    # Instance variables
    host: Optional[str] = None
    auth_provider = None
    runtime_client = None

    def on_start(self):
        """Initialize AWS SageMaker runtime client.

        This method is called when each user starts. Use it to:
        - Initialize API clients
        - Set up authentication
        - Load configurations
        """
        # Note: For custom backends, auth_provider may be None
        # You can implement custom auth logic here

        try:
            import boto3
            from botocore.config import Config
        except ImportError as e:
            raise ImportError(
                "boto3 is required for SageMaker. "
                "Install it with: pip install boto3"
            ) from e

        # Get AWS credentials - you can customize this logic
        # For this example, we'll use default credentials
        session = boto3.Session()

        # Create SageMaker runtime client
        region = "us-east-1"  # You can make this configurable
        config = Config(region_name=region)

        self.runtime_client = session.client(
            service_name="sagemaker-runtime",
            config=config
        )

        logger.info(f"Initialized SageMaker runtime client in region {region}")

    @task
    def chat(self):
        """Perform a chat request to SageMaker endpoint.

        This is the main task method that gets called by the load tester.
        """
        # Get request from the sampler
        user_request = self.sample()

        if not isinstance(user_request, UserChatRequest):
            raise AttributeError(
                "user_request should be of type UserChatRequest for chat task, "
                f"got {type(user_request)}"
            )

        # Prepare request body
        model_id = user_request.model
        request_body = self._prepare_request_body(user_request)

        # Track timing
        start_time = time.monotonic()

        try:
            # Make streaming request to SageMaker
            response = self.runtime_client.invoke_endpoint_with_response_stream(
                EndpointName=model_id,
                Body=json.dumps(request_body),
                ContentType="application/json",
                Accept="application/json",
            )

            # Parse the streaming response
            user_response = self.parse_chat_response(
                response["Body"],
                start_time,
                user_request.num_prefill_tokens
            )

            # Collect metrics
            self.collect_metrics(user_response, "/sagemaker/chat")

        except Exception as e:
            logger.exception(f"SageMaker request failed: {e}")
            user_response = UserResponse(
                status_code=500,
                error_message=str(e),
            )
            self.collect_metrics(user_response, "/sagemaker/chat")

    def _prepare_request_body(self, request: UserChatRequest) -> Dict[str, Any]:
        """Prepare request body for SageMaker endpoint.

        Customize this based on your model's expected format.
        """
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

        # Add optional parameters
        if request.additional_request_params.get("temperature") is not None:
            body["temperature"] = request.additional_request_params["temperature"]
        if request.additional_request_params.get("top_p") is not None:
            body["top_p"] = request.additional_request_params["top_p"]

        return body

    def parse_chat_response(
        self,
        response: EventStream,
        start_time: float,
        num_prefill_tokens: int,
    ) -> UserResponse:
        """Parse streaming response from SageMaker.

        This parser handles OpenAI-compatible streaming format.
        Customize this based on your endpoint's response format.
        """
        stream_chunk_prefix = "data: "
        end_chunk = "[DONE]"

        generated_text = ""
        tokens_received = 0
        time_at_first_token = None
        finish_reason = None
        num_prompt_tokens = None
        chunk = ""

        for chunk_part in response:
            # Parse the streaming chunk
            chunk_part = chunk_part['PayloadPart']['Bytes'].decode("utf8")
            chunk += chunk_part

            if not chunk.endswith("\n\n"):
                continue

            chunk = chunk[len(stream_chunk_prefix):].strip()
            if chunk == end_chunk:
                break
            if not chunk:
                continue

            data = json.loads(chunk)

            # Handle errors
            if data.get("error") is not None:
                return UserResponse(
                    status_code=data["error"].get("code", -1),
                    error_message=data["error"].get(
                        "message", "Unknown error"
                    ),
                )

            # Process usage info in final chunk
            if not data["choices"] and finish_reason and "usage" in data:
                num_prompt_tokens = data["usage"].get(
                    "prompt_tokens", num_prefill_tokens
                )
                tokens_received = data["usage"].get("completion_tokens", 0)
                if not time_at_first_token and tokens_received > 0:
                    time_at_first_token = time.monotonic()
                break

            try:
                delta = data["choices"][0]["delta"]
                content = delta.get("content") or delta.get("reasoning_content")
                usage = delta.get("usage")

                if usage:
                    tokens_received = usage["completion_tokens"]

                if content:
                    if not time_at_first_token:
                        time_at_first_token = time.monotonic()
                    generated_text += content

                finish_reason = data["choices"][0].get("finish_reason", None)

            except (IndexError, KeyError) as e:
                logger.debug(f"Error processing chunk: {e}")

            chunk = ""

        end_time = time.monotonic()

        # Estimate tokens if not provided
        if not tokens_received and generated_text:
            tokens_received = self.environment.sampler.get_token_length(
                generated_text, add_special_tokens=False
            )
            logger.warning("No usage info returned, estimated tokens from text")

        return UserChatResponse(
            status_code=200,
            generated_text=generated_text,
            tokens_received=tokens_received,
            time_at_first_token=time_at_first_token,
            num_prefill_tokens=num_prefill_tokens or num_prompt_tokens,
            start_time=start_time,
            end_time=end_time,
        )
