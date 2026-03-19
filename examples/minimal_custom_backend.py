"""Minimal example of a custom backend implementation.

This is the simplest possible custom backend, useful for testing and
as a template.

Usage:
    genai-bench benchmark \
        --api-backend custom \
        --custom-backend examples/minimal_custom_backend.py \
        --api-base http://localhost:8000 \
        --api-model-name test-model \
        --task text-to-text \
        --model-tokenizer gpt2

    # Or with explicit class name:
    genai-bench benchmark \
        --api-backend custom \
        --custom-backend \
            examples/minimal_custom_backend.py:MinimalCustomUser \
        --api-base http://localhost:8000 \
        --api-model-name test-model \
        --task text-to-text \
        --model-tokenizer gpt2
"""

from locust import task
import time

from genai_bench.logging import init_logger
from genai_bench.protocol import UserChatRequest, UserChatResponse, UserResponse
from genai_bench.user.base_user import BaseUser

logger = init_logger(__name__)


class MinimalCustomUser(BaseUser):
    """Minimal custom backend for testing."""

    BACKEND_NAME = "minimal-custom"

    supported_tasks = {
        "text-to-text": "chat",
    }

    def on_start(self):
        """Initialize - nothing needed for this minimal example."""
        logger.info("MinimalCustomUser initialized")

    @task
    def chat(self):
        """Handle chat requests with a mock response."""
        user_request = self.sample()

        if not isinstance(user_request, UserChatRequest):
            raise AttributeError(
                f"Expected UserChatRequest, got {type(user_request)}"
            )

        start_time = time.monotonic()

        try:
            # Simulate some processing time
            time.sleep(0.1)

            # Create a mock response
            generated_text = f"Response to: {user_request.prompt[:50]}..."
            tokens_received = len(generated_text.split())  # Rough estimate

            user_response = UserChatResponse(
                status_code=200,
                generated_text=generated_text,
                tokens_received=tokens_received,
                time_at_first_token=time.monotonic(),
                num_prefill_tokens=user_request.num_prefill_tokens,
                start_time=start_time,
                end_time=time.monotonic(),
            )

            self.collect_metrics(user_response, "/minimal/chat")

        except Exception as e:
            logger.exception(f"Request failed: {e}")
            error_response = UserResponse(
                status_code=500,
                error_message=str(e),
            )
            self.collect_metrics(error_response, "/minimal/chat")
