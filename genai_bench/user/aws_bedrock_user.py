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
    UserImageChatRequest,
    UserResponse,
)
from genai_bench.user.base_user import BaseUser

logger = init_logger(__name__)


class AWSBedrockUser(BaseUser):
    """AWS Bedrock API user implementation."""

    BACKEND_NAME = "aws-bedrock"

    # Task mapping
    supported_tasks = {
        "text-to-text": "chat",
        "text-to-embeddings": "embeddings",
        "image-to-text": "chat",  # Same method handles both text and image
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
            service_name="bedrock-runtime", config=config
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
        first_token_time = None

        try:
            # Determine if streaming
            use_streaming = user_request.additional_request_params.get(
                "stream", True
            ) and self._supports_streaming(model_id)

            if use_streaming:
                # Streaming request
                response = self.runtime_client.invoke_model_with_response_stream(
                    modelId=model_id,
                    body=json.dumps(request_body),
                    contentType="application/json",
                    accept="application/json",
                )

                # Process streaming response
                full_response = ""
                for event in response["body"]:
                    chunk = json.loads(event["chunk"]["bytes"])

                    if first_token_time is None:
                        first_token_time = time.monotonic()

                    # Extract text based on model type
                    chunk_text = self._extract_chunk_text(chunk, model_id)
                    if chunk_text:
                        full_response += chunk_text

                response_text = full_response

            else:
                # Non-streaming request
                response = self.runtime_client.invoke_model(
                    modelId=model_id,
                    body=json.dumps(request_body),
                    contentType="application/json",
                    accept="application/json",
                )

                # Parse response
                response_body = json.loads(response["body"].read())
                first_token_time = time.monotonic()

                # Extract response text based on model type
                response_text = self._extract_response_text(response_body, model_id)

            end_time = time.monotonic()

            # Create response
            user_response = UserChatResponse(
                status_code=200,
                generated_text=response_text,
                tokens_received=len(response_text.split()),  # Approximate token count
                time_at_first_token=first_token_time,
                num_prefill_tokens=user_request.num_prefill_tokens,
                start_time=start_time,
                end_time=end_time,
            )

            # Collect metrics
            self.collect_metrics(user_response, "/bedrock/chat")

        except Exception as e:
            logger.error(f"AWS Bedrock request failed: {e}")
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
        # Handle different model families
        body: Dict[str, Any]
        if "claude" in model_id.lower():
            # Anthropic Claude models
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "messages": [],
                "max_tokens": request.max_tokens or 1000,
            }

            # Add messages
            if isinstance(request, UserImageChatRequest) and request.image_content:
                # Multimodal request
                content: List[Dict[str, Any]] = [
                    {"type": "text", "text": request.prompt}
                ]
                for image in request.image_content:
                    source_data: Dict[str, str] = {
                        "type": "base64",
                        "media_type": "image/jpeg",  # Assume JPEG
                        "data": image,
                    }
                    content.append({"type": "image", "source": source_data})
                body["messages"].append({"role": "user", "content": content})
            else:
                # Text-only request
                body["messages"].append({"role": "user", "content": request.prompt})

            # Add additional params
            if request.additional_request_params.get("temperature") is not None:
                body["temperature"] = request.additional_request_params["temperature"]
            if request.additional_request_params.get("top_p") is not None:
                body["top_p"] = request.additional_request_params["top_p"]

        elif "titan" in model_id.lower():
            # Amazon Titan models
            body = {
                "inputText": request.prompt,
                "textGenerationConfig": {
                    "maxTokenCount": request.max_tokens or 1000,
                },
            }

            if request.additional_request_params.get("temperature") is not None:
                body["textGenerationConfig"]["temperature"] = (
                    request.additional_request_params["temperature"]
                )
            if request.additional_request_params.get("top_p") is not None:
                body["textGenerationConfig"]["topP"] = (
                    request.additional_request_params["top_p"]
                )

        elif "llama" in model_id.lower():
            # Meta Llama models
            body = {
                "prompt": request.prompt,
                "max_gen_len": request.max_tokens or 1000,
            }

            if request.additional_request_params.get("temperature") is not None:
                body["temperature"] = request.additional_request_params["temperature"]
            if request.additional_request_params.get("top_p") is not None:
                body["top_p"] = request.additional_request_params["top_p"]

        else:
            # Generic format
            body = {
                "prompt": request.prompt,
                "max_tokens": request.max_tokens or 1000,
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

    def _extract_chunk_text(self, chunk: Dict[str, Any], model_id: str) -> str:
        """Extract text from streaming chunk based on model type.

        Args:
            chunk: Streaming chunk
            model_id: Model ID

        Returns:
            Extracted text
        """
        if "claude" in model_id.lower():
            # Claude models
            if chunk.get("type") == "content_block_delta":
                return chunk.get("delta", {}).get("text", "")
        elif "titan" in model_id.lower():
            # Titan models
            return chunk.get("outputText", "")
        elif "llama" in model_id.lower():
            # Llama models
            return chunk.get("generation", "")

        # Try common fields
        return chunk.get("text", chunk.get("output", ""))

    def _extract_response_text(self, response: Dict[str, Any], model_id: str) -> str:
        """Extract response text based on model type.

        Args:
            response: Response body
            model_id: Model ID

        Returns:
            Extracted text
        """
        if "claude" in model_id.lower():
            # Claude models
            content = response.get("content", [])
            if content and isinstance(content, list):
                return content[0].get("text", "")
        elif "titan" in model_id.lower():
            # Titan models
            results = response.get("results", [])
            if results:
                return results[0].get("outputText", "")
        elif "llama" in model_id.lower():
            # Llama models
            return response.get("generation", "")

        # Try common fields
        return response.get(
            "text", response.get("output", response.get("completion", ""))
        )
