"""GCP Vertex AI user implementation."""

from locust import task

import json
import time
from typing import Any, Callable, Dict, List, Optional

import requests
from requests import Response

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


class GCPVertexUser(BaseUser):
    """GCP Vertex AI API user implementation."""

    BACKEND_NAME = "gcp-vertex"

    # Task mapping
    supported_tasks = {
        "text-to-text": "chat",
        "text-to-embeddings": "embeddings",
        "image-text-to-text": "chat",  # Same method handles both text and image
    }

    host: Optional[str] = None
    auth_provider: Optional[AuthProvider] = None
    project_id: Optional[str] = None
    location: Optional[str] = None
    headers: Dict[str, str] = {}

    def on_start(self):
        """Initialize GCP Vertex AI client using auth provider."""
        if not self.auth_provider:
            raise ValueError("Auth provider not set for GCP Vertex AI")

        # Get config
        config = self.auth_provider.get_config()
        self.project_id = config.get("project_id")
        self.location = config.get("location", "us-central1")

        if not self.project_id:
            raise ValueError("Project ID is required for Vertex AI")

        # Set up authentication
        auth_type = config.get("auth_type")

        if auth_type == "api_key":
            # API key authentication
            self.headers = self.auth_provider.get_headers()
            self.headers["Content-Type"] = "application/json"
        else:
            # Service account authentication
            try:
                import google.auth
                from google.auth.transport.requests import Request
                from google.oauth2 import service_account
            except ImportError as e:
                raise ImportError(
                    "google-auth is required for Vertex AI service account auth. "
                    "Install it with: pip install google-auth"
                ) from e

            credentials_path = config.get("credentials_path")

            if credentials_path:
                # Load service account credentials
                credentials = service_account.Credentials.from_service_account_file(
                    credentials_path,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )
            else:
                # Use default credentials
                credentials, _ = google.auth.default(
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )

            # Get access token
            auth_req = Request()
            credentials.refresh(auth_req)

            self.headers = {
                "Authorization": f"Bearer {credentials.token}",
                "Content-Type": "application/json",
            }

        logger.info(f"Initialized GCP Vertex AI client for project {self.project_id}")
        super().on_start()

    @task
    def chat(self):
        """Perform a chat request to GCP Vertex AI."""
        # Get request using sample method
        user_request = self.sample()

        if not isinstance(user_request, UserChatRequest):
            raise AttributeError(
                "user_request should be of type UserChatRequest for "
                f"GCPVertexUser.chat, got {type(user_request)}"
            )

        # Determine model family
        model_name = user_request.model
        is_gemini = "gemini" in model_name.lower()
        use_streaming = user_request.additional_request_params.get("stream", True)

        # Build URL based on model type
        if is_gemini:
            # Gemini models use generateContent endpoint
            endpoint_method = (
                "streamGenerateContent" if use_streaming else "generateContent"
            )
            endpoint = (
                f"/v1/projects/{self.project_id}/locations/{self.location}/"
                f"publishers/google/models/{model_name}:{endpoint_method}"
            )
        else:
            # Other models (e.g., text-bison) use predict endpoint
            endpoint = (
                f"/v1/projects/{self.project_id}/locations/{self.location}/"
                f"publishers/google/models/{model_name}:predict"
            )

        # Prepare request body
        request_body = self._prepare_request_body(user_request, model_name)

        # Send request
        self.send_request(
            use_streaming and is_gemini,  # Only Gemini supports streaming
            endpoint,
            request_body,
            self.parse_chat_response if is_gemini else self.parse_palm_response,
            user_request.num_prefill_tokens,
            model_name,
        )

    @task
    def embeddings(self):
        """Perform an embeddings request to GCP Vertex AI."""
        # Get request using sample method
        user_request = self.sample()

        if not isinstance(user_request, UserEmbeddingRequest):
            raise AttributeError(
                "user_request should be of type UserEmbeddingRequest for "
                f"GCPVertexUser.embeddings, got {type(user_request)}"
            )

        # Build URL for embeddings model
        model_name = user_request.model
        endpoint = (
            f"/v1/projects/{self.project_id}/locations/{self.location}/"
            f"publishers/google/models/{model_name}:predict"
        )

        # Prepare request body
        # Vertex AI expects each document as a separate instance
        request_body = {
            "instances": [{"content": doc} for doc in user_request.documents]
        }

        # Add additional params
        if user_request.additional_request_params:
            request_body["parameters"] = user_request.additional_request_params

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
        model_name: Optional[str] = None,
    ) -> UserResponse:
        """
        Sends a POST request, handling both streaming and non-streaming responses.

        Args:
            endpoint (str): The API endpoint.
            payload (Dict[str, Any]): The JSON payload for the request.
            stream (bool): Whether to stream the response.
            parse_strategy (Callable): The function to parse the response.
            num_prefill_tokens (Optional[int]): The num of tokens in the prefill/prompt.
            model_name (Optional[str]): The model name for response parsing.

        Returns:
            UserResponse: A response object containing status and metrics data.
        """
        response = None
        base_url = f"https://{self.location}-aiplatform.googleapis.com"

        try:
            start_time = time.monotonic()
            response = requests.post(
                url=f"{base_url}{endpoint}",
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
                    model_name,
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
                status_code=500,
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
        model_name: Optional[str] = None,
    ) -> UserResponse:
        """
        Parses a Gemini streaming response.

        Args:
            response (Response): The response object.
            start_time (float): The time when the request was started.
            num_prefill_tokens (int): The num of tokens in the prefill/prompt.
            _ (float): Placeholder for an unused var.
            model_name (Optional[str]): The model name.

        Returns:
            UserChatResponse: A response object with metrics and generated text.
        """
        generated_text = ""
        time_at_first_token = None
        reasoning_tokens = None

        for line in response.iter_lines():
            if line:
                try:
                    # Vertex AI returns JSON objects, not SSE
                    chunk = json.loads(line)

                    if time_at_first_token is None:
                        time_at_first_token = time.monotonic()

                    # Extract content from chunk
                    candidates = chunk.get("candidates", [])
                    if candidates:
                        content = candidates[0].get("content", {})
                        parts = content.get("parts", [])
                        for part in parts:
                            text = part.get("text", "")
                            if text:
                                generated_text += text

                    usage_metadata = chunk.get("usageMetadata", {})
                    reasoning_tokens = usage_metadata.get("thoughtsTokenCount", 0)

                except json.JSONDecodeError:
                    continue

        end_time = time.monotonic()

        # Estimate tokens if not provided
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
            reasoning_tokens=reasoning_tokens,
        )

    def parse_palm_response(
        self,
        response: Response,
        start_time: float,
        num_prefill_tokens: int,
        end_time: float,
        model_name: Optional[str] = None,
    ) -> UserResponse:
        """
        Parses a PaLM/non-Gemini response.

        Args:
            response (Response): The response object.
            start_time (float): The time when the request was started.
            num_prefill_tokens (int): The num of tokens in the prefill/prompt.
            end_time (float): The time when the request was finished.
            model_name (Optional[str]): The model name.

        Returns:
            UserChatResponse: A response object with metrics and generated text.
        """
        data = response.json()

        # Extract response text
        predictions = data.get("predictions", [])
        response_text = predictions[0].get("content", "") if predictions else ""

        # Estimate tokens
        tokens_received = self.environment.sampler.get_token_length(
            response_text, add_special_tokens=False
        )

        # No reasoning tokens for non-gemini models
        return UserChatResponse(
            status_code=200,
            generated_text=response_text,
            tokens_received=tokens_received,
            reasoning_tokens=0,
            time_at_first_token=end_time,
            num_prefill_tokens=num_prefill_tokens,
            start_time=start_time,
            end_time=end_time,
        )

    @staticmethod
    def parse_embedding_response(
        response: Response,
        start_time: float,
        _: Optional[int],
        end_time: float,
        model_name: Optional[str] = None,
    ) -> UserResponse:
        """
        Parses an embedding response.

        Args:
            response (Response): The response object.
            start_time (float): The time when the request was started.
            _ (Optional[int]): Placeholder for unused parameter.
            end_time (float): The time when the request was finished.
            model_name (Optional[str]): The model name.

        Returns:
            UserResponse: A response object with metrics.
        """
        data = response.json()

        # Extract embeddings - count tokens from all predictions
        predictions = data.get("predictions", [])
        num_embeddings = len(predictions)

        return UserResponse(
            status_code=200,
            start_time=start_time,
            end_time=end_time,
            time_at_first_token=end_time,
            num_prefill_tokens=num_embeddings,  # Use number of embeddings as a proxy
        )

    def _prepare_request_body(
        self, request: UserChatRequest, model_name: str
    ) -> Dict[str, Any]:
        """Prepare request body based on model type.

        Args:
            request: User request
            model_name: Model name

        Returns:
            Request body dict
        """
        is_gemini = "gemini" in model_name.lower()
        body: Dict[str, Any]

        if is_gemini:
            # Gemini models use different format
            contents = []

            if isinstance(request, UserImageChatRequest) and request.image_content:
                # Multimodal request
                parts: List[Dict[str, Any]] = [{"text": request.prompt}]
                for image in request.image_content:
                    if image.startswith("data:image/"):
                        # Extract base64 data from data URL for inline data
                        image_data = image.split(",", 1)[1]
                        parts.append(
                            {
                                "inlineData": {
                                    "mimeType": "image/jpeg",
                                    "data": image_data,
                                }
                            }
                        )
                    elif image.startswith(("http://", "https://", "gs://")):
                        # Use fileData for HTTP URLs and Cloud Storage URIs
                        parts.append(
                            {
                                "fileData": {
                                    "mimeType": "image/jpeg",
                                    "fileUri": image,
                                }
                            }
                        )
                    else:
                        raise ValueError(
                            f"Unsupported image format for GCP Vertex AI: {type(image)}"
                        )

                contents.append({"parts": parts})
            else:
                # Text-only request
                contents.append({"parts": [{"text": request.prompt}]})

            # Build request body
            body = {"contents": contents}

            # Add generation config
            generation_config = {}
            if request.max_tokens:
                generation_config["maxOutputTokens"] = request.max_tokens
            if request.additional_request_params.get("temperature") is not None:
                generation_config["temperature"] = request.additional_request_params[
                    "temperature"
                ]
            if request.additional_request_params.get("top_p") is not None:
                generation_config["topP"] = request.additional_request_params["top_p"]
            if request.additional_request_params.get("top_k") is not None:
                generation_config["topK"] = request.additional_request_params["top_k"]

            if generation_config:
                body["generationConfig"] = generation_config

        else:
            # PaLM and other models
            body = {
                "instances": [{"content": request.prompt}],
                "parameters": {},
            }

            # Add parameters
            if request.max_tokens:
                body["parameters"]["maxOutputTokens"] = request.max_tokens
            if request.additional_request_params.get("temperature") is not None:
                body["parameters"]["temperature"] = request.additional_request_params[
                    "temperature"
                ]
            if request.additional_request_params.get("top_p") is not None:
                body["parameters"]["topP"] = request.additional_request_params["top_p"]
            if request.additional_request_params.get("top_k") is not None:
                body["parameters"]["topK"] = request.additional_request_params["top_k"]

        # Add any additional params
        if request.additional_request_params:
            if is_gemini:
                # For Gemini, merge additional params at the top level
                for key, value in request.additional_request_params.items():
                    if key not in ["temperature", "top_p", "top_k", "stream"]:
                        body[key] = value
            else:
                # For other models, merge into parameters
                body["parameters"].update(request.additional_request_params)

        return body
