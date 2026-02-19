"""Tests for GCP Vertex AI user implementation."""

import json
import time
from unittest.mock import MagicMock, patch

import pytest
import requests

from genai_bench.auth.model_auth_provider import ModelAuthProvider
from genai_bench.protocol import (
    UserChatRequest,
    UserChatResponse,
    UserEmbeddingRequest,
    UserImageChatRequest,
    UserResponse,
)
from genai_bench.user.gcp_vertex_user import GCPVertexUser


class TestGCPVertexUser:
    """Test GCP Vertex AI user implementation."""

    @pytest.fixture
    def mock_auth(self):
        """Create mock auth provider."""
        auth = MagicMock(spec=ModelAuthProvider)
        auth.get_headers.return_value = {"x-goog-api-key": "test_key"}
        auth.get_config.return_value = {
            "project_id": "test-project",
            "location": "us-central1",
            "auth_type": "api_key",
        }
        return auth

    @pytest.fixture
    def vertex_user(self, mock_auth):
        """Create GCP Vertex user instance."""
        env = MagicMock()
        env.sampler = MagicMock()
        env.sampler.get_token_length.return_value = 50
        GCPVertexUser.host = "http://localhost"
        user = GCPVertexUser(environment=env)
        user.auth_provider = mock_auth
        user.config = MagicMock()
        user.config.model = "gemini-pro"
        return user

    def test_backend_name(self):
        """Test backend name constant."""
        assert GCPVertexUser.BACKEND_NAME == "gcp-vertex"

    def test_supported_tasks(self):
        """Test supported tasks mapping."""
        assert GCPVertexUser.supported_tasks == {
            "text-to-text": "chat",
            "text-to-embeddings": "embeddings",
            "image-text-to-text": "chat",
        }

    def test_init(self):
        """Test initialization."""
        GCPVertexUser.host = "http://localhost"
        user = GCPVertexUser(environment=MagicMock())
        # host is set as class attribute
        assert user.host == "http://localhost"
        assert user.auth_provider is None
        assert user.project_id is None
        assert user.location is None
        assert user.headers == {}

    def test_on_start_with_auth(self, vertex_user):
        """Test on_start with auth provider."""
        # Mock the super().on_start() call
        with patch.object(GCPVertexUser.__bases__[0], "on_start"):
            vertex_user.on_start()

        assert vertex_user.project_id == "test-project"
        assert vertex_user.location == "us-central1"
        assert vertex_user.headers == {
            "x-goog-api-key": "test_key",
            "Content-Type": "application/json",
        }

    def test_on_start_no_auth(self, vertex_user):
        """Test on_start without auth provider."""
        vertex_user.auth_provider = None

        with pytest.raises(ValueError, match="Auth provider not set"):
            vertex_user.on_start()

    def test_on_start_with_service_account(self, vertex_user):
        """Test on_start with service account auth."""
        vertex_user.auth_provider.get_headers.return_value = {}
        # Change config to not use api_key auth
        vertex_user.auth_provider.get_config.return_value = {
            "project_id": "test-project",
            "location": "us-central1",
            # No auth_type means it will use service account auth
        }

        with patch("google.auth.default") as mock_auth:
            mock_creds = MagicMock()
            mock_creds.token = "test-token"
            mock_auth.return_value = (mock_creds, "test-project")

            with patch("google.auth.transport.requests.Request"):
                with patch.object(GCPVertexUser.__bases__[0], "on_start"):
                    vertex_user.on_start()

                mock_creds.refresh.assert_called_once()
                assert vertex_user.headers["Authorization"] == "Bearer test-token"

    def test_on_start_import_error(self, vertex_user):
        """Test on_start with google auth import error."""
        vertex_user.auth_provider.get_headers.return_value = {}
        # Change config to not use api_key auth to trigger import
        vertex_user.auth_provider.get_config.return_value = {
            "project_id": "test-project",
            "location": "us-central1",
            # No auth_type means it will try to use service account auth
        }

        with patch.dict("sys.modules", {"google.auth": None}):
            with pytest.raises(ImportError, match="google-auth is required"):
                vertex_user.on_start()

    @patch("requests.post")
    def test_chat_gemini_text_request(self, mock_post, vertex_user):
        """Test chat with Gemini text-only request."""
        # Setup
        vertex_user.on_start()

        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = [
            json.dumps(
                {"candidates": [{"content": {"parts": [{"text": "Test response"}]}}]}
            ).encode()
        ]
        mock_post.return_value = mock_response

        # Create request
        request = UserChatRequest(
            prompt="Test prompt",
            model="gemini-pro",
            max_tokens=100,
            temperature=0.7,
            num_prefill_tokens=10,
            additional_request_params={"temperature": 0.7},
        )

        # Mock sample method
        vertex_user.sample = MagicMock(return_value=request)
        vertex_user.collect_metrics = MagicMock()

        # Execute
        vertex_user.chat()

        # Verify
        mock_post.assert_called_once()
        call_args = mock_post.call_args

        # Check the URL
        url = call_args.kwargs["url"]
        assert ":streamGenerateContent" in url
        assert "test-project" in url
        assert "us-central1" in url
        assert "gemini-pro" in url

        # Get the request body
        body = call_args.kwargs.get("json", {})
        assert body["contents"][0]["parts"][0]["text"] == "Test prompt"
        assert body["generationConfig"]["maxOutputTokens"] == 100
        assert body["generationConfig"]["temperature"] == 0.7

        vertex_user.collect_metrics.assert_called_once()
        response = vertex_user.collect_metrics.call_args[0][0]
        assert isinstance(response, UserChatResponse)
        assert response.status_code == 200
        assert response.generated_text == "Test response"

    @patch("requests.post")
    def test_chat_gemini_image_request(self, mock_post, vertex_user):
        """Test chat with Gemini image request."""
        # Setup
        vertex_user.on_start()

        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = [
            json.dumps(
                {
                    "candidates": [
                        {"content": {"parts": [{"text": "Image description"}]}}
                    ]
                }
            ).encode()
        ]
        mock_post.return_value = mock_response

        # Create request
        request = UserImageChatRequest(
            prompt="Describe this image",
            model="gemini-pro-vision",
            max_tokens=100,
            image_content=["data:image/jpeg;base64,base64_image_data"],
            num_prefill_tokens=10,
            num_images=1,
        )

        # Mock sample method
        vertex_user.sample = MagicMock(return_value=request)
        vertex_user.collect_metrics = MagicMock()

        # Execute
        vertex_user.chat()

        # Verify
        call_args = mock_post.call_args
        body = call_args.kwargs["json"]

        parts = body["contents"][0]["parts"]
        assert len(parts) == 2
        assert parts[0]["text"] == "Describe this image"
        assert parts[1]["inlineData"]["data"] == "base64_image_data"
        assert parts[1]["inlineData"]["mimeType"] == "image/jpeg"

    @patch("requests.post")
    def test_chat_palm_model(self, mock_post, vertex_user):
        """Test chat with PaLM model."""
        # Setup
        vertex_user.on_start()
        vertex_user.config.model = "text-bison"

        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "predictions": [{"content": "PaLM response"}]
        }
        mock_post.return_value = mock_response

        # Create request
        request = UserChatRequest(
            prompt="Test prompt",
            model="text-bison",
            max_tokens=100,
            temperature=0.5,
            top_p=0.9,
            top_k=40,
            num_prefill_tokens=10,
            additional_request_params={"temperature": 0.5, "top_p": 0.9, "top_k": 40},
        )

        # Mock sample method
        vertex_user.sample = MagicMock(return_value=request)
        vertex_user.collect_metrics = MagicMock()

        # Execute
        vertex_user.chat()

        # Verify
        call_args = mock_post.call_args
        url = call_args.kwargs["url"]
        assert ":predict" in url
        assert "text-bison" in url

        body = call_args.kwargs["json"]
        assert body["instances"][0]["content"] == "Test prompt"
        assert body["parameters"]["maxOutputTokens"] == 100
        assert body["parameters"]["temperature"] == 0.5
        assert body["parameters"]["topP"] == 0.9
        assert body["parameters"]["topK"] == 40

    @patch("requests.post")
    def test_chat_non_streaming(self, mock_post, vertex_user):
        """Test chat with non-streaming response."""
        # Setup
        vertex_user.on_start()

        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        # For non-streaming, the response still goes through parse_chat_response
        # which expects iter_lines
        mock_response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "Non-streaming response"}]}}]
        }
        mock_response.iter_lines.return_value = [
            json.dumps(
                {
                    "candidates": [
                        {"content": {"parts": [{"text": "Non-streaming response"}]}}
                    ]
                }
            ).encode()
        ]
        mock_post.return_value = mock_response

        # Create request with streaming disabled
        request = UserChatRequest(
            prompt="Test prompt",
            model="gemini-pro",
            max_tokens=100,
            additional_request_params={"stream": False},
            num_prefill_tokens=10,
        )

        # Mock sample method
        vertex_user.sample = MagicMock(return_value=request)
        vertex_user.collect_metrics = MagicMock()

        # Execute
        vertex_user.chat()

        # Verify
        url = mock_post.call_args.kwargs["url"]
        assert ":generateContent" in url
        response = vertex_user.collect_metrics.call_args[0][0]
        assert response.generated_text == "Non-streaming response"

    @patch("requests.post")
    def test_chat_error_handling(self, mock_post, vertex_user):
        """Test chat error handling."""
        # Setup
        vertex_user.on_start()
        mock_post.side_effect = requests.exceptions.RequestException("API Error")

        # Create request
        request = UserChatRequest(
            prompt="Test prompt",
            model="gemini-pro",
            max_tokens=100,
            num_prefill_tokens=10,
        )

        # Mock sample method
        vertex_user.sample = MagicMock(return_value=request)
        vertex_user.collect_metrics = MagicMock()

        # Execute
        vertex_user.chat()

        # Verify error response
        response = vertex_user.collect_metrics.call_args[0][0]
        assert response.status_code == 500
        assert "API Error" in response.error_message

    @patch("requests.post")
    def test_embeddings_request(self, mock_post, vertex_user):
        """Test embeddings request."""
        # Setup
        vertex_user.on_start()
        vertex_user.config.model = "textembedding-gecko"

        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "predictions": [
                {"embeddings": {"values": [0.1, 0.2, 0.3]}},
                {"embeddings": {"values": [0.4, 0.5, 0.6]}},
            ]
        }
        mock_post.return_value = mock_response

        # Create request
        request = UserEmbeddingRequest(
            documents=["Doc 1", "Doc 2"],
            model="textembedding-gecko",
            num_prefill_tokens=10,
        )

        # Mock sample method
        vertex_user.sample = MagicMock(return_value=request)
        vertex_user.collect_metrics = MagicMock()

        # Execute
        vertex_user.embeddings()

        # Verify
        call_args = mock_post.call_args
        body = call_args.kwargs["json"]

        assert len(body["instances"]) == 2
        assert body["instances"][0]["content"] == "Doc 1"
        assert body["instances"][1]["content"] == "Doc 2"

        response = vertex_user.collect_metrics.call_args[0][0]
        assert response.status_code == 200

    @patch("requests.post")
    def test_embeddings_error_handling(self, mock_post, vertex_user):
        """Test embeddings error handling."""
        # Setup
        vertex_user.on_start()
        vertex_user.config.model = "textembedding-gecko"
        mock_post.return_value.status_code = 400
        mock_post.return_value.text = "Bad request"

        # Create request
        request = UserEmbeddingRequest(
            documents=["Test"],
            model="textembedding-gecko",
            num_prefill_tokens=10,
        )

        # Mock sample method
        vertex_user.sample = MagicMock(return_value=request)
        vertex_user.collect_metrics = MagicMock()

        # Execute
        vertex_user.embeddings()

        # Verify error response
        response = vertex_user.collect_metrics.call_args[0][0]
        assert response.status_code == 400
        assert "Bad request" in response.error_message

    def test_send_request_success(self, vertex_user):
        """Test send_request with successful response."""
        vertex_user.on_start()

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("requests.post", return_value=mock_response) as mock_post:
            # The send_request method has different signature
            parse_fn = MagicMock(
                return_value=UserResponse(
                    status_code=200,
                    num_prefill_tokens=10,
                    end_to_end_time=1.0,
                    time_at_first_token=0.5,
                    start_time=time.time(),
                    end_time=time.time() + 1.0,
                )
            )
            response = vertex_user.send_request(
                stream=False,
                endpoint="/test/endpoint",
                payload={"test": "data"},
                parse_strategy=parse_fn,
                num_prefill_tokens=10,
                model_name="test-model",
            )

            assert isinstance(response, UserResponse)
            assert response.status_code == 200
            mock_post.assert_called_once_with(
                url="https://us-central1-aiplatform.googleapis.com/test/endpoint",
                headers=vertex_user.headers,
                json={"test": "data"},
                stream=False,
            )

    def test_send_request_error(self, vertex_user):
        """Test send_request with error response."""
        vertex_user.on_start()

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Error message"
        mock_response.raise_for_status.side_effect = requests.HTTPError()

        with patch("requests.post", return_value=mock_response):
            parse_fn = MagicMock()
            response = vertex_user.send_request(
                stream=False,
                endpoint="/test/endpoint",
                payload={},
                parse_strategy=parse_fn,
                num_prefill_tokens=10,
                model_name="test-model",
            )
            # The method catches errors and returns UserResponse with error
            assert response.status_code == 400

    def test_parse_streaming_response_gemini(self, vertex_user):
        """Test parsing streaming response for Gemini."""
        mock_response = MagicMock()
        mock_response.iter_lines.return_value = [
            json.dumps(
                {"candidates": [{"content": {"parts": [{"text": "Part 1"}]}}]}
            ).encode(),
            json.dumps(
                {"candidates": [{"content": {"parts": [{"text": " Part 2"}]}}]}
            ).encode(),
        ]

        result = vertex_user.parse_chat_response(
            mock_response, 0.0, 10, 1.0, "gemini-pro"
        )

        assert result.generated_text == "Part 1 Part 2"
        assert result.status_code == 200

    def test_parse_chat_response_reasoning_tokens_from_usage_metadata(
        self, vertex_user
    ):
        """Reasoning tokens come from usageMetadata.thoughtsTokenCount."""
        mock_response = MagicMock()
        mock_response.iter_lines.return_value = [
            json.dumps(
                {"candidates": [{"content": {"parts": [{"text": "Hi"}]}}]}
            ).encode(),
            json.dumps({"usageMetadata": {"thoughtsTokenCount": 3}}).encode(),
        ]

        result = vertex_user.parse_chat_response(
            mock_response, 0.0, 10, 1.0, "gemini-pro"
        )

        assert result.reasoning_tokens == 3
        assert result.generated_text == "Hi"

    def test_parse_chat_response_no_usage_metadata(self, vertex_user):
        """When no usageMetadata, reasoning_tokens is None and tokens estimated."""
        mock_response = MagicMock()
        mock_response.iter_lines.return_value = [
            json.dumps(
                {"candidates": [{"content": {"parts": [{"text": "Hi"}]}}]}
            ).encode(),
        ]
        vertex_user.environment.sampler.get_token_length.return_value = 50

        result = vertex_user.parse_chat_response(
            mock_response, 0.0, 10, 1.0, "gemini-pro"
        )

        assert result.reasoning_tokens is None
        assert result.tokens_received == 50
        assert result.generated_text == "Hi"

    def test_parse_chat_response_usage_metadata_without_thoughts_token_count(
        self, vertex_user
    ):
        """usageMetadata without thoughtsTokenCount: reasoning_tokens is 0."""
        mock_response = MagicMock()
        mock_response.iter_lines.return_value = [
            json.dumps(
                {"candidates": [{"content": {"parts": [{"text": "Hi"}]}}]}
            ).encode(),
            json.dumps({"usageMetadata": {"candidatesTokenCount": 2}}).encode(),
        ]
        vertex_user.environment.sampler.get_token_length.return_value = 50

        result = vertex_user.parse_chat_response(
            mock_response, 0.0, 10, 1.0, "gemini-pro"
        )

        assert result.reasoning_tokens == 0
        assert result.tokens_received == 50
        assert result.generated_text == "Hi"

    def test_parse_non_streaming_response(self, vertex_user):
        """Test parsing non-streaming response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "predictions": [{"content": "Response text"}]
        }

        mock_response.status_code = 200
        result = vertex_user.parse_palm_response(
            mock_response, 0.0, 10, 1.0, "text-bison"
        )

        assert result.generated_text == "Response text"
        assert result.status_code == 200

    def test_parse_embedding_response(self, vertex_user):
        """Test parsing embedding response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "predictions": [
                {"embeddings": {"values": [0.1, 0.2]}},
                {"embeddings": {"values": [0.3, 0.4]}},
            ]
        }

        result = GCPVertexUser.parse_embedding_response(
            mock_response, 0.0, None, 1.0, "textembedding-gecko"
        )

        assert result.status_code == 200
        assert result.num_prefill_tokens == 2

    def test_prepare_request_body_additional_params(self, vertex_user):
        """Test prepare request body with additional params."""
        request = UserChatRequest(
            prompt="Test",
            model="gemini-pro",
            max_tokens=100,
            additional_request_params={
                "stopSequences": ["END"],
                "candidateCount": 3,
            },
            num_prefill_tokens=10,
        )

        body = vertex_user._prepare_request_body(request, "gemini-pro")

        assert body["stopSequences"] == ["END"]
        assert body["candidateCount"] == 3

    def test_chat_wrong_request_type(self, vertex_user):
        """Test chat with wrong request type."""
        request = UserEmbeddingRequest(
            documents=["Test"],
            model="test",
            num_prefill_tokens=10,
        )
        vertex_user.sample = MagicMock(return_value=request)

        with pytest.raises(AttributeError, match="should be of type UserChatRequest"):
            vertex_user.chat()

    def test_embeddings_wrong_request_type(self, vertex_user):
        """Test embeddings with wrong request type."""
        request = UserChatRequest(
            prompt="Test",
            model="test",
            max_tokens=100,
            num_prefill_tokens=10,
        )
        vertex_user.sample = MagicMock(return_value=request)

        with pytest.raises(
            AttributeError, match="should be of type UserEmbeddingRequest"
        ):
            vertex_user.embeddings()
