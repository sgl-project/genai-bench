"""Tests for Azure OpenAI user implementation."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from genai_bench.auth.model_auth_provider import ModelAuthProvider
from genai_bench.protocol import (
    UserChatRequest,
    UserChatResponse,
    UserEmbeddingRequest,
    UserImageChatRequest,
)
from genai_bench.user.azure_openai_user import AzureOpenAIUser


class TestAzureOpenAIUser:
    """Test Azure OpenAI user implementation."""

    @pytest.fixture
    def mock_auth(self):
        """Create mock auth provider."""
        auth = MagicMock(spec=ModelAuthProvider)
        auth.get_headers.return_value = {"api-key": "test_key"}
        auth.get_config.return_value = {
            "azure_endpoint": "https://test.openai.azure.com",
            "api_version": "2024-02-15-preview",
            "azure_deployment": "test-deployment",
        }
        return auth

    @pytest.fixture
    def azure_user(self, mock_auth):
        """Create Azure OpenAI user instance."""
        env = MagicMock()
        env.sampler = MagicMock()
        env.sampler.get_token_length.return_value = 50
        AzureOpenAIUser.host = "http://localhost"
        user = AzureOpenAIUser(environment=env)
        user.auth_provider = mock_auth
        user.config = MagicMock()
        user.config.model = "gpt-4"
        return user

    def test_backend_name(self):
        """Test backend name constant."""
        assert AzureOpenAIUser.BACKEND_NAME == "azure-openai"

    def test_supported_tasks(self):
        """Test supported tasks mapping."""
        assert AzureOpenAIUser.supported_tasks == {
            "text-to-text": "chat",
            "text-to-embeddings": "embeddings",
            "image-text-to-text": "chat",
        }

    def test_init(self):
        """Test initialization."""
        AzureOpenAIUser.host = "http://localhost"
        user = AzureOpenAIUser(environment=MagicMock())
        # host is set as class attribute
        assert user.host == "http://localhost"
        assert user.auth_provider is None
        assert user.api_version is None
        assert user.deployment_name is None
        assert user.headers == {}

    def test_on_start_with_auth(self, azure_user):
        """Test on_start with auth provider."""
        azure_user.on_start()

        assert azure_user.host == "https://test.openai.azure.com"
        assert azure_user.api_version == "2024-02-15-preview"
        assert azure_user.deployment_name == "test-deployment"
        assert azure_user.headers == {
            "api-key": "test_key",
            "Content-Type": "application/json",
        }

    def test_on_start_no_auth(self, azure_user):
        """Test on_start without auth provider."""
        azure_user.auth_provider = None

        with pytest.raises(ValueError, match="Auth provider not set"):
            azure_user.on_start()

    def test_on_start_missing_config(self, azure_user):
        """Test on_start with missing configuration."""
        azure_user.auth_provider.get_config.return_value = {}

        # Should not raise error - uses defaults
        azure_user.on_start()
        assert azure_user.api_version == "2024-02-01"  # default
        assert azure_user.deployment_name is None

    @patch("requests.post")
    def test_chat_text_request(self, mock_post, azure_user):
        """Test chat with text-only request."""
        # Setup
        azure_user.on_start()

        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = [
            b'data: {"choices":[{"delta":{"content":"Hello "}}]}',
            b'data: {"choices":[{"delta":{"content":"world!"}}]}',
            b"data: [DONE]",
        ]
        mock_post.return_value = mock_response

        # Create request
        request = UserChatRequest(
            prompt="Test prompt",
            model="gpt-4",
            max_tokens=100,
            additional_request_params={"temperature": 0.7},
            num_prefill_tokens=10,
        )

        # Mock sample method
        azure_user.sample = MagicMock(return_value=request)
        azure_user.collect_metrics = MagicMock()

        # Execute
        azure_user.chat()

        # Verify
        mock_post.assert_called_once()
        call_args = mock_post.call_args

        expected_url = (
            "https://test.openai.azure.com/openai/deployments/"
            "test-deployment/chat/completions?api-version=2024-02-15-preview"
        )
        assert call_args.kwargs["url"] == expected_url

        body = call_args.kwargs["json"]
        assert body["messages"][0]["content"] == "Test prompt"
        assert body["max_tokens"] == 100
        assert body["temperature"] == 0.7
        assert body["stream"] is True

        azure_user.collect_metrics.assert_called_once()
        response = azure_user.collect_metrics.call_args[0][0]
        assert isinstance(response, UserChatResponse)
        assert response.status_code == 200
        assert response.generated_text == "Hello world!"

    @patch("requests.post")
    def test_chat_image_request(self, mock_post, azure_user):
        """Test chat with image request."""
        # Setup
        azure_user.on_start()

        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = [
            b'data: {"choices":[{"delta":{"content":"I see an image"}}]}',
            b"data: [DONE]",
        ]
        mock_post.return_value = mock_response

        # Create request
        request = UserImageChatRequest(
            prompt="Describe this image",
            model="gpt-4-vision",
            max_tokens=100,
            image_content=["data:image/jpeg;base64,base64_image_data"],
            num_images=1,
            num_prefill_tokens=10,
        )

        # Mock sample method
        azure_user.sample = MagicMock(return_value=request)
        azure_user.collect_metrics = MagicMock()

        # Execute
        azure_user.chat()

        # Verify
        call_args = mock_post.call_args
        body = call_args[1]["json"]

        content = body["messages"][0]["content"]
        assert len(content) == 2
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "Describe this image"
        assert content[1]["type"] == "image_url"
        assert (
            content[1]["image_url"]["url"] == "data:image/jpeg;base64,base64_image_data"
        )

    @patch("requests.post")
    def test_chat_non_streaming(self, mock_post, azure_user):
        """Test chat with non-streaming response."""
        # Setup
        azure_user.on_start()

        # Mock response for non-streaming (still goes through iter_lines)
        mock_response = MagicMock()
        mock_response.status_code = 200
        # Azure sends non-streaming as a single data chunk
        mock_response.iter_lines.return_value = [
            b'data: {"choices":[{"message":{"content":"Non-streaming response"},'
            + b'"delta":{"content":"Non-streaming response"},"finish_reason":"stop"}]}',
            b"data: [DONE]",
        ]
        mock_post.return_value = mock_response

        # Create request with streaming disabled
        request = UserChatRequest(
            prompt="Test prompt",
            model="gpt-4",
            max_tokens=100,
            additional_request_params={"stream": False},
            num_prefill_tokens=10,
        )

        # Mock sample method
        azure_user.sample = MagicMock(return_value=request)
        azure_user.collect_metrics = MagicMock()

        # Execute
        azure_user.chat()

        # Verify
        body = mock_post.call_args.kwargs["json"]
        assert body["stream"] is False

        response = azure_user.collect_metrics.call_args[0][0]
        assert response.generated_text == "Non-streaming response"

    @patch("requests.post")
    def test_chat_with_system_message(self, mock_post, azure_user):
        """Test chat with system message."""
        # Setup
        azure_user.on_start()

        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = [
            b'data: {"choices":[{"delta":{"content":"Response"}}]}',
            b"data: [DONE]",
        ]
        mock_post.return_value = mock_response

        # Create request with system message
        request = UserChatRequest(
            prompt="Test prompt",
            model="gpt-4",
            max_tokens=100,
            additional_request_params={
                "messages": [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "Test prompt"},
                ]
            },
            num_prefill_tokens=10,
        )

        # Mock sample method
        azure_user.sample = MagicMock(return_value=request)
        azure_user.collect_metrics = MagicMock()

        # Execute
        azure_user.chat()

        # Verify
        body = mock_post.call_args.kwargs["json"]
        assert len(body["messages"]) == 2
        assert body["messages"][0]["role"] == "system"
        assert body["messages"][1]["role"] == "user"

    @patch("requests.post")
    def test_chat_error_handling(self, mock_post, azure_user):
        """Test chat error handling."""
        # Setup
        azure_user.on_start()
        mock_post.side_effect = requests.exceptions.RequestException("API Error")

        # Create request
        request = UserChatRequest(
            prompt="Test prompt",
            model="gpt-4",
            max_tokens=100,
            num_prefill_tokens=10,
        )

        # Mock sample method
        azure_user.sample = MagicMock(return_value=request)
        azure_user.collect_metrics = MagicMock()

        # Execute
        azure_user.chat()

        # Verify error response
        response = azure_user.collect_metrics.call_args[0][0]
        assert response.status_code == 500
        assert "API Error" in response.error_message

    @patch("requests.post")
    def test_chat_http_error(self, mock_post, azure_user):
        """Test chat with HTTP error response."""
        # Setup
        azure_user.on_start()

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = "Rate limit exceeded"
        mock_post.return_value = mock_response

        # Create request
        request = UserChatRequest(
            prompt="Test prompt",
            model="gpt-4",
            max_tokens=100,
            num_prefill_tokens=10,
        )

        # Mock sample method
        azure_user.sample = MagicMock(return_value=request)
        azure_user.collect_metrics = MagicMock()

        # Execute
        azure_user.chat()

        # Verify error response
        response = azure_user.collect_metrics.call_args[0][0]
        assert response.status_code == 429
        assert "Rate limit exceeded" in response.error_message

    @patch("requests.post")
    def test_embeddings_request(self, mock_post, azure_user):
        """Test embeddings request."""
        # Setup
        azure_user.on_start()
        azure_user.deployment_name = "text-embedding-ada-002"

        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"embedding": [0.1, 0.2, 0.3]},
                {"embedding": [0.4, 0.5, 0.6]},
            ],
            "usage": {"prompt_tokens": 10, "total_tokens": 10},
        }
        mock_post.return_value = mock_response

        # Create request
        request = UserEmbeddingRequest(
            documents=["Doc 1", "Doc 2"],
            model="text-embedding-ada-002",
            num_prefill_tokens=10,
        )

        # Mock sample method
        azure_user.sample = MagicMock(return_value=request)
        azure_user.collect_metrics = MagicMock()

        # Execute
        azure_user.embeddings()

        # Verify
        mock_post.assert_called_once()
        call_args = mock_post.call_args

        expected_url = (
            "https://test.openai.azure.com/openai/deployments/"
            "text-embedding-ada-002/embeddings?api-version=2024-02-15-preview"
        )
        assert call_args.kwargs["url"] == expected_url

        body = call_args.kwargs["json"]
        assert body["input"] == ["Doc 1", "Doc 2"]

        response = azure_user.collect_metrics.call_args[0][0]
        assert response.status_code == 200

    @patch("requests.post")
    def test_embeddings_error_handling(self, mock_post, azure_user):
        """Test embeddings error handling."""
        # Setup
        azure_user.on_start()
        mock_post.return_value.status_code = 400
        mock_post.return_value.text = "Bad request"

        # Create request
        request = UserEmbeddingRequest(
            documents=["Test"],
            model="text-embedding-ada-002",
            num_prefill_tokens=10,
        )

        # Mock sample method
        azure_user.sample = MagicMock(return_value=request)
        azure_user.collect_metrics = MagicMock()

        # Execute
        azure_user.embeddings()

        # Verify error response
        response = azure_user.collect_metrics.call_args[0][0]
        assert response.status_code == 400
        assert "Bad request" in response.error_message

    def test_parse_streaming_response(self, azure_user):
        """Test parsing streaming response."""
        mock_response = MagicMock()
        mock_response.iter_lines.return_value = [
            b'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            b'data: {"choices":[{"delta":{"content":" world"}}]}',
            b'data: {"choices":[{"delta":{"content":"!"}}]}',
            b"data: [DONE]",
        ]

        result = azure_user.parse_chat_response(mock_response, 0.0, 10, 1.0)

        assert result.generated_text == "Hello world!"
        assert result.status_code == 200

    def test_parse_streaming_response_empty_delta(self, azure_user):
        """Test parsing streaming response with empty delta."""
        mock_response = MagicMock()
        mock_response.iter_lines.return_value = [
            b'data: {"choices":[{"delta":{}}]}',
            b'data: {"choices":[{"delta":{"content":"Text"}}]}',
            b"data: [DONE]",
        ]

        result = azure_user.parse_chat_response(mock_response, 0.0, 10, 1.0)

        assert result.generated_text == "Text"

    def test_parse_streaming_response_error(self, azure_user):
        """Test parsing streaming response with error."""
        mock_response = MagicMock()
        # Return error response in stream format
        mock_response.iter_lines.return_value = [
            b'data: {"error":{"code":500,"message":"Stream error"}}',
        ]

        result = azure_user.parse_chat_response(mock_response, 0.0, 10, 1.0)

        assert result.status_code == 500
        assert "Stream error" in result.error_message

    def test_parse_non_streaming_response(self, azure_user):
        """Test parsing non-streaming response."""
        mock_response = MagicMock()
        mock_response.iter_lines.return_value = [
            b'data: {"choices":[{"message":{"content":"Response text"},'
            + b'"finish_reason":"stop"}]}',
        ]

        result = azure_user.parse_chat_response(mock_response, 0.0, 10, 1.0)

        # Non-streaming still uses the streaming parser but gets full message at once
        assert result.generated_text == ""  # The parser expects delta format
        assert result.status_code == 200

    def test_parse_non_streaming_response_error(self, azure_user):
        """Test parsing non-streaming response with error."""
        mock_response = MagicMock()
        mock_response.iter_lines.return_value = [
            b'data: {"error":{"code":400,"message":"Parse error"}}',
        ]

        result = azure_user.parse_chat_response(mock_response, 0.0, 10, 1.0)

        assert result.status_code == 400
        assert "Parse error" in result.error_message

    def test_parse_embedding_response(self, azure_user):
        """Test parsing embedding response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {"embedding": [0.1, 0.2]},
                {"embedding": [0.3, 0.4]},
            ],
            "usage": {"prompt_tokens": 10, "total_tokens": 10},
        }

        result = azure_user.parse_embedding_response(mock_response, 0.0, None, 1.0)

        assert result.status_code == 200
        assert result.num_prefill_tokens == 10

    def test_prepare_request_body_additional_params(self, azure_user):
        """Test prepare request body with additional params."""
        request = UserChatRequest(
            prompt="Test",
            model="gpt-4",
            max_tokens=100,
            additional_request_params={
                "top_p": 0.9,
                "frequency_penalty": 0.5,
                "presence_penalty": 0.5,
            },
            num_prefill_tokens=10,
        )

        body = azure_user._prepare_chat_request(request)

        assert body["top_p"] == 0.9
        assert body["frequency_penalty"] == 0.5
        assert body["presence_penalty"] == 0.5

    def test_prepare_request_body_with_tools(self, azure_user):
        """Test prepare request body with tools."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {},
                },
            }
        ]

        request = UserChatRequest(
            prompt="Test",
            model="gpt-4",
            max_tokens=100,
            additional_request_params={"tools": tools},
            num_prefill_tokens=10,
        )

        body = azure_user._prepare_chat_request(request)

        assert body["tools"] == tools

    def test_chat_wrong_request_type(self, azure_user):
        """Test chat with wrong request type."""
        request = UserEmbeddingRequest(
            documents=["Test"],
            model="test",
            num_prefill_tokens=10,
        )
        azure_user.sample = MagicMock(return_value=request)

        with pytest.raises(AttributeError, match="should be of type UserChatRequest"):
            azure_user.chat()

    def test_embeddings_wrong_request_type(self, azure_user):
        """Test embeddings with wrong request type."""
        request = UserChatRequest(
            prompt="Test",
            model="test",
            max_tokens=100,
            num_prefill_tokens=10,
        )
        azure_user.sample = MagicMock(return_value=request)

        with pytest.raises(
            AttributeError, match="should be of type UserEmbeddingRequest"
        ):
            azure_user.embeddings()

    @patch("requests.post")
    def test_chat_with_n_parameter(self, mock_post, azure_user):
        """Test chat with n parameter for multiple completions."""
        # Setup
        azure_user.on_start()

        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = [
            b'data: {"choices":[{"message":{"content":"Response 1"},'
            + b'"delta":{"content":"Response 1"},"finish_reason":"stop"},'
            + b'{"message":{"content":"Response 2"},"delta":{"content":"Response 2"},'
            + b'"finish_reason":"stop"}]}',
            b"data: [DONE]",
        ]
        mock_post.return_value = mock_response

        # Create request
        request = UserChatRequest(
            prompt="Test prompt",
            model="gpt-4",
            max_tokens=100,
            additional_request_params={"n": 2, "stream": False},
            num_prefill_tokens=10,
        )

        # Mock sample method
        azure_user.sample = MagicMock(return_value=request)
        azure_user.collect_metrics = MagicMock()

        # Execute
        azure_user.chat()

        # Verify
        body = mock_post.call_args.kwargs["json"]
        assert body["n"] == 2

        response = azure_user.collect_metrics.call_args[0][0]
        # Should return first choice
        assert response.generated_text == "Response 1"

    def test_api_version_override(self, azure_user):
        """Test API version override from config."""
        azure_user.auth_provider.get_config.return_value = {
            "api_base": "https://test.openai.azure.com",
            "api_version": "2023-05-15",
            "deployment_id": "test-deployment",
        }

        azure_user.on_start()

        assert azure_user.api_version == "2023-05-15"

    def test_deployment_id_from_model_config(self, azure_user):
        """Test deployment ID from model config."""
        azure_user.auth_provider.get_config.return_value = {
            "azure_endpoint": "https://test.openai.azure.com",
            "api_version": "2024-02-15-preview",
        }
        azure_user.config.model = "gpt-4-deployment"

        azure_user.on_start()

        assert (
            azure_user.deployment_name is None
        )  # deployment_name only comes from auth config

    @patch("requests.post")
    def test_chat_connection_error(self, mock_post, azure_user):
        """Test chat with connection error."""
        # Setup
        azure_user.on_start()
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection failed")

        # Create request
        request = UserChatRequest(
            prompt="Test prompt",
            model="gpt-4",
            max_tokens=100,
            num_prefill_tokens=10,
        )

        # Mock sample method
        azure_user.sample = MagicMock(return_value=request)
        azure_user.collect_metrics = MagicMock()

        # Execute
        azure_user.chat()

        # Verify error response
        response = azure_user.collect_metrics.call_args[0][0]
        assert response.status_code == 503
        assert "Connection error" in response.error_message

    @patch("requests.post")
    def test_chat_timeout_error(self, mock_post, azure_user):
        """Test chat with timeout error."""
        # Setup
        azure_user.on_start()
        mock_post.side_effect = requests.exceptions.Timeout("Request timed out")

        # Create request
        request = UserChatRequest(
            prompt="Test prompt",
            model="gpt-4",
            max_tokens=100,
            num_prefill_tokens=10,
        )

        # Mock sample method
        azure_user.sample = MagicMock(return_value=request)
        azure_user.collect_metrics = MagicMock()

        # Execute
        azure_user.chat()

        # Verify error response
        response = azure_user.collect_metrics.call_args[0][0]
        assert response.status_code == 408
        assert "Request timed out" in response.error_message

    def test_parse_streaming_response_empty_chunk(self, azure_user):
        """Test parsing streaming response with empty chunk."""
        mock_response = MagicMock()
        mock_response.iter_lines.return_value = [
            b"",  # Empty chunk
            b'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            b"data: [DONE]",
        ]

        result = azure_user.parse_chat_response(mock_response, 0.0, 10, 1.0)

        assert result.generated_text == "Hello"

    def test_parse_streaming_response_with_usage_in_chunk(self, azure_user):
        """Test parsing streaming response with usage info in chunk."""
        mock_response = MagicMock()
        mock_response.iter_lines.return_value = [
            b'data: {"choices":[{"delta":{"content":"Hello",'
            + b'"usage":{"completion_tokens":5}}}]}',
            b'data: {"choices":[{"delta":{"content":" world"}}]}',
            b"data: [DONE]",
        ]

        result = azure_user.parse_chat_response(mock_response, 0.0, 10, 1.0)

        assert result.generated_text == "Hello world"
        assert result.tokens_received == 5

    def test_parse_streaming_response_with_usage_in_last_chunk(self, azure_user):
        """Test parsing streaming response with usage info in last chunk."""
        mock_response = MagicMock()
        mock_response.iter_lines.return_value = [
            b'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            b'data: {"choices":[{"delta":{"content":" world"},'
            + b'"finish_reason":"stop"}],"usage":{"prompt_tokens":10,'
            + b'"completion_tokens":2}}',
            b"data: [DONE]",
        ]

        result = azure_user.parse_chat_response(mock_response, 0.0, 10, 1.0)

        assert result.generated_text == "Hello world"
        assert result.tokens_received == 2

    def test_parse_streaming_response_with_first_token_warning(self, azure_user):
        """Test parsing streaming response with large first token chunk."""
        mock_response = MagicMock()
        mock_response.iter_lines.return_value = [
            b'data: {"choices":[{"delta":{"content":'
            + b'"Hello world this is a long first chunk",'
            + b'"usage":{"completion_tokens":10}}}]}',
            b"data: [DONE]",
        ]

        with patch("genai_bench.user.azure_openai_user.logger") as mock_logger:
            result = azure_user.parse_chat_response(mock_response, 0.0, 10, 1.0)

            # Check warning was logged
            mock_logger.warning.assert_called_once()
            assert "has >1 tokens" in mock_logger.warning.call_args[0][0]

        assert result.generated_text == "Hello world this is a long first chunk"

    def test_parse_streaming_response_with_finish_reason_and_usage(self, azure_user):
        """Test parsing streaming response with finish reason and usage."""
        mock_response = MagicMock()
        mock_response.iter_lines.return_value = [
            b'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            b'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":1}}',
            b"data: [DONE]",
        ]

        # Set finish_reason in the parser
        result = azure_user.parse_chat_response(mock_response, 0.0, 10, 1.0)

        assert result.generated_text == "Hello"

    def test_parse_streaming_response_reasoning_tokens_from_usage(self, azure_user):
        """Test reasoning_tokens from completion_tokens_details is passed through."""
        mock_response = MagicMock()
        mock_response.iter_lines.return_value = [
            b'data: {"choices":[{"delta":{"content":"Hi"}}]}',
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
            b'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":2,'
            b'"completion_tokens_details":{"reasoning_tokens":5}}}',
            b"data: [DONE]",
        ]
        result = azure_user.parse_chat_response(mock_response, 0.0, 10, 1.0)
        assert result.generated_text == "Hi"
        assert result.tokens_received == 2
        assert result.reasoning_tokens == 5

    def test_get_usage_info_for_vision_task(self, azure_user):
        """Test _get_usage_info with None prefill tokens (vision task)."""
        data = {"usage": {"prompt_tokens": 100, "completion_tokens": 50}}

        num_prefill, num_prompt, tokens_received, reasoning_tokens = (
            azure_user._get_usage_info(data, None)
        )

        assert num_prefill == 100  # Uses prompt tokens for vision
        assert num_prompt == 100
        assert tokens_received == 50
        assert reasoning_tokens == 0

    def test_get_usage_info_prefers_server_tokens(self, azure_user):
        """Test _get_usage_info prefers server-reported prompt tokens."""
        data = {"usage": {"prompt_tokens": 100, "completion_tokens": 50}}

        (
            num_prefill,
            num_prompt,
            tokens_received,
            reasoning_tokens,
        ) = azure_user._get_usage_info(data, 40)

        assert num_prefill == 100  # Prefers server-reported prompt tokens
        assert num_prompt == 100
        assert tokens_received == 50
        assert reasoning_tokens == 0

    def test_parse_streaming_response_with_index_error(self, azure_user):
        """Test parsing streaming response with malformed data causing IndexError."""
        mock_response = MagicMock()
        mock_response.iter_lines.return_value = [
            b'data: {"choices":[]}',  # Empty choices array
            b'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            b"data: [DONE]",
        ]

        result = azure_user.parse_chat_response(mock_response, 0.0, 10, 1.0)

        # The empty choices array is handled gracefully, no error is raised
        assert result.generated_text == "Hello"

    def test_parse_streaming_response_with_key_error(self, azure_user):
        """Test parsing streaming response with missing keys causing KeyError."""
        mock_response = MagicMock()
        mock_response.iter_lines.return_value = [
            b'data: {"choices":[{"other_key":"value"}]}',  # Missing 'delta' key
            b'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            b"data: [DONE]",
        ]

        result = azure_user.parse_chat_response(mock_response, 0.0, 10, 1.0)

        # The parser handles missing keys gracefully
        assert result.generated_text == "Hello"

    def test_parse_streaming_response_no_tokens_received(self, azure_user):
        """Test parsing streaming response with no token count."""
        mock_response = MagicMock()
        mock_response.iter_lines.return_value = [
            b'data: {"choices":[{"delta":{"content":"Hello world"}}]}',
            b"data: [DONE]",
        ]

        with patch("genai_bench.user.azure_openai_user.logger") as mock_logger:
            result = azure_user.parse_chat_response(mock_response, 0.0, 10, 1.0)

            # Check warning was logged about missing usage info
            mock_logger.warning.assert_called()
            assert "no usage info returned" in mock_logger.warning.call_args[0][0]

        assert result.generated_text == "Hello world"
        assert result.tokens_received == 50  # From mock sampler

    @patch("requests.post")
    def test_embeddings_connection_error(self, mock_post, azure_user):
        """Test embeddings with connection error."""
        # Setup
        azure_user.on_start()
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection failed")

        # Create request
        request = UserEmbeddingRequest(
            documents=["Test"],
            model="text-embedding-ada-002",
            num_prefill_tokens=10,
        )

        # Mock sample method
        azure_user.sample = MagicMock(return_value=request)
        azure_user.collect_metrics = MagicMock()

        # Execute
        azure_user.embeddings()

        # Verify error response
        response = azure_user.collect_metrics.call_args[0][0]
        assert response.status_code == 503
        assert "Connection error" in response.error_message

    @patch("requests.post")
    def test_embeddings_timeout_error(self, mock_post, azure_user):
        """Test embeddings with timeout error."""
        # Setup
        azure_user.on_start()
        mock_post.side_effect = requests.exceptions.Timeout("Request timed out")

        # Create request
        request = UserEmbeddingRequest(
            documents=["Test"],
            model="text-embedding-ada-002",
            num_prefill_tokens=10,
        )

        # Mock sample method
        azure_user.sample = MagicMock(return_value=request)
        azure_user.collect_metrics = MagicMock()

        # Execute
        azure_user.embeddings()

        # Verify error response
        response = azure_user.collect_metrics.call_args[0][0]
        assert response.status_code == 408
        assert "Request timed out" in response.error_message

    def test_parse_streaming_response_with_finish_and_usage_no_choices(
        self, azure_user
    ):
        """Test parsing streaming response with finish_reason and usage."""
        mock_response = MagicMock()
        mock_response.iter_lines.return_value = [
            b'data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":"stop"}]}',
            # Empty choices array with usage data
            b'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":1}}',
            b"data: [DONE]",
        ]

        result = azure_user.parse_chat_response(mock_response, 0.0, 10, 1.0)

        assert result.generated_text == "Hello"
        assert result.tokens_received == 1
