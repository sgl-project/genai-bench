"""Tests for AWS Bedrock user implementation."""

import json
from unittest.mock import MagicMock, patch

import pytest

from genai_bench.auth.model_auth_provider import ModelAuthProvider
from genai_bench.protocol import (
    UserChatRequest,
    UserChatResponse,
    UserEmbeddingRequest,
    UserImageChatRequest,
)
from genai_bench.user.aws_bedrock_user import AWSBedrockUser


class TestAWSBedrockUser:
    """Test AWS Bedrock user implementation."""

    @pytest.fixture
    def mock_auth(self):
        """Create mock auth provider."""
        auth = MagicMock(spec=ModelAuthProvider)
        auth.get_credentials.return_value = {
            "aws_access_key_id": "test_key",
            "aws_secret_access_key": "test_secret",
            "region_name": "us-east-1",
        }
        return auth

    @pytest.fixture
    def bedrock_user(self, mock_auth):
        """Create AWS Bedrock user instance."""
        env = MagicMock()
        env.sampler = MagicMock()
        env.sampler.get_token_length.return_value = 50
        AWSBedrockUser.host = "http://localhost"
        user = AWSBedrockUser(environment=env)
        user.auth_provider = mock_auth
        user.config = MagicMock()
        user.config.model = "anthropic.claude-v2"
        return user

    def test_backend_name(self):
        """Test backend name constant."""
        assert AWSBedrockUser.BACKEND_NAME == "aws-bedrock"

    def test_supported_tasks(self):
        """Test supported tasks mapping."""
        assert AWSBedrockUser.supported_tasks == {
            "text-to-text": "chat",
            "text-to-embeddings": "embeddings",
            "image-text-to-text": "chat",
        }

    def test_init(self):
        """Test initialization."""
        AWSBedrockUser.host = "http://localhost"
        user = AWSBedrockUser(environment=MagicMock())
        # client comes from Locust base class
        assert hasattr(user, "client")
        assert user.runtime_client is None

    @patch("boto3.Session")
    def test_on_start_with_credentials(self, mock_session, bedrock_user):
        """Test on_start with credentials."""
        mock_client = MagicMock()
        mock_session.return_value.client.return_value = mock_client

        bedrock_user.on_start()

        mock_session.assert_called_once_with(
            aws_access_key_id="test_key",
            aws_secret_access_key="test_secret",
        )
        mock_session.return_value.client.assert_called_once_with(
            service_name="bedrock-runtime",
            config=mock_session.return_value.client.call_args[1]["config"],
        )
        assert bedrock_user.runtime_client == mock_client

    @patch("boto3.Session")
    def test_on_start_with_profile(self, mock_session, bedrock_user):
        """Test on_start with profile."""
        bedrock_user.auth_provider.get_credentials.return_value = {
            "profile_name": "test-profile",
            "region_name": "us-west-2",
        }

        mock_client = MagicMock()
        mock_session.return_value.client.return_value = mock_client

        bedrock_user.on_start()

        mock_session.assert_called_once_with(profile_name="test-profile")

    def test_on_start_no_auth(self, bedrock_user):
        """Test on_start without auth provider."""
        bedrock_user.auth_provider = None

        with pytest.raises(ValueError, match="Auth provider not set"):
            bedrock_user.on_start()

    def test_on_start_import_error(self, bedrock_user):
        """Test on_start with boto3 import error."""
        with patch.dict("sys.modules", {"boto3": None}):
            with pytest.raises(ImportError, match="boto3 is required"):
                bedrock_user.on_start()

    @patch("boto3.Session")
    def test_chat_text_request(self, mock_session, bedrock_user):
        """Test chat with text-only request."""
        # Setup
        bedrock_user.on_start()
        mock_client = bedrock_user.runtime_client

        # Mock streaming response for Claude
        mock_event = {
            "chunk": {
                "bytes": json.dumps(
                    {"type": "content_block_delta", "delta": {"text": "Test response"}}
                ).encode()
            }
        }
        mock_response = {"body": [mock_event]}
        mock_client.invoke_model_with_response_stream.return_value = mock_response

        # Create request
        request = UserChatRequest(
            prompt="Test prompt",
            model="anthropic.claude-v2",
            max_tokens=100,
            temperature=0.7,
            num_prefill_tokens=10,
        )

        # Mock sample method
        bedrock_user.sample = MagicMock(return_value=request)
        bedrock_user.collect_metrics = MagicMock()

        # Execute
        bedrock_user.chat()

        # Verify
        mock_client.invoke_model_with_response_stream.assert_called_once()
        call_args = mock_client.invoke_model_with_response_stream.call_args
        assert call_args[1]["modelId"] == "anthropic.claude-v2"

        body = json.loads(call_args[1]["body"])
        assert body["messages"][0]["content"] == "Test prompt"
        assert body["max_tokens"] == 100

        bedrock_user.collect_metrics.assert_called_once()
        response = bedrock_user.collect_metrics.call_args[0][0]
        assert isinstance(response, UserChatResponse)
        assert response.status_code == 200
        assert response.generated_text == "Test response"

    @patch("boto3.Session")
    def test_chat_image_request(self, mock_session, bedrock_user):
        """Test chat with image request."""
        # Setup
        bedrock_user.on_start()
        mock_client = bedrock_user.runtime_client

        # Mock streaming response for Claude
        mock_event = {
            "chunk": {
                "bytes": json.dumps(
                    {
                        "type": "content_block_delta",
                        "delta": {"text": "Image description"},
                    }
                ).encode()
            }
        }
        mock_response = {"body": [mock_event]}
        mock_client.invoke_model_with_response_stream.return_value = mock_response

        # Create request
        request = UserImageChatRequest(
            prompt="Describe this image",
            model="anthropic.claude-v2",
            max_tokens=100,
            image_content=["data:image/jpeg;base64,base64_image_data"],
            num_images=1,
            num_prefill_tokens=10,
        )

        # Mock sample method
        bedrock_user.sample = MagicMock(return_value=request)
        bedrock_user.collect_metrics = MagicMock()

        # Execute
        bedrock_user.chat()

        # Verify
        call_args = mock_client.invoke_model_with_response_stream.call_args
        body = json.loads(call_args[1]["body"])

        content = body["messages"][0]["content"]
        assert len(content) == 2
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "Describe this image"
        assert content[1]["type"] == "image"
        assert content[1]["source"]["data"] == "base64_image_data"

    @patch("boto3.Session")
    def test_chat_streaming(self, mock_session, bedrock_user):
        """Test chat with streaming response."""
        # Setup
        bedrock_user.on_start()
        mock_client = bedrock_user.runtime_client

        # Mock streaming response
        mock_event1 = {
            "chunk": {
                "bytes": json.dumps(
                    {"type": "content_block_delta", "delta": {"text": "Hello "}}
                ).encode()
            }
        }
        mock_event2 = {
            "chunk": {
                "bytes": json.dumps(
                    {"type": "content_block_delta", "delta": {"text": "world!"}}
                ).encode()
            }
        }

        mock_response = {"body": [mock_event1, mock_event2]}
        mock_client.invoke_model_with_response_stream.return_value = mock_response

        # Create request
        request = UserChatRequest(
            prompt="Test prompt",
            model="anthropic.claude-v2",
            max_tokens=100,
            additional_request_params={"stream": True},
            num_prefill_tokens=10,
        )

        # Mock sample method
        bedrock_user.sample = MagicMock(return_value=request)
        bedrock_user.collect_metrics = MagicMock()

        # Execute
        bedrock_user.chat()

        # Verify
        mock_client.invoke_model_with_response_stream.assert_called_once()

        response = bedrock_user.collect_metrics.call_args[0][0]
        assert response.generated_text == "Hello world!"
        assert response.reasoning_tokens == 0

    @patch("boto3.Session")
    def test_chat_titan_model(self, mock_session, bedrock_user):
        """Test chat with Titan model."""
        # Setup
        bedrock_user.on_start()
        bedrock_user.config.model = "amazon.titan-text-express"
        mock_client = bedrock_user.runtime_client

        # Mock streaming response for Titan
        mock_event = {
            "chunk": {"bytes": json.dumps({"outputText": "Titan response"}).encode()}
        }
        mock_response = {"body": [mock_event]}
        mock_client.invoke_model_with_response_stream.return_value = mock_response

        # Create request
        request = UserChatRequest(
            prompt="Test prompt",
            model="amazon.titan-text-express",
            max_tokens=100,
            additional_request_params={
                "temperature": 0.5,
                "top_p": 0.9,
            },
            num_prefill_tokens=10,
        )

        # Mock sample method
        bedrock_user.sample = MagicMock(return_value=request)
        bedrock_user.collect_metrics = MagicMock()

        # Execute
        bedrock_user.chat()

        # Verify
        call_args = mock_client.invoke_model_with_response_stream.call_args
        body = json.loads(call_args[1]["body"])

        assert body["inputText"] == "Test prompt"
        assert body["textGenerationConfig"]["maxTokenCount"] == 100
        assert body["textGenerationConfig"]["temperature"] == 0.5
        assert body["textGenerationConfig"]["topP"] == 0.9

    @patch("boto3.Session")
    def test_chat_llama_model(self, mock_session, bedrock_user):
        """Test chat with Llama model."""
        # Setup
        bedrock_user.on_start()
        bedrock_user.config.model = "meta.llama2-70b"
        mock_client = bedrock_user.runtime_client

        # Mock streaming response for Llama
        mock_event = {
            "chunk": {"bytes": json.dumps({"generation": "Llama response"}).encode()}
        }
        mock_response = {"body": [mock_event]}
        mock_client.invoke_model_with_response_stream.return_value = mock_response

        # Create request
        request = UserChatRequest(
            prompt="Test prompt",
            model="meta.llama2-70b",
            max_tokens=100,
            num_prefill_tokens=10,
        )

        # Mock sample method
        bedrock_user.sample = MagicMock(return_value=request)
        bedrock_user.collect_metrics = MagicMock()

        # Execute
        bedrock_user.chat()

        # Verify
        call_args = mock_client.invoke_model_with_response_stream.call_args
        body = json.loads(call_args[1]["body"])

        assert body["prompt"] == "Test prompt"
        assert body["max_gen_len"] == 100

    @patch("boto3.Session")
    def test_chat_error_handling(self, mock_session, bedrock_user):
        """Test chat error handling."""
        # Setup
        bedrock_user.on_start()
        mock_client = bedrock_user.runtime_client
        mock_client.invoke_model_with_response_stream.side_effect = Exception(
            "API Error"
        )

        # Create request
        request = UserChatRequest(
            prompt="Test prompt",
            model="anthropic.claude-v2",
            max_tokens=100,
            num_prefill_tokens=10,
        )

        # Mock sample method
        bedrock_user.sample = MagicMock(return_value=request)
        bedrock_user.collect_metrics = MagicMock()

        # Execute
        bedrock_user.chat()

        # Verify error response
        response = bedrock_user.collect_metrics.call_args[0][0]
        assert response.status_code == 500
        assert "API Error" in response.error_message

    @patch("boto3.Session")
    def test_embeddings_titan_model(self, mock_session, bedrock_user):
        """Test embeddings with Titan model."""
        # Setup
        bedrock_user.on_start()
        bedrock_user.config.model = "amazon.titan-embed-text-v1"
        mock_client = bedrock_user.runtime_client

        # Mock response
        mock_response = {
            "body": MagicMock(
                read=lambda: json.dumps({"embedding": [0.1, 0.2, 0.3]}).encode()
            )
        }
        mock_client.invoke_model.return_value = mock_response

        # Create request
        request = UserEmbeddingRequest(
            documents=["Test document"],
            model="amazon.titan-embed-text-v1",
            num_prefill_tokens=10,
        )

        # Mock sample method
        bedrock_user.sample = MagicMock(return_value=request)
        bedrock_user.collect_metrics = MagicMock()

        # Execute
        bedrock_user.embeddings()

        # Verify
        call_args = mock_client.invoke_model.call_args
        body = json.loads(call_args[1]["body"])

        assert body["inputText"] == "Test document"

    @patch("boto3.Session")
    def test_embeddings_unsupported_model(self, mock_session, bedrock_user):
        """Test embeddings with unsupported model."""
        # Setup
        bedrock_user.on_start()
        bedrock_user.config.model = "anthropic.claude-v2"

        # Create request
        request = UserEmbeddingRequest(
            documents=["Test document"],
            model="anthropic.claude-v2",
            num_prefill_tokens=10,
        )

        # Mock sample method
        bedrock_user.sample = MagicMock(return_value=request)
        bedrock_user.collect_metrics = MagicMock()

        # Execute
        bedrock_user.embeddings()

        # Verify error response
        response = bedrock_user.collect_metrics.call_args[0][0]
        assert response.status_code == 400
        assert "does not support embeddings" in response.error_message

    def test_supports_streaming_claude(self, bedrock_user):
        """Test streaming support for Claude models."""
        assert bedrock_user._supports_streaming("anthropic.claude-v2") is True

    def test_supports_streaming_embeddings(self, bedrock_user):
        """Test streaming support for embeddings models."""
        assert bedrock_user._supports_streaming("amazon.titan-embed-v1") is False

    def test_extract_chunk_text_claude(self, bedrock_user):
        """Test extracting text from Claude streaming chunk."""
        chunk = {"type": "content_block_delta", "delta": {"text": "Hello"}}
        assert bedrock_user._extract_chunk_text(chunk, "anthropic.claude") == "Hello"

    def test_extract_chunk_text_titan(self, bedrock_user):
        """Test extracting text from Titan streaming chunk."""
        chunk = {"outputText": "Hello"}
        assert bedrock_user._extract_chunk_text(chunk, "amazon.titan") == "Hello"

    def test_extract_chunk_text_llama(self, bedrock_user):
        """Test extracting text from Llama streaming chunk."""
        chunk = {"generation": "Hello"}
        assert bedrock_user._extract_chunk_text(chunk, "meta.llama") == "Hello"

    def test_extract_response_text_claude(self, bedrock_user):
        """Test extracting text from Claude response."""
        response = {"content": [{"text": "Response"}]}
        assert (
            bedrock_user._extract_response_text(response, "anthropic.claude")
            == "Response"
        )

    def test_extract_response_text_titan(self, bedrock_user):
        """Test extracting text from Titan response."""
        response = {"results": [{"outputText": "Response"}]}
        assert (
            bedrock_user._extract_response_text(response, "amazon.titan") == "Response"
        )

    def test_extract_response_text_llama(self, bedrock_user):
        """Test extracting text from Llama response."""
        response = {"generation": "Response"}
        assert bedrock_user._extract_response_text(response, "meta.llama") == "Response"

    def test_extract_response_text_generic(self, bedrock_user):
        """Test extracting text from generic response."""
        response = {"text": "Response"}
        assert (
            bedrock_user._extract_response_text(response, "unknown.model") == "Response"
        )

    def test_chat_wrong_request_type(self, bedrock_user):
        """Test chat with wrong request type."""
        request = UserEmbeddingRequest(
            documents=["Test"],
            model="test",
            num_prefill_tokens=10,
        )
        bedrock_user.sample = MagicMock(return_value=request)

        with pytest.raises(AttributeError, match="should be of type UserChatRequest"):
            bedrock_user.chat()

    def test_embeddings_wrong_request_type(self, bedrock_user):
        """Test embeddings with wrong request type."""
        request = UserChatRequest(
            prompt="Test",
            model="test",
            max_tokens=100,
            num_prefill_tokens=10,
        )
        bedrock_user.sample = MagicMock(return_value=request)

        with pytest.raises(
            AttributeError, match="should be of type UserEmbeddingRequest"
        ):
            bedrock_user.embeddings()

    @patch("boto3.Session")
    def test_on_start_with_session_token(self, mock_session, bedrock_user):
        """Test on_start with session token."""
        bedrock_user.auth_provider.get_credentials.return_value = {
            "aws_access_key_id": "test_key",
            "aws_secret_access_key": "test_secret",
            "aws_session_token": "test_token",
            "region_name": "us-east-1",
        }

        mock_client = MagicMock()
        mock_session.return_value.client.return_value = mock_client

        bedrock_user.on_start()

        mock_session.assert_called_once_with(
            aws_access_key_id="test_key",
            aws_secret_access_key="test_secret",
            aws_session_token="test_token",
        )

    @patch("boto3.Session")
    def test_chat_non_streaming_request(self, mock_session, bedrock_user):
        """Test chat with non-streaming request."""
        # Setup
        bedrock_user.on_start()
        mock_client = bedrock_user.runtime_client

        # Mock non-streaming response
        mock_response = {
            "body": MagicMock(
                read=lambda: json.dumps(
                    {"content": [{"text": "Non-streaming response"}]}
                ).encode()
            )
        }
        mock_client.invoke_model.return_value = mock_response

        # Create request with stream=False
        request = UserChatRequest(
            prompt="Test prompt",
            model="anthropic.claude-v2",
            max_tokens=100,
            additional_request_params={"stream": False},
            num_prefill_tokens=10,
        )

        # Mock sample method
        bedrock_user.sample = MagicMock(return_value=request)
        bedrock_user.collect_metrics = MagicMock()

        # Execute
        bedrock_user.chat()

        # Verify
        mock_client.invoke_model.assert_called_once()
        mock_client.invoke_model_with_response_stream.assert_not_called()

        response = bedrock_user.collect_metrics.call_args[0][0]
        assert response.generated_text == "Non-streaming response"
        assert response.reasoning_tokens == 0

    @patch("boto3.Session")
    def test_chat_llama_with_temperature_top_p(self, mock_session, bedrock_user):
        """Test chat with Llama model with temperature and top_p."""
        # Setup
        bedrock_user.on_start()
        bedrock_user.config.model = "meta.llama2-70b"
        mock_client = bedrock_user.runtime_client

        # Mock streaming response
        mock_event = {
            "chunk": {"bytes": json.dumps({"generation": "Llama response"}).encode()}
        }
        mock_response = {"body": [mock_event]}
        mock_client.invoke_model_with_response_stream.return_value = mock_response

        # Create request with temperature and top_p
        request = UserChatRequest(
            prompt="Test prompt",
            model="meta.llama2-70b",
            max_tokens=100,
            additional_request_params={"temperature": 0.8, "top_p": 0.95},
            num_prefill_tokens=10,
        )

        # Mock sample method
        bedrock_user.sample = MagicMock(return_value=request)
        bedrock_user.collect_metrics = MagicMock()

        # Execute
        bedrock_user.chat()

        # Verify
        call_args = mock_client.invoke_model_with_response_stream.call_args
        body = json.loads(call_args[1]["body"])

        assert body["temperature"] == 0.8
        assert body["top_p"] == 0.95

    @patch("boto3.Session")
    def test_chat_generic_model(self, mock_session, bedrock_user):
        """Test chat with generic model format."""
        # Setup
        bedrock_user.on_start()
        bedrock_user.config.model = "unknown.model-v1"
        mock_client = bedrock_user.runtime_client

        # Mock streaming response
        mock_event = {
            "chunk": {"bytes": json.dumps({"text": "Generic response"}).encode()}
        }
        mock_response = {"body": [mock_event]}
        mock_client.invoke_model_with_response_stream.return_value = mock_response

        # Create request
        request = UserChatRequest(
            prompt="Test prompt",
            model="unknown.model-v1",
            max_tokens=100,
            additional_request_params={"temperature": 0.7, "top_p": 0.9},
            num_prefill_tokens=10,
        )

        # Mock sample method
        bedrock_user.sample = MagicMock(return_value=request)
        bedrock_user.collect_metrics = MagicMock()

        # Execute
        bedrock_user.chat()

        # Verify
        call_args = mock_client.invoke_model_with_response_stream.call_args
        body = json.loads(call_args[1]["body"])

        assert body["prompt"] == "Test prompt"
        assert body["max_tokens"] == 100
        assert body["temperature"] == 0.7
        assert body["top_p"] == 0.9

    @patch("boto3.Session")
    def test_embeddings_error_handling(self, mock_session, bedrock_user):
        """Test embeddings error handling."""
        # Setup
        bedrock_user.on_start()
        bedrock_user.config.model = "amazon.titan-embed-text-v1"
        mock_client = bedrock_user.runtime_client
        mock_client.invoke_model.side_effect = Exception("Embeddings API Error")

        # Create request
        request = UserEmbeddingRequest(
            documents=["Test document"],
            model="amazon.titan-embed-text-v1",
            num_prefill_tokens=10,
        )

        # Mock sample method
        bedrock_user.sample = MagicMock(return_value=request)
        bedrock_user.collect_metrics = MagicMock()

        # Execute
        bedrock_user.embeddings()

        # Verify error response
        response = bedrock_user.collect_metrics.call_args[0][0]
        assert response.status_code == 500
        assert "Embeddings API Error" in response.error_message

    def test_extract_chunk_text_generic_fallback(self, bedrock_user):
        """Test extracting text from generic chunk with fallback."""
        # Test with 'output' field
        chunk = {"output": "Generic output"}
        assert (
            bedrock_user._extract_chunk_text(chunk, "unknown.model") == "Generic output"
        )

        # Test with no recognized fields
        chunk = {"data": "Some data"}
        assert bedrock_user._extract_chunk_text(chunk, "unknown.model") == ""

    @patch("boto3.Session")
    def test_chat_claude_with_temperature_only(self, mock_session, bedrock_user):
        """Test chat with Claude model with only temperature."""
        # Setup
        bedrock_user.on_start()
        mock_client = bedrock_user.runtime_client

        # Mock streaming response
        mock_event = {
            "chunk": {
                "bytes": json.dumps(
                    {"type": "content_block_delta", "delta": {"text": "Response"}}
                ).encode()
            }
        }
        mock_response = {"body": [mock_event]}
        mock_client.invoke_model_with_response_stream.return_value = mock_response

        # Create request with only temperature
        request = UserChatRequest(
            prompt="Test prompt",
            model="anthropic.claude-v2",
            max_tokens=100,
            additional_request_params={"temperature": 0.5},
            num_prefill_tokens=10,
        )

        # Mock sample method
        bedrock_user.sample = MagicMock(return_value=request)
        bedrock_user.collect_metrics = MagicMock()

        # Execute
        bedrock_user.chat()

        # Verify
        call_args = mock_client.invoke_model_with_response_stream.call_args
        body = json.loads(call_args[1]["body"])

        assert body["temperature"] == 0.5
        assert "top_p" not in body

    @patch("boto3.Session")
    def test_chat_claude_with_top_p_only(self, mock_session, bedrock_user):
        """Test chat with Claude model with only top_p."""
        # Setup
        bedrock_user.on_start()
        mock_client = bedrock_user.runtime_client

        # Mock streaming response
        mock_event = {
            "chunk": {
                "bytes": json.dumps(
                    {"type": "content_block_delta", "delta": {"text": "Response"}}
                ).encode()
            }
        }
        mock_response = {"body": [mock_event]}
        mock_client.invoke_model_with_response_stream.return_value = mock_response

        # Create request with only top_p
        request = UserChatRequest(
            prompt="Test prompt",
            model="anthropic.claude-v2",
            max_tokens=100,
            additional_request_params={"top_p": 0.9},
            num_prefill_tokens=10,
        )

        # Mock sample method
        bedrock_user.sample = MagicMock(return_value=request)
        bedrock_user.collect_metrics = MagicMock()

        # Execute
        bedrock_user.chat()

        # Verify
        call_args = mock_client.invoke_model_with_response_stream.call_args
        body = json.loads(call_args[1]["body"])

        assert body["top_p"] == 0.9
        assert "temperature" not in body

    @patch("boto3.Session")
    def test_chat_streaming_reasoning_tokens_from_usage_openai_model(
        self, mock_session, bedrock_user
    ):
        """Streaming with OpenAI model: reasoning_tokens from usage chunk."""
        bedrock_user.on_start()
        mock_client = bedrock_user.runtime_client

        chunk_payload = {
            "text": "OK",
            "usage": {
                "completion_tokens_details": {"reasoning_tokens": 5},
            },
        }
        mock_event = {
            "chunk": {"bytes": json.dumps(chunk_payload).encode()},
        }
        mock_response = {"body": [mock_event]}
        mock_client.invoke_model_with_response_stream.return_value = mock_response

        request = UserChatRequest(
            prompt="Think",
            model="openai.gpt-oss-120b",
            max_tokens=50,
            additional_request_params={"stream": True},
            num_prefill_tokens=10,
        )
        bedrock_user.sample = MagicMock(return_value=request)
        bedrock_user.collect_metrics = MagicMock()

        bedrock_user.chat()

        response = bedrock_user.collect_metrics.call_args[0][0]
        assert response.generated_text == "OK"
        assert response.reasoning_tokens == 5

    @patch("boto3.Session")
    def test_chat_non_streaming_reasoning_tokens_from_usage_openai_model(
        self, mock_session, bedrock_user
    ):
        """Non-streaming with OpenAI model: reasoning_tokens from response body."""
        bedrock_user.on_start()
        mock_client = bedrock_user.runtime_client

        response_body = {
            "text": "Done",
            "usage": {
                "completion_tokens_details": {"reasoning_tokens": 3},
            },
        }
        mock_response = {
            "body": MagicMock(read=lambda: json.dumps(response_body).encode()),
        }
        mock_client.invoke_model.return_value = mock_response

        request = UserChatRequest(
            prompt="Think",
            model="openai.gpt-oss-120b",
            max_tokens=50,
            additional_request_params={"stream": False},
            num_prefill_tokens=10,
        )
        bedrock_user.sample = MagicMock(return_value=request)
        bedrock_user.collect_metrics = MagicMock()

        bedrock_user.chat()

        mock_client.invoke_model.assert_called_once()
        response = bedrock_user.collect_metrics.call_args[0][0]
        assert response.generated_text == "Done"
        assert response.reasoning_tokens == 3
