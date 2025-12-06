"""Tests for Baseten user implementation."""

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
from genai_bench.user.baseten_user import BasetenUser


class TestBasetenUser:
    """Test Baseten user implementation."""

    @pytest.fixture
    def mock_auth(self):
        """Create mock auth provider."""
        auth = MagicMock(spec=ModelAuthProvider)
        auth.get_credentials.return_value = "fake_api_key"
        auth.get_config.return_value = {
            "api_base": "https://model-test.api.baseten.co/environments/production/predict",
            "api_key": "fake_api_key",
        }
        return auth

    @pytest.fixture
    def baseten_user(self, mock_auth):
        """Create Baseten user instance."""
        env = MagicMock()
        env.sampler = MagicMock()
        env.sampler.get_token_length.return_value = 50
        BasetenUser.host = (
            "https://model-test.api.baseten.co/environments/production/predict"
        )
        user = BasetenUser(environment=env)
        user.auth_provider = mock_auth
        user.headers = {
            "Authorization": "Bearer fake_api_key",
            "Content-Type": "application/json",
        }
        return user

    def test_backend_name(self):
        """Test backend name constant."""
        assert BasetenUser.BACKEND_NAME == "baseten"

    def test_supported_tasks(self):
        """Test supported tasks mapping."""
        assert BasetenUser.supported_tasks == {
            "text-to-text": "chat",
            "image-text-to-text": "chat",
            "text-to-embeddings": "embeddings",
        }

    def test_init(self):
        """Test initialization."""
        BasetenUser.host = (
            "https://model-test.api.baseten.co/environments/production/predict"
        )
        user = BasetenUser(environment=MagicMock())
        assert hasattr(user, "client")
        assert user.disable_streaming is False

    def test_on_start_missing_auth(self, baseten_user):
        """Test on_start without auth provider."""
        baseten_user.auth_provider = None
        with pytest.raises(
            ValueError, match="API key and base must be set for OpenAIUser."
        ):
            baseten_user.on_start()

    def test_on_start_missing_host(self, baseten_user):
        """Test on_start without host."""
        baseten_user.host = None
        with pytest.raises(
            ValueError, match="API key and base must be set for OpenAIUser."
        ):
            baseten_user.on_start()

    @patch("genai_bench.user.baseten_user.requests.post")
    def test_chat_prompt_format(self, mock_post, baseten_user):
        """Test chat with prompt format."""
        baseten_user.on_start()
        baseten_user.sample = lambda: UserChatRequest(
            model="test-model",
            prompt="Hello, world!",
            num_prefill_tokens=10,
            max_tokens=100,
            additional_request_params={
                "use_prompt_format": True,
                "temperature": 0.7,
            },
        )

        # Mock response
        response_mock = MagicMock()
        response_mock.status_code = 200
        response_mock.text = "This is a test response"
        mock_post.return_value = response_mock

        # Mock the parse method to return a proper response
        with patch.object(
            baseten_user, "parse_non_streaming_chat_response"
        ) as mock_parse:
            mock_parse.return_value = UserChatResponse(
                status_code=200,
                generated_text="This is a test response",
                tokens_received=50,
                time_at_first_token=1.1,
                num_prefill_tokens=10,
                start_time=1.0,
                end_time=1.2,
            )
            # Mock the collect_metrics method to avoid the assertion error
            with patch.object(baseten_user, "collect_metrics"):
                baseten_user.chat()

        # Verify request was made with correct payload
        mock_post.assert_called_once_with(
            url="https://model-test.api.baseten.co/environments/production/predict",
            json={
                "prompt": "Hello, world!",
                "max_tokens": 100,
                "temperature": 0.7,
                "stream": True,  # Default streaming enabled
            },
            stream=True,
            headers=baseten_user.headers,
        )

    @patch("genai_bench.user.baseten_user.requests.post")
    def test_chat_prompt_format_non_streaming(self, mock_post, baseten_user):
        """Test chat with prompt format and non-streaming."""
        baseten_user.disable_streaming = True
        baseten_user.on_start()
        baseten_user.sample = lambda: UserChatRequest(
            model="test-model",
            prompt="Hello, world!",
            num_prefill_tokens=10,
            max_tokens=100,
            additional_request_params={
                "use_prompt_format": True,
                "temperature": 0.7,
            },
        )

        # Mock response
        response_mock = MagicMock()
        response_mock.status_code = 200
        response_mock.text = "This is a test response"
        mock_post.return_value = response_mock

        baseten_user.chat()

        # Verify request was made with correct payload
        mock_post.assert_called_once_with(
            url="https://model-test.api.baseten.co/environments/production/predict",
            json={
                "prompt": "Hello, world!",
                "max_tokens": 100,
                "temperature": 0.7,
                "stream": False,  # Streaming disabled
            },
            stream=False,
            headers=baseten_user.headers,
        )

    @patch("genai_bench.user.baseten_user.requests.post")
    def test_chat_openai_format(self, mock_post, baseten_user):
        """Test chat with OpenAI format (default)."""
        baseten_user.on_start()
        baseten_user.sample = lambda: UserChatRequest(
            model="test-model",
            prompt="Hello, world!",
            num_prefill_tokens=10,
            max_tokens=100,
            additional_request_params={
                "temperature": 0.7,
            },
        )

        # Mock response
        response_mock = MagicMock()
        response_mock.status_code = 200
        response_mock.text = (
            '{"choices": [{"message": {"content": "This is a test response"}}]}'
        )
        mock_post.return_value = response_mock

        # Mock the parse method to return a proper response
        with patch.object(baseten_user, "parse_chat_response") as mock_parse:
            mock_parse.return_value = UserChatResponse(
                status_code=200,
                generated_text="This is a test response",
                tokens_received=50,
                time_at_first_token=1.1,
                num_prefill_tokens=10,
                start_time=1.0,
                end_time=1.2,
            )
            baseten_user.chat()

        # Verify request was made with correct payload
        mock_post.assert_called_once_with(
            url="https://model-test.api.baseten.co/environments/production/predict",
            json={
                "model": "test-model",
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello, world!",
                    }
                ],
                "max_tokens": 100,
                "temperature": 0.7,
                "ignore_eos": True,
                "stream": True,
                "stream_options": {
                    "include_usage": True,
                },
            },
            stream=True,
            headers=baseten_user.headers,
        )

    @patch("genai_bench.user.baseten_user.requests.post")
    def test_chat_image_request(self, mock_post, baseten_user):
        """Test chat with image request.

        Note: image_content should contain full data URLs (e.g., data:image/jpeg;base64,...)
        as the code passes them through directly without modification.
        """
        baseten_user.on_start()
        # Note: image_content should contain full data URLs as per implementation
        baseten_user.sample = lambda: UserImageChatRequest(
            model="test-model",
            prompt="Describe this image",
            num_prefill_tokens=10,
            max_tokens=100,
            image_content=["data:image/jpeg;base64,base64_image_data"],
            num_images=1,
            additional_request_params={
                "temperature": 0.7,
            },
        )

        # Mock response
        response_mock = MagicMock()
        response_mock.status_code = 200
        response_mock.text = (
            '{"choices": [{"message": {"content": "This is an image description"}}]}'
        )
        mock_post.return_value = response_mock

        # Mock the parse method to return a proper response
        with patch.object(baseten_user, "parse_chat_response") as mock_parse:
            mock_parse.return_value = UserChatResponse(
                status_code=200,
                generated_text="This is an image description",
                tokens_received=50,
                time_at_first_token=1.1,
                num_prefill_tokens=10,
                start_time=1.0,
                end_time=1.2,
            )
            baseten_user.chat()

        # Verify request was made with correct payload
        mock_post.assert_called_once_with(
            url="https://model-test.api.baseten.co/environments/production/predict",
            json={
                "model": "test-model",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "data:image/jpeg;base64,base64_image_data"
                                },
                            },
                        ],
                    }
                ],
                "max_tokens": 100,
                "temperature": 0.7,
                "ignore_eos": True,
                "stream": True,
                "stream_options": {
                    "include_usage": True,
                },
            },
            stream=True,
            headers=baseten_user.headers,
        )

    def test_chat_wrong_request_type(self, baseten_user):
        """Test chat with wrong request type."""
        baseten_user.sample = lambda: UserEmbeddingRequest(
            model="test-model",
            documents=["test document"],
            additional_request_params={},
        )

        with pytest.raises(
            AttributeError, match="user_request should be of type UserChatRequest"
        ):
            baseten_user.chat()

    @patch("genai_bench.user.baseten_user.requests.post")
    def test_send_request_success(self, mock_post, baseten_user):
        """Test successful send_request."""
        baseten_user.on_start()

        # Mock response
        response_mock = MagicMock()
        response_mock.status_code = 200
        response_mock.text = "Test response"
        mock_post.return_value = response_mock

        payload = {"test": "data"}
        result = baseten_user.send_request(
            stream=False,
            endpoint="test",
            payload=payload,
            parse_strategy=lambda *args: UserChatResponse(
                status_code=200,
                generated_text="Test response",
                tokens_received=50,
                time_at_first_token=1.1,
                num_prefill_tokens=10,
                start_time=1.0,
                end_time=1.2,
            ),
        )

        mock_post.assert_called_once_with(
            url="https://model-test.api.baseten.co/environments/production/predict",
            json=payload,
            stream=False,
            headers=baseten_user.headers,
        )
        assert result.status_code == 200

    @patch("genai_bench.user.baseten_user.requests.post")
    def test_send_request_error(self, mock_post, baseten_user):
        """Test send_request with error response."""
        baseten_user.on_start()

        # Mock error response
        response_mock = MagicMock()
        response_mock.status_code = 500
        response_mock.text = "Internal Server Error"
        mock_post.return_value = response_mock

        payload = {"test": "data"}
        result = baseten_user.send_request(
            stream=False,
            endpoint="test",
            payload=payload,
            parse_strategy=lambda *args: UserResponse(status_code=200),
        )

        assert result.status_code == 500
        assert result.error_message == "Internal Server Error"

    @patch("genai_bench.user.baseten_user.requests.post")
    def test_send_request_connection_error(self, mock_post, baseten_user):
        """Test send_request with connection error."""
        baseten_user.on_start()

        # Mock connection error
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection failed")

        payload = {"test": "data"}
        result = baseten_user.send_request(
            stream=False,
            endpoint="test",
            payload=payload,
            parse_strategy=lambda *args: UserResponse(status_code=200),
        )

        assert result.status_code == 503
        assert "Connection error" in result.error_message

    @patch("genai_bench.user.baseten_user.requests.post")
    def test_send_request_timeout(self, mock_post, baseten_user):
        """Test send_request with timeout."""
        baseten_user.on_start()

        # Mock timeout error
        mock_post.side_effect = requests.exceptions.Timeout("Request timed out")

        payload = {"test": "data"}
        result = baseten_user.send_request(
            stream=False,
            endpoint="test",
            payload=payload,
            parse_strategy=lambda *args: UserResponse(status_code=200),
        )

        assert result.status_code == 408
        assert "Request timed out" in result.error_message

    def test_prepare_prompt_request(self, baseten_user):
        """Test _prepare_prompt_request method."""
        user_request = UserChatRequest(
            model="test-model",
            prompt="Hello, world!",
            num_prefill_tokens=10,
            max_tokens=100,
            additional_request_params={
                "use_prompt_format": True,
                "temperature": 0.7,
                "top_p": 0.9,
            },
        )

        payload = baseten_user._prepare_prompt_request(user_request)

        expected_payload = {
            "prompt": "Hello, world!",
            "max_tokens": 100,
            "temperature": 0.7,
            "top_p": 0.9,
            "stream": True,  # Default streaming enabled
        }
        assert payload == expected_payload

    def test_prepare_prompt_request_non_streaming(self, baseten_user):
        """Test _prepare_prompt_request method with non-streaming."""
        baseten_user.disable_streaming = True
        user_request = UserChatRequest(
            model="test-model",
            prompt="Hello, world!",
            num_prefill_tokens=10,
            max_tokens=100,
            additional_request_params={
                "use_prompt_format": True,
                "temperature": 0.7,
            },
        )

        payload = baseten_user._prepare_prompt_request(user_request)

        expected_payload = {
            "prompt": "Hello, world!",
            "max_tokens": 100,
            "temperature": 0.7,
            "stream": False,  # Streaming disabled
        }
        assert payload == expected_payload

    def test_prepare_chat_request(self, baseten_user):
        """Test _prepare_chat_request method."""
        user_request = UserChatRequest(
            model="test-model",
            prompt="Hello, world!",
            num_prefill_tokens=10,
            max_tokens=100,
            additional_request_params={
                "temperature": 0.7,
                "top_p": 0.9,
            },
        )

        payload = baseten_user._prepare_chat_request(user_request)

        expected_payload = {
            "model": "test-model",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, world!",
                }
            ],
            "max_tokens": 100,
            "temperature": 0.7,
            "top_p": 0.9,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {
                "include_usage": True,
            },
        }
        assert payload == expected_payload

    def test_prepare_chat_request_non_streaming(self, baseten_user):
        """Test _prepare_chat_request method with non-streaming."""
        baseten_user.disable_streaming = True
        user_request = UserChatRequest(
            model="test-model",
            prompt="Hello, world!",
            num_prefill_tokens=10,
            max_tokens=100,
            additional_request_params={
                "temperature": 0.7,
            },
        )

        payload = baseten_user._prepare_chat_request(user_request)

        expected_payload = {
            "model": "test-model",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, world!",
                }
            ],
            "max_tokens": 100,
            "temperature": 0.7,
            "ignore_eos": True,
            "stream": False,  # Streaming disabled
        }
        assert payload == expected_payload

    def test_prepare_chat_request_image(self, baseten_user):
        """Test _prepare_chat_request method with image."""
        # Note: image_content should contain full data URLs as per implementation
        user_request = UserImageChatRequest(
            model="test-model",
            prompt="Describe this image",
            num_prefill_tokens=10,
            max_tokens=100,
            image_content=["data:image/jpeg;base64,base64_image_data"],
            num_images=1,
            additional_request_params={
                "temperature": 0.7,
            },
        )

        payload = baseten_user._prepare_chat_request(user_request)

        expected_payload = {
            "model": "test-model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64,base64_image_data"
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 100,
            "temperature": 0.7,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {
                "include_usage": True,
            },
        }
        assert payload == expected_payload

    def test_parse_plain_text_response(self, baseten_user):
        """Test _parse_plain_text_response method."""
        # Mock response
        response_mock = MagicMock()
        response_mock.text = "This is a plain text response"

        result = baseten_user._parse_plain_text_response(
            response_mock,
            start_time=1.0,
            num_prefill_tokens=10,
            _=2.0,
        )

        assert isinstance(result, UserChatResponse)
        assert result.status_code == 200
        assert result.generated_text == "This is a plain text response"
        assert result.tokens_received == 50  # From mock sampler
        assert result.num_prefill_tokens == 10
        assert result.start_time == 1.0
        assert result.end_time > 1.0

    def test_parse_plain_text_response_json(self, baseten_user):
        """Test _parse_plain_text_response method with JSON response."""
        # Mock response with JSON
        response_mock = MagicMock()
        response_mock.text = '{"text": "This is a JSON response"}'

        result = baseten_user._parse_plain_text_response(
            response_mock,
            start_time=1.0,
            num_prefill_tokens=10,
            _=2.0,
        )

        assert isinstance(result, UserChatResponse)
        assert result.status_code == 200
        assert result.generated_text == "This is a JSON response"
        assert result.tokens_received == 50  # From mock sampler

    def test_parse_plain_text_response_json_alternative_fields(self, baseten_user):
        """Test _parse_plain_text_response method with different JSON fields."""
        # Test with "output" field
        response_mock = MagicMock()
        response_mock.text = '{"output": "This is an output response"}'

        result = baseten_user._parse_plain_text_response(
            response_mock,
            start_time=1.0,
            num_prefill_tokens=10,
            _=2.0,
        )

        assert result.generated_text == "This is an output response"

        # Test with "response" field
        response_mock.text = '{"response": "This is a response field"}'
        result = baseten_user._parse_plain_text_response(
            response_mock,
            start_time=1.0,
            num_prefill_tokens=10,
            _=2.0,
        )

        assert result.generated_text == "This is a response field"

        # Test with "generated_text" field
        response_mock.text = '{"generated_text": "This is generated text"}'
        result = baseten_user._parse_plain_text_response(
            response_mock,
            start_time=1.0,
            num_prefill_tokens=10,
            _=2.0,
        )

        assert result.generated_text == "This is generated text"

    def test_parse_plain_text_response_error(self, baseten_user):
        """Test _parse_plain_text_response method with error."""
        # Mock response that raises exception during json.loads
        response_mock = MagicMock()
        response_mock.text = "Invalid JSON"

        # Mock json.loads to raise an exception
        with patch("json.loads", side_effect=Exception("Test error")):
            result = baseten_user._parse_plain_text_response(
                response_mock,
                start_time=1.0,
                num_prefill_tokens=10,
                _=2.0,
            )

        assert isinstance(result, UserResponse)
        assert result.status_code == 500
        assert "Failed to parse plain text response" in result.error_message

    def test_parse_plain_text_streaming_response(self, baseten_user):
        """Test _parse_plain_text_streaming_response method."""
        # Mock response
        response_mock = MagicMock()
        response_mock.iter_lines.return_value = [
            b"First chunk",
            b"Second chunk",
            b"Third chunk",
        ]

        result = baseten_user._parse_plain_text_streaming_response(
            response_mock,
            start_time=1.0,
            num_prefill_tokens=10,
            _=2.0,
        )

        assert isinstance(result, UserChatResponse)
        assert result.status_code == 200
        assert result.generated_text == "First chunkSecond chunkThird chunk"
        assert result.tokens_received == 50  # From mock sampler
        assert result.num_prefill_tokens == 10
        assert result.start_time == 1.0
        assert result.end_time > 1.0

    def test_parse_plain_text_streaming_response_error(self, baseten_user):
        """Test _parse_plain_text_streaming_response method with error."""
        # Mock response that raises exception
        response_mock = MagicMock()
        response_mock.iter_lines.side_effect = Exception("Test error")

        result = baseten_user._parse_plain_text_streaming_response(
            response_mock,
            start_time=1.0,
            num_prefill_tokens=10,
            _=2.0,
        )

        assert isinstance(result, UserResponse)
        assert result.status_code == 500
        assert "Failed to parse plain text streaming response" in result.error_message

    def test_stream_parameter_filtering(self, baseten_user):
        """Test that stream parameter is filtered out from additional_request_params."""
        user_request = UserChatRequest(
            model="test-model",
            prompt="Hello, world!",
            num_prefill_tokens=10,
            max_tokens=100,
            additional_request_params={
                "use_prompt_format": True,
                "stream": False,  # This should be filtered out
                "temperature": 0.7,
                "top_p": 0.9,
            },
        )

        payload = baseten_user._prepare_prompt_request(user_request)

        # Check that other parameters are preserved
        assert payload["temperature"] == 0.7
        assert payload["top_p"] == 0.9
        # Check that stream is set by our logic (not from additional_request_params)
        assert payload["stream"]  # Default streaming enabled

    def test_use_prompt_format_parameter_filtering(self, baseten_user):
        """Test that use_prompt_format parameter is filtered out from additional_request_params."""
        user_request = UserChatRequest(
            model="test-model",
            prompt="Hello, world!",
            num_prefill_tokens=10,
            max_tokens=100,
            additional_request_params={
                "use_prompt_format": True,  # This should be filtered out
                "temperature": 0.7,
                "top_p": 0.9,
            },
        )

        payload = baseten_user._prepare_prompt_request(user_request)

        # Check that use_prompt_format parameter is not in the payload
        assert "use_prompt_format" not in payload
        # Check that other parameters are preserved
        assert payload["temperature"] == 0.7
        assert payload["top_p"] == 0.9
