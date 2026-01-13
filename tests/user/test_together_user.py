"""Tests for TogetherUser class."""

from unittest.mock import MagicMock, patch

import pytest

from genai_bench.protocol import (
    UserChatRequest,
    UserEmbeddingRequest,
    UserImageChatRequest,
)
from genai_bench.user.together_user import TogetherUser


@pytest.fixture
def mock_together_user():
    """Create a mock TogetherUser for testing."""
    mock_auth = MagicMock()
    mock_auth.get_credentials.return_value = "test_api_key"

    TogetherUser.auth_provider = mock_auth
    TogetherUser.host = "https://api.together.xyz"

    user = TogetherUser(environment=MagicMock())
    user.sampler = MagicMock()
    return user


def test_backend_name():
    """Test backend name is set correctly."""
    assert TogetherUser.BACKEND_NAME == "together"


def test_supported_tasks():
    """Test supported tasks are defined."""
    assert "text-to-text" in TogetherUser.supported_tasks
    assert "image-text-to-text" in TogetherUser.supported_tasks
    assert "text-to-embeddings" in TogetherUser.supported_tasks


def test_on_start_missing_host():
    """Test on_start raises error when host is missing."""
    # Set valid host first to create the user, then test None host in on_start
    TogetherUser.host = "https://api.together.xyz"
    TogetherUser.auth_provider = MagicMock()
    user = TogetherUser(environment=MagicMock())
    user.host = None  # Now set to None to trigger the error
    with pytest.raises(ValueError, match="API key and base must be set"):
        user.on_start()


def test_on_start_missing_auth():
    """Test on_start raises error when auth_provider is missing."""
    TogetherUser.host = "https://api.together.xyz"
    TogetherUser.auth_provider = None
    user = TogetherUser(environment=MagicMock())
    with pytest.raises(ValueError, match="API key and base must be set"):
        user.on_start()


def test_on_start_success(mock_together_user):
    """Test successful on_start initialization."""
    mock_together_user.on_start()
    assert mock_together_user.headers is not None
    assert "Authorization" in mock_together_user.headers
    assert mock_together_user.headers["Authorization"] == "Bearer test_api_key"
    assert mock_together_user.headers["Content-Type"] == "application/json"


@patch("genai_bench.user.together_user.requests.post")
def test_chat_text_request(mock_post, mock_together_user):
    """Test chat with text request."""
    mock_together_user.on_start()
    mock_together_user.sample = lambda: UserChatRequest(
        model="test-model",
        prompt="Hello",
        num_prefill_tokens=5,
        max_tokens=10,
        additional_request_params={},
    )

    response_mock = MagicMock()
    response_mock.status_code = 200
    response_mock.iter_lines.return_value = [
        b'data: {"choices":[{"delta":{"content":"Hi"}}],"usage":null}',
        b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
        b'"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}',
        b"data: [DONE]",
    ]
    mock_post.return_value = response_mock

    mock_together_user.chat()

    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args.kwargs
    assert call_kwargs["url"] == "https://api.together.xyz/v1/chat/completions"
    assert call_kwargs["json"]["model"] == "test-model"
    assert call_kwargs["json"]["messages"][0]["content"] == "Hello"
    assert call_kwargs["json"]["max_tokens"] == 10
    assert call_kwargs["stream"] is True


@patch("genai_bench.user.together_user.requests.post")
def test_chat_image_request(mock_post, mock_together_user):
    """Test chat with image request."""
    mock_together_user.on_start()
    mock_together_user.sample = lambda: UserImageChatRequest(
        model="test-model",
        prompt="Describe this image",
        image_content=["http://example.com/image.jpg"],
        num_prefill_tokens=10,
        num_images=1,
        max_tokens=20,
        additional_request_params={},
    )

    response_mock = MagicMock()
    response_mock.status_code = 200
    response_mock.iter_lines.return_value = [
        b'data: {"choices":[{"delta":{"content":"A cat"}}],"usage":null}',
        b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
        b'"usage":{"prompt_tokens":10,"completion_tokens":3,"total_tokens":13}}',
        b"data: [DONE]",
    ]
    mock_post.return_value = response_mock

    mock_together_user.chat()

    call_kwargs = mock_post.call_args.kwargs
    messages_content = call_kwargs["json"]["messages"][0]["content"]
    assert isinstance(messages_content, list)
    assert messages_content[0]["type"] == "text"
    assert messages_content[1]["type"] == "image_url"


def test_chat_wrong_request_type(mock_together_user):
    """Test chat with wrong request type raises error."""
    mock_together_user.on_start()
    mock_together_user.sample = lambda: UserEmbeddingRequest(
        model="test-model",
        documents=["Hello"],
        num_prefill_tokens=5,
        additional_request_params={},
    )

    with pytest.raises(AttributeError, match="UserChatRequest"):
        mock_together_user.chat()


@patch("genai_bench.user.together_user.requests.post")
def test_embeddings(mock_post, mock_together_user):
    """Test embeddings request."""
    mock_together_user.on_start()
    mock_together_user.sample = lambda: UserEmbeddingRequest(
        model="test-embedding-model",
        documents=["Embed this text", "And this"],
        num_prefill_tokens=5,
        additional_request_params={},
    )

    response_mock = MagicMock()
    response_mock.status_code = 200
    response_mock.json.return_value = {
        "data": [{"embedding": [0.1, 0.2, 0.3]}, {"embedding": [0.4, 0.5, 0.6]}],
        "usage": {"prompt_tokens": 5, "total_tokens": 5},
    }
    mock_post.return_value = response_mock

    mock_together_user.embeddings()

    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args.kwargs
    assert call_kwargs["url"] == "https://api.together.xyz/v1/embeddings"
    assert call_kwargs["json"]["model"] == "test-embedding-model"
    # input should be the documents list (possibly shuffled)
    assert isinstance(call_kwargs["json"]["input"], list)
    assert len(call_kwargs["json"]["input"]) == 2


def test_embeddings_wrong_request_type(mock_together_user):
    """Test embeddings with wrong request type raises error."""
    mock_together_user.on_start()
    mock_together_user.sample = lambda: UserChatRequest(
        model="test-model",
        prompt="Hello",
        num_prefill_tokens=5,
        max_tokens=10,
        additional_request_params={},
    )

    with pytest.raises(AttributeError, match="UserEmbeddingRequest"):
        mock_together_user.embeddings()
