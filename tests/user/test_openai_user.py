import logging
from unittest.mock import ANY, MagicMock, patch

import pytest
import requests

from genai_bench.protocol import (
    UserChatRequest,
    UserChatResponse,
    UserEmbeddingRequest,
    UserImageChatRequest,
    UserResponse,
)
from genai_bench.user.openai_user import OpenAIUser


@pytest.fixture
def mock_openai_user():
    # Set up mock auth provider
    mock_auth = MagicMock()
    mock_auth.get_credentials.return_value = "fake_api_key"
    mock_auth.get_headers.return_value = {"Authorization": "Bearer fake_api_key"}
    mock_auth.get_config.return_value = {
        "api_base": "http://example.com",
        "api_key": "fake_api_key",
    }
    OpenAIUser.auth_provider = mock_auth
    OpenAIUser.host = "http://example.com"

    user = OpenAIUser(environment=MagicMock())
    user.user_requests = [
        UserChatRequest(
            model="gpt-3",
            prompt="Hello",
            num_prefill_tokens=5,
            additional_request_params={"ignore_eos": False},
            max_tokens=10,
        )
    ] * 5
    return user


def test_on_start_missing_api_key_base():
    env = MagicMock()
    OpenAIUser.host = "https://api.openai.com"
    user = OpenAIUser(env)
    user.host = None
    user.auth_signer = None
    with pytest.raises(
        ValueError, match="API key and base must be set for OpenAIUser."
    ):
        user.on_start()


@patch("genai_bench.user.openai_user.requests.post")
def test_chat(mock_post, mock_openai_user):
    mock_openai_user.on_start()
    mock_openai_user.sample = lambda: UserChatRequest(
        model="gpt-3",
        prompt="Hello",
        num_prefill_tokens=5,
        additional_request_params={"ignore_eos": False},
        max_tokens=10,
    )

    response_mock = MagicMock()
    response_mock.status_code = 200
    # Mock the iter_content method to simulate streaming
    response_mock.iter_lines = MagicMock(
        return_value=[
            b'data: {"id":"chat-c084b07b809048d88f3fad11cface2b7","object":"chat.completion.chunk","created":1724364845,"model":"meta-llama/Meta-Llama-3-70B-Instruct","choices":[{"index":0,"delta":{"role":"assistant"},"logprobs":null,"finish_reason":null}]}',  # noqa:E501
            b'data: {"id":"chat-c084b07b809048d88f3fad11cface2b7","object":"chat.completion.chunk","created":1724364845,"model":"meta-llama/Meta-Llama-3-70B-Instruct","choices":[{"index":0,"delta":{"content":"R"},"logprobs":null,"finish_reason":null}]}',  # noqa:E501
            b'data: {"id":"chat-c084b07b809048d88f3fad11cface2b7","object":"chat.completion.chunk","created":1724364845,"model":"meta-llama/Meta-Llama-3-70B-Instruct","choices":[{"index":0,"delta":{"content":"AG"},"logprobs":null,"finish_reason":null}]}',  # noqa:E501
            b'data: {"id":"chat-c084b07b809048d88f3fad11cface2b7","object":"chat.completion.chunk","created":1724364845,"model":"meta-llama/Meta-Llama-3-70B-Instruct","choices":[{"index":0,"delta":{"content":" ("},"logprobs":null,"finish_reason":null}]}',  # noqa:E501
            b'data: {"id":"chat-c084b07b809048d88f3fad11cface2b7","object":"chat.completion.chunk","created":1724364845,"model":"meta-llama/Meta-Llama-3-70B-Instruct","choices":[{"index":0,"delta":{"content":"Ret"},"logprobs":null,"finish_reason":null}]}',  # noqa:E501
            b'data: {"id":"chat-c084b07b809048d88f3fad11cface2b7","object":"chat.completion.chunk","created":1724364845,"model":"meta-llama/Meta-Llama-3-70B-Instruct","choices":[{"index":0,"delta":{"content":"rie"},"logprobs":null,"finish_reason":null}]}',  # noqa:E501
            b'data: {"id":"chat-c084b07b809048d88f3fad11cface2b7","object":"chat.completion.chunk","created":1724364845,"model":"meta-llama/Meta-Llama-3-70B-Instruct","choices":[{"index":0,"delta":{"content":"val"},"logprobs":null,"finish_reason":null}]}',  # noqa:E501
            b'data: {"id":"chat-c084b07b809048d88f3fad11cface2b7","object":"chat.completion.chunk","created":1724364845,"model":"meta-llama/Meta-Llama-3-70B-Instruct","choices":[{"index":0,"delta":{"content":"-Aug"},"logprobs":null,"finish_reason":null}]}',  # noqa:E501
            b'data: {"id":"chat-c084b07b809048d88f3fad11cface2b7","object":"chat.completion.chunk","created":1724364845,"model":"meta-llama/Meta-Llama-3-70B-Instruct","choices":[{"index":0,"delta":{"content":"mented"},"logprobs":null,"finish_reason":null}]}',  # noqa:E501
            b'data: {"id":"chat-c084b07b809048d88f3fad11cface2b7","object":"chat.completion.chunk","created":1724364845,"model":"meta-llama/Meta-Llama-3-70B-Instruct","choices":[{"index":0,"delta":{"content":" Generation"},"logprobs":null,"finish_reason":null}]}',  # noqa:E501
            b'data: {"id":"chat-c084b07b809048d88f3fad11cface2b7","object":"chat.completion.chunk","created":1724364845,"model":"meta-llama/Meta-Llama-3-70B-Instruct","choices":[{"index":0,"delta":{"content":")"},"logprobs":null,"finish_reason":"length","stop_reason":null}]}',  # noqa:E501
            b"",
            b'data: {"id":"chat-f774576725a9470ea37c7706a45a6557","object":"chat.completion.chunk","created":1724448805,"model":"meta-llama/Meta-Llama-3-70B-Instruct","choices":[],"usage":{"prompt_tokens":5,"total_tokens":15,"completion_tokens":10}}',  # noqa:E501
            b"data: [DONE]",
        ]
    )
    mock_post.return_value = response_mock

    mock_openai_user.chat()

    mock_post.assert_called_once_with(
        url="http://example.com/v1/chat/completions",
        json={
            "model": "gpt-3",
            "messages": [{"role": "user", "content": ANY}],
            "max_tokens": 10,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {
                "include_usage": True,
            },
        },
        stream=True,
        headers={
            "Authorization": "Bearer fake_api_key",
            "Content-Type": "application/json",
        },
    )


@patch("genai_bench.user.openai_user.requests.post")
def test_vision(mock_post, mock_openai_user):
    mock_openai_user.on_start()
    mock_openai_user.sample = lambda: UserImageChatRequest(
        model="Phi-3-vision-128k-instruct",
        prompt="what's in the image?",
        num_prefill_tokens=5,
        image_content=[
            "data:image/jpeg;base64,UklGRhowDgBXRUJQVlA4WAoAAAAgAAAA/wkAhAYASUNDUAwCAAAAAAIMbGNtcwIQAABtbnRyUkdCIFhZWiAH3AABABkAAwApADlhY3NwQVBQTAAAAAAA"
        ],  # noqa:E501
        num_images=1,
        max_tokens=None,
    )

    response_mock = MagicMock()
    response_mock.status_code = 200
    # Mock the iter_content method to simulate streaming
    response_mock.iter_lines = MagicMock(
        return_value=[
            b'data: {"id":"chat-81667d8b92f74ae8ad207009cc1d2a5b","object":"chat.completion.chunk","created":1727154331,"model":"meta-llama/Meta-Llama-3-70B-Instruct","choices":[{"index":0,"delta":{"role":"assistant","content":""},"logprobs":null,"finish_reason":null}]}',  # noqa:E501
            b'data: {"id":"chat-81667d8b92f74ae8ad207009cc1d2a5b","object":"chat.completion.chunk","created":1727154331,"model":"meta-llama/Meta-Llama-3-70B-Instruct","choices":[{"index":0,"delta":{"content":"The"},"logprobs":null,"finish_reason":null}]}',  # noqa:E501
            b'data: {"id":"chat-81667d8b92f74ae8ad207009cc1d2a5b","object":"chat.completion.chunk","created":1727154331,"model":"meta-llama/Meta-Llama-3-70B-Instruct","choices":[{"index":0,"delta":{"content":" image"},"logprobs":null,"finish_reason":null}]}',  # noqa:E501
            b'data: {"id":"chat-81667d8b92f74ae8ad207009cc1d2a5b","object":"chat.completion.chunk","created":1727154331,"model":"meta-llama/Meta-Llama-3-70B-Instruct","choices":[{"index":0,"delta":{"content":" depicts"},"logprobs":null,"finish_reason":null}]}',  # noqa:E501
            b'data: {"id":"chat-81667d8b92f74ae8ad207009cc1d2a5b","object":"chat.completion.chunk","created":1727154331,"model":"meta-llama/Meta-Llama-3-70B-Instruct","choices":[{"index":0,"delta":{"content":" a"},"logprobs":null,"finish_reason":null}]}',  # noqa:E501
            b'data: {"id":"chat-81667d8b92f74ae8ad207009cc1d2a5b","object":"chat.completion.chunk","created":1727154331,"model":"meta-llama/Meta-Llama-3-70B-Instruct","choices":[{"index":0,"delta":{"content":" serene"},"logprobs":null,"finish_reason":"length","stop_reason":null}]}',  # noqa:E501
            b'data: {"id":"chat-81667d8b92f74ae8ad207009cc1d2a5b","object":"chat.completion.chunk","created":1727154331,"model":"meta-llama/Meta-Llama-3-70B-Instruct","choices":[],"usage":{"prompt_tokens":6421,"total_tokens":6426,"completion_tokens":5}}',  # noqa:E501
            b"data: [DONE]",
        ]
    )
    mock_post.return_value = response_mock

    text_content = [{"type": "text", "text": "what's in the image?"}]
    image = "UklGRhowDgBXRUJQVlA4WAoAAAAgAAAA/wkAhAYASUNDUAwCAAAAAAIMbGNtcwIQAABtbnRyUkdCIFhZWiAH3AABABkAAwApADlhY3NwQVBQTAAAAAAA"  # noqa:E501
    image_content = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image}"},
        }
    ]

    mock_openai_user.chat()

    mock_post.assert_called_once_with(
        url="http://example.com/v1/chat/completions",
        json={
            "model": "Phi-3-vision-128k-instruct",
            "messages": [{"role": "user", "content": text_content + image_content}],
            "max_tokens": None,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {
                "include_usage": True,
            },
        },
        stream=True,
        headers={
            "Authorization": "Bearer fake_api_key",
            "Content-Type": "application/json",
        },
    )


@patch("genai_bench.user.openai_user.requests.post")
def test_embeddings(mock_post, mock_openai_user):
    mock_openai_user.on_start()
    mock_openai_user.sample = lambda: UserEmbeddingRequest(
        model="gpt-3", documents=["Document 1", "Document 2"]
    )

    response_mock = MagicMock()
    response_mock.status_code = 200
    response_mock.usage = {"prompt_tokens": 8, "total_tokens": 8}
    mock_post.return_value = response_mock

    mock_openai_user.embeddings()

    mock_post.assert_called_once_with(
        url="http://example.com/v1/embeddings",
        json={
            "model": "gpt-3",
            "input": ANY,
            "encoding_format": "float",
        },
        stream=False,
        headers={
            "Authorization": "Bearer fake_api_key",
            "Content-Type": "application/json",
        },
    )


def test_chat_with_wrong_request_type(mock_openai_user):
    mock_openai_user.on_start()
    mock_openai_user.sample = lambda: "InvalidRequestType"

    with pytest.raises(
        AttributeError,
        match="user_request should be of type UserChatRequest for OpenAIUser.chat",
    ):
        mock_openai_user.chat()


def test_embeddings_with_wrong_request_type(mock_openai_user):
    mock_openai_user.on_start()
    mock_openai_user.sample = lambda: "InvalidRequestType"

    with pytest.raises(
        AttributeError,
        match="user_request should be of type UserEmbeddingRequest for "
        "OpenAIUser.embeddings",
    ):
        mock_openai_user.embeddings()


@patch("genai_bench.user.openai_user.requests.post")
def test_send_request_non_200_response(mock_post, mock_openai_user):
    mock_openai_user.on_start()

    # Simulate a non-200 response
    response_mock = MagicMock()
    response_mock.status_code = 500
    response_mock.text = "Internal Server Error"
    mock_post.return_value = response_mock

    user_response = mock_openai_user.send_request(
        stream=True,
        endpoint="/v1/test",
        payload={"key": "value"},
        num_prefill_tokens=5,
        parse_strategy=mock_openai_user.parse_chat_response,
    )

    # Assert that the UserResponse contains the error details
    assert isinstance(user_response, UserResponse)
    assert user_response.status_code == 500
    assert user_response.error_message == "Internal Server Error"
    mock_post.assert_called_once()  # fix for python 3.12


@patch("genai_bench.user.openai_user.requests.post")
def test_send_request_embeddings_response(mock_post, mock_openai_user):
    mock_openai_user.on_start()

    # Simulate a 200 embeddings response
    response_mock = MagicMock()
    response_mock.status_code = 200
    response_mock.json.return_value = {"usage": {"prompt_tokens": 5, "total_tokens": 5}}
    mock_post.return_value = response_mock

    user_response = mock_openai_user.send_request(
        stream=False,
        endpoint="/v1/embeddings",
        payload={"key": "value"},
        num_prefill_tokens=5,
        parse_strategy=mock_openai_user.parse_embedding_response,
    )

    # Assert type is UserResponse for embeddings request
    assert isinstance(user_response, UserResponse)
    assert user_response.status_code == 200
    assert user_response.num_prefill_tokens == 5
    mock_post.assert_called_once()


@patch("genai_bench.user.openai_user.requests.post")
def test_send_request_chat_response(mock_post, mock_openai_user):
    mock_openai_user.on_start()

    # Simulate a 200 chat response
    response_mock = MagicMock()
    response_mock.status_code = 200
    response_mock.iter_lines = MagicMock(
        return_value=[
            b'data: {"id":"chat-81667d8b92f74ae8ad207009cc1d2a5b","object":"chat.completion.chunk","created":1727154331,"model":"meta-llama/Meta-Llama-3-70B-Instruct","choices":[{"index":0,"delta":{"role":"assistant","content":""},"logprobs":null,"finish_reason":null}]}',  # noqa:E501
            b'data: {"id":"chat-81667d8b92f74ae8ad207009cc1d2a5b","object":"chat.completion.chunk","created":1727154331,"model":"meta-llama/Meta-Llama-3-70B-Instruct","choices":[{"index":0,"delta":{"content":"The"},"logprobs":null,"finish_reason":null}]}',  # noqa:E501
            b'data: {"id":"chat-81667d8b92f74ae8ad207009cc1d2a5b","object":"chat.completion.chunk","created":1727154331,"model":"meta-llama/Meta-Llama-3-70B-Instruct","choices":[{"index":0,"delta":{"content":" image"},"logprobs":null,"finish_reason":null}]}',  # noqa:E501
            b'data: {"id":"chat-81667d8b92f74ae8ad207009cc1d2a5b","object":"chat.completion.chunk","created":1727154331,"model":"meta-llama/Meta-Llama-3-70B-Instruct","choices":[{"index":0,"delta":{"content":" depicts"},"logprobs":null,"finish_reason":null}]}',  # noqa:E501
            b'data: {"id":"chat-81667d8b92f74ae8ad207009cc1d2a5b","object":"chat.completion.chunk","created":1727154331,"model":"meta-llama/Meta-Llama-3-70B-Instruct","choices":[{"index":0,"delta":{"content":" a"},"logprobs":null,"finish_reason":null}]}',  # noqa:E501
            b'data: {"id":"chat-81667d8b92f74ae8ad207009cc1d2a5b","object":"chat.completion.chunk","created":1727154331,"model":"meta-llama/Meta-Llama-3-70B-Instruct","choices":[{"index":0,"delta":{"content":" serene"},"logprobs":null,"finish_reason":"length","stop_reason":null}]}',  # noqa:E501
            b'data: {"id":"chat-81667d8b92f74ae8ad207009cc1d2a5b","object":"chat.completion.chunk","created":1727154331,"model":"meta-llama/Meta-Llama-3-70B-Instruct","choices":[],"usage":{"prompt_tokens":6421,"total_tokens":6426,"completion_tokens":5}}',  # noqa:E501
            b"data: [DONE]",
        ]
    )
    mock_post.return_value = response_mock

    user_response = mock_openai_user.send_request(
        stream=True,
        endpoint="/v1/test",
        payload={"key": "value"},
        num_prefill_tokens=5,
        parse_strategy=mock_openai_user.parse_chat_response,
    )

    # Assert type is UserChatResponse for chat request
    assert isinstance(user_response, UserChatResponse)
    assert user_response.status_code == 200
    assert user_response.tokens_received == 5
    assert user_response.num_prefill_tokens == 5
    assert user_response.generated_text == "The image depicts a serene"
    mock_post.assert_called_once()


@patch("genai_bench.user.openai_user.requests.post")
def test_chat_no_usage_info(mock_post, mock_openai_user, caplog):
    mock_openai_user.environment.sampler = MagicMock()
    mock_openai_user.environment.sampler.get_token_length = (
        lambda text, add_special_tokens=True: len(text)
    )
    mock_openai_user.on_start()
    mock_openai_user.sample = lambda: UserChatRequest(
        model="gpt-3",
        prompt="Hello",
        num_prefill_tokens=5,
        additional_request_params={"ignore_eos": False},
        max_tokens=10,
    )

    response_mock = MagicMock()
    response_mock.status_code = 200
    # Mock the iter_content method to simulate streaming
    response_mock.iter_lines = MagicMock(
        return_value=[
            b'data: {"id":"chat-c084b07b809048d88f3fad11cface2b7","object":"chat.completion.chunk","created":1724364845,"model":"meta-llama/Meta-Llama-3-70B-Instruct","choices":[{"index":0,"delta":{"role":"assistant"},"logprobs":null,"finish_reason":null}]}',  # noqa:E501
            b'data: {"id":"chat-c084b07b809048d88f3fad11cface2b7","object":"chat.completion.chunk","created":1724364845,"model":"meta-llama/Meta-Llama-3-70B-Instruct","choices":[{"index":0,"delta":{"content":"R"},"logprobs":null,"finish_reason":null}]}',  # noqa:E501
            b'data: {"id":"chat-c084b07b809048d88f3fad11cface2b7","object":"chat.completion.chunk","created":1724364845,"model":"meta-llama/Meta-Llama-3-70B-Instruct","choices":[{"index":0,"delta":{"content":"AG"},"logprobs":null,"finish_reason":null}]}',  # noqa:E501
            b'data: {"id":"chat-c084b07b809048d88f3fad11cface2b7","object":"chat.completion.chunk","created":1724364845,"model":"meta-llama/Meta-Llama-3-70B-Instruct","choices":[{"index":0,"delta":{"content":" ("},"logprobs":null,"finish_reason":null}]}',  # noqa:E501
            b'data: {"id":"chat-c084b07b809048d88f3fad11cface2b7","object":"chat.completion.chunk","created":1724364845,"model":"meta-llama/Meta-Llama-3-70B-Instruct","choices":[{"index":0,"delta":{"content":"Ret"},"logprobs":null,"finish_reason":null}]}',  # noqa:E501
            b'data: {"id":"chat-c084b07b809048d88f3fad11cface2b7","object":"chat.completion.chunk","created":1724364845,"model":"meta-llama/Meta-Llama-3-70B-Instruct","choices":[{"index":0,"delta":{"content":"rie"},"logprobs":null,"finish_reason":null}]}',  # noqa:E501
            b'data: {"id":"chat-c084b07b809048d88f3fad11cface2b7","object":"chat.completion.chunk","created":1724364845,"model":"meta-llama/Meta-Llama-3-70B-Instruct","choices":[{"index":0,"delta":{"content":"val"},"logprobs":null,"finish_reason":null}]}',  # noqa:E501
            b'data: {"id":"chat-c084b07b809048d88f3fad11cface2b7","object":"chat.completion.chunk","created":1724364845,"model":"meta-llama/Meta-Llama-3-70B-Instruct","choices":[{"index":0,"delta":{"content":"-Aug"},"logprobs":null,"finish_reason":null}]}',  # noqa:E501
            b'data: {"id":"chat-c084b07b809048d88f3fad11cface2b7","object":"chat.completion.chunk","created":1724364845,"model":"meta-llama/Meta-Llama-3-70B-Instruct","choices":[{"index":0,"delta":{"content":"mented"},"logprobs":null,"finish_reason":null}]}',  # noqa:E501
            b'data: {"id":"chat-c084b07b809048d88f3fad11cface2b7","object":"chat.completion.chunk","created":1724364845,"model":"meta-llama/Meta-Llama-3-70B-Instruct","choices":[{"index":0,"delta":{"content":" Generation"},"logprobs":null,"finish_reason":null}]}',  # noqa:E501
            b'data: {"id":"chat-c084b07b809048d88f3fad11cface2b7","object":"chat.completion.chunk","created":1724364845,"model":"meta-llama/Meta-Llama-3-70B-Instruct","choices":[{"index":0,"delta":{"content":")"},"logprobs":null,"finish_reason":"length","stop_reason":null}]}',  # noqa:E501
            b"",
            b'data: {"id":"chat-f774576725a9470ea37c7706a45a6557","object":"chat.completion.chunk","created":1724448805,"model":"meta-llama/Meta-Llama-3-70B-Instruct","choices":[]}',  # noqa:E501
            b"data: [DONE]",
        ]
    )
    mock_post.return_value = response_mock

    with caplog.at_level(logging.WARNING):
        user_response = mock_openai_user.send_request(
            stream=True,
            endpoint="/v1/test",
            payload={"key": "value"},
            num_prefill_tokens=5,
            parse_strategy=mock_openai_user.parse_chat_response,
        )

    assert (
        "There is no usage info returned from the model server. Estimated "
        "tokens_received based on the model tokenizer." in caplog.text
    )
    assert user_response.tokens_received == len(user_response.generated_text)


@patch("genai_bench.user.openai_user.requests.post")
def test_chat_request_exception(mock_post, mock_openai_user):
    """Test handling of request exceptions during chat."""
    mock_openai_user.on_start()
    mock_openai_user.sample = lambda: UserChatRequest(
        model="gpt-3",
        prompt="Hello",
        num_prefill_tokens=5,
        additional_request_params={},
        max_tokens=10,
    )

    # Simulate a request exception
    mock_post.side_effect = requests.exceptions.RequestException("Network error")

    user_response = mock_openai_user.send_request(
        stream=True,
        endpoint="/v1/test",
        payload={"key": "value"},
        num_prefill_tokens=5,
        parse_strategy=mock_openai_user.parse_chat_response,
    )

    assert user_response.status_code == 500

    # Simulate a request exception
    mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")

    user_response = mock_openai_user.send_request(
        stream=True,
        endpoint="/v1/test",
        payload={"key": "value"},
        num_prefill_tokens=5,
        parse_strategy=mock_openai_user.parse_chat_response,
    )

    assert user_response.status_code == 503
    assert user_response.error_message == "Connection error: Connection refused"


@patch("genai_bench.user.openai_user.requests.post")
def test_chat_with_warning_first_chunk_tokens(mock_post, mock_openai_user, caplog):
    """Test warning when first chunk has multiple tokens."""
    mock_openai_user.on_start()
    mock_openai_user.sample = lambda: UserChatRequest(
        model="gpt-3",
        prompt="Hello",
        num_prefill_tokens=5,
        additional_request_params={},
        max_tokens=10,
    )

    response_mock = MagicMock()
    response_mock.status_code = 200
    response_mock.iter_lines = MagicMock(
        return_value=[
            b'data: {"id":"chat-f774576725a9470ea37c7706a45a6557","object":"chat.completion.chunk","created":1724448805,"model":"meta-llama/Meta-Llama-3-70B-Instruct","choices":[{"delta":{"content":"First chunk with multiple tokens","usage":{"completion_tokens":5}},"finish_reason":null}]}',  # noqa:E501
            b'data: {"id":"chat-f774576725a9470ea37c7706a45a6557","object":"chat.completion.chunk","created":1724448805,"model":"meta-llama/Meta-Llama-3-70B-Instruct","choices":[]}',  # noqa:E501
            b"data: [DONE]",
        ]
    )
    mock_post.return_value = response_mock

    with caplog.at_level(logging.WARNING):
        mock_openai_user.chat()

    assert "The first chunk the server returned has >1 tokens: 5" in caplog.text


@patch("genai_bench.user.openai_user.requests.post")
def test_chat_empty_choices_warning(mock_post, mock_openai_user, caplog):
    """Test warning when choices array is empty."""
    mock_openai_user.on_start()
    mock_openai_user.sample = lambda: UserChatRequest(
        model="gpt-3",
        prompt="Hello",
        num_prefill_tokens=5,
        additional_request_params={},
        max_tokens=10,
    )

    response_mock = MagicMock()
    response_mock.status_code = 200
    response_mock.iter_lines = MagicMock(
        return_value=[
            b'data: {"choices":[]}',
            b'data: {"choices":[{"delta":{"content":"First chunk with multiple tokens"},"finish_reason":null,"usage":{"completion_tokens":5}}]}',  # noqa:E501
        ]
    )
    mock_post.return_value = response_mock

    with caplog.at_level(logging.WARNING):
        mock_openai_user.chat()

    assert "Error processing chunk: " in caplog.text


@patch("genai_bench.user.openai_user.requests.post")
def test_chat_significant_token_difference(mock_post, mock_openai_user, caplog):
    """Test warning when there's a significant difference in prompt tokens."""
    mock_openai_user.on_start()
    mock_openai_user.sample = lambda: UserChatRequest(
        model="gpt-3",
        prompt="Hello",
        num_prefill_tokens=5,
        additional_request_params={},
        max_tokens=10,
    )

    response_mock = MagicMock()
    response_mock.status_code = 200
    response_mock.iter_lines = MagicMock(
        return_value=[
            b'data: {"choices":[{"delta":{"content":"test"},"finish_reason":"stop"}]}',
            b'data: {"choices":[],"usage":{"prompt_tokens":100,"completion_tokens":1}}',
        ]
    )
    mock_post.return_value = response_mock

    with caplog.at_level(logging.WARNING):
        mock_openai_user.chat()

    assert "Significant difference detected in prompt tokens" in caplog.text
    assert (
        "differs from the number of prefill tokens returned by the sampler (5) by 95 tokens"  # noqa:E501
        in caplog.text
    )


@patch("genai_bench.user.openai_user.requests.post")
def test_chat_vision_without_prefill_tokens(mock_post, mock_openai_user):
    """Test chat with vision request without prefill tokens."""
    mock_openai_user.on_start()
    mock_openai_user.sample = lambda: UserImageChatRequest(
        model="gpt-4-vision",
        prompt="Describe this image",
        num_prefill_tokens=None,  # Vision request without prefill tokens
        image_content=["data:image/jpeg;base64,base64_image_content"],
        num_images=1,
        additional_request_params={},
        max_tokens=10,
    )

    response_mock = MagicMock()
    response_mock.status_code = 200
    response_mock.iter_lines = MagicMock(
        return_value=[
            b'data: {"choices":[{"delta":{"content":"A description"},"finish_reason":"stop"}]}',  # noqa:E501
            b'data: {"choices":[],"usage":{"prompt_tokens":50,"completion_tokens":3}}',
        ]
    )
    mock_post.return_value = response_mock

    response = mock_openai_user.send_request(
        stream=True,
        endpoint="/v1/test",
        payload={"key": "value"},
        num_prefill_tokens=None,
        parse_strategy=mock_openai_user.parse_chat_response,
    )
    assert (
        response.num_prefill_tokens == 50
    )  # Should use prompt_tokens as prefill tokens


@patch("genai_bench.user.openai_user.requests.post")
def test_vllm_model_format(mock_post, mock_openai_user):
    """Test handling of meta-llama/Meta-Llama-3-70B-Instruct format chunks."""
    mock_openai_user.environment.sampler = MagicMock()
    mock_openai_user.environment.sampler.get_token_length = (
        lambda text, add_special_tokens=True: len(text)
    )
    mock_openai_user.on_start()
    mock_openai_user.sample = lambda: UserChatRequest(
        model="gpt-3",
        prompt="Hello",
        num_prefill_tokens=5,
        additional_request_params={"ignore_eos": False},
        max_tokens=10,
    )

    response_mock = MagicMock()
    response_mock.status_code = 200
    response_mock.iter_lines = MagicMock(
        return_value=[
            b'data: {"id":"chatcmpl-213c0c2a84f145f1b7c934a794b6fc82","object":"chat.completion.chunk","created":1744238720,"model":"meta-llama/Meta-Llama-3-70B-Instruct","choices":[{"index":0,"delta":{"content":" a"},"logprobs":null,"finish_reason":null}]}',  # noqa:E501
            b"",
            b'data: {"id":"chatcmpl-213c0c2a84f145f1b7c934a794b6fc82","object":"chat.completion.chunk","created":1744238720,"model":"meta-llama/Meta-Llama-3-70B-Instruct","choices":[{"index":0,"delta":{"content":" sequence"},"logprobs":null,"finish_reason":"length","stop_reason":null}]}',  # noqa:E501
            b"",
            b'data: {"id":"chatcmpl-213c0c2a84f145f1b7c934a794b6fc82","object":"chat.completion.chunk","created":1744238720,"model":"meta-llama/Meta-Llama-3-70B-Instruct","choices":[],"usage":{"prompt_tokens":5,"total_tokens":7,"completion_tokens":2}}',  # noqa:E501
        ]
    )
    mock_post.return_value = response_mock

    response = mock_openai_user.send_request(
        stream=True,
        endpoint="/v1/test",
        payload={"key": "value"},
        num_prefill_tokens=5,
        parse_strategy=mock_openai_user.parse_chat_response,
    )

    assert response.status_code == 200
    assert response.generated_text == " a sequence"
    assert response.tokens_received == 2
    assert response.num_prefill_tokens == 5


@patch("genai_bench.user.openai_user.requests.post")
def test_openai_model_format(mock_post, mock_openai_user):
    """Test handling of OpenAI model format chunks."""
    mock_openai_user.environment.sampler = MagicMock()
    mock_openai_user.environment.sampler.get_token_length = (
        lambda text, add_special_tokens=True: len(text)
    )
    mock_openai_user.on_start()
    mock_openai_user.sample = lambda: UserChatRequest(
        model="gpt-3.5-turbo-0125",
        prompt="Hello",
        max_tokens=1,
    )

    response_mock = MagicMock()
    response_mock.status_code = 200
    response_mock.iter_lines = MagicMock(
        return_value=[
            # First chunk: role=assistant, empty content
            b'data: {"id": "chatcmpl-BmQFhYgMLvmAaaI8bPdesZN8z42iv", "choices": [{"delta": {"content": "", "function_call": null, "refusal": null, "role": "assistant", "tool_calls": null}, "finish_reason": null, "index": 0, "logprobs": null}], "created": 1750880357, "model": "gpt-3.5-turbo-0125", "object": "chat.completion.chunk", "service_tier": "default", "system_fingerprint": null, "usage": null}',  # noqa:E501
            # Second chunk: content "Hello"
            b'data: {"id": "chatcmpl-BmQFhYgMLvmAaaI8bPdesZN8z42iv", "choices": [{"delta": {"content": "Hello", "function_call": null, "refusal": null, "role": null, "tool_calls": null}, "finish_reason": null, "index": 0, "logprobs": null}], "created": 1750880357, "model": "gpt-3.5-turbo-0125", "object": "chat.completion.chunk", "service_tier": "default", "system_fingerprint": null, "usage": null}',  # noqa:E501
            # Third chunk: finish_reason "length", content is null
            b'data: {"id": "chatcmpl-BmQFhYgMLvmAaaI8bPdesZN8z42iv", "choices": [{"delta": {"content": null, "function_call": null, "refusal": null, "role": null, "tool_calls": null}, "finish_reason": "length", "index": 0, "logprobs": null}], "created": 1750880357, "model": "gpt-3.5-turbo-0125", "object": "chat.completion.chunk", "service_tier": "default", "system_fingerprint": null, "usage": null}',  # noqa:E501
            # Fourth chunk: usage info
            b'data: {"id": "chatcmpl-BmQFhYgMLvmAaaI8bPdesZN8z42iv", "choices": [], "created": 1750880357, "model": "gpt-3.5-turbo-0125", "object": "chat.completion.chunk", "service_tier": "default", "system_fingerprint": null, "usage": {"completion_tokens": 1, "prompt_tokens": 8, "total_tokens": 9, "completion_tokens_details": {"accepted_prediction_tokens": 0, "audio_tokens": 0, "reasoning_tokens": 0, "rejected_prediction_tokens": 0}, "prompt_tokens_details": {"audio_tokens": 0, "cached_tokens": 0}}}',  # noqa:E501
        ]
    )
    mock_post.return_value = response_mock

    response = mock_openai_user.send_request(
        stream=True,
        endpoint="/v1/test",
        payload={"key": "value"},
        parse_strategy=mock_openai_user.parse_chat_response,
    )

    assert response.status_code == 200
    assert response.generated_text == "Hello"
    assert response.tokens_received == 1
    assert response.num_prefill_tokens == 8


@patch("genai_bench.user.openai_user.requests.post")
def test_sgl_model_format(mock_post, mock_openai_user):
    """Test handling of sgl-model format chunks."""
    mock_openai_user.environment.sampler = MagicMock()
    mock_openai_user.environment.sampler.get_token_length = (
        lambda text, add_special_tokens=True: len(text)
    )
    mock_openai_user.on_start()
    mock_openai_user.sample = lambda: UserChatRequest(
        model="gpt-3",
        prompt="Hello",
        num_prefill_tokens=5,
        additional_request_params={"ignore_eos": False},
        max_tokens=10,
    )

    response_mock = MagicMock()
    response_mock.status_code = 200
    response_mock.iter_lines = MagicMock(
        return_value=[
            b'data: {"id":"4e28a148aa324b98b91853b724469d91","object":"chat.completion.chunk","created":1744317699,"model":"sgl-model","choices":[{"index":0,"delta":{"role":null,"content":" on","reasoning_content":null,"tool_calls":null},"logprobs":null,"finish_reason":null,"matched_stop":null}]}',  # noqa:E501
            b"",
            b'data: {"id":"4e28a148aa324b98b91853b724469d91","object":"chat.completion.chunk","created":1744317699,"model":"sgl-model","choices":[{"index":0,"delta":{"role":null,"content":null,"reasoning_content":null,"tool_calls":null},"logprobs":null,"finish_reason":"length","matched_stop":null}],"usage":{"prompt_tokens":5,"total_tokens":6,"completion_tokens":1,"prompt_tokens_details":null}}',  # noqa:E501
        ]
    )
    mock_post.return_value = response_mock

    response = mock_openai_user.send_request(
        stream=True,
        endpoint="/v1/test",
        payload={"key": "value"},
        num_prefill_tokens=5,
        parse_strategy=mock_openai_user.parse_chat_response,
    )

    assert response.status_code == 200
    assert response.generated_text == " on"
    assert response.tokens_received == 1
    assert response.num_prefill_tokens == 5


@patch("genai_bench.user.openai_user.requests.post")
def test_chat_with_reasoning_content_and_token_estimation(
    mock_post,
    mock_openai_user,
    caplog,
):
    """
    Ensure TTFT is triggered by reasoning_content,
    generated_text excludes it, and token estimation includes both
    reasoning_content + content when usage is missing.
    """
    mock_openai_user.on_start()
    mock_openai_user.sample = lambda: UserChatRequest(
        model="gpt-oss-20b-h100-chat",
        prompt="Why is the sky blue?",
        num_prefill_tokens=5,
        additional_request_params={},
        max_tokens=20,
    )

    # Prepare text pieces
    reasoning_text = "Thinking..."
    final_text = "The sky is blue"
    combined_text = reasoning_text + final_text

    # Mock sampler so token estimation equals len(combined_text)
    mock_openai_user.environment.sampler = MagicMock()
    mock_openai_user.environment.sampler.get_token_length.return_value = len(
        combined_text
    )

    # Stream: first reasoning_content, then content,
    # then a final chunk without usage (forces estimation)
    response_mock = MagicMock()
    response_mock.status_code = 200
    response_mock.iter_lines = MagicMock(
        return_value=[
            (
                b'data: {"id": "chat-xxx", "choices": [{"delta": '
                b'{"reasoning_content": "Thinking..."}, "index": 0}], '
                b'"model": "gpt-oss-llama-3"}'
            ),
            (
                b'data: {"id": "chat-xxx", "choices": [{"delta": '
                b'{"content": "The sky is blue"}, "index": 0}], '
                b'"model": "gpt-oss-llama-3"}'
            ),
            (
                b'data: {"id": "chat-xxx", "choices": [{"delta": {}, '
                b'"finish_reason": "stop"}]}'
            ),
            b"data: [DONE]",
        ]
    )

    mock_post.return_value = response_mock

    with caplog.at_level(logging.WARNING):
        # Call send_request directly to get a UserChatResponse
        resp = mock_openai_user.send_request(
            stream=True,
            endpoint="/v1/test",
            payload={"key": "value"},
            num_prefill_tokens=5,
            parse_strategy=mock_openai_user.parse_chat_response,
        )

    # Assertions: got a UserChatResponse
    assert isinstance(resp, UserChatResponse)
    assert resp.status_code == 200
    assert resp.time_at_first_token is not None

    # generated_text should include reasoning_content
    assert resp.generated_text == combined_text

    # Warning about missing usage must be present
    assert "There is no usage info returned from the model server" in caplog.text

    # Token estimation must be based on reasoning + content
    assert resp.tokens_received == len(combined_text)
    mock_openai_user.environment.sampler.get_token_length.assert_called_once_with(
        combined_text, add_special_tokens=False
    )


@patch("genai_bench.user.openai_user.requests.post")
def test_ignore_eos_vllm_backend_default(mock_post, mock_openai_user):
    """Test that ignore_eos is added with default value for vLLM backend."""
    mock_openai_user.on_start()
    mock_openai_user.api_backend = "vllm"
    mock_openai_user.sample = lambda: UserChatRequest(
        model="gpt-3",
        prompt="Hello",
        num_prefill_tokens=5,
        additional_request_params={},
        max_tokens=10,
    )

    response_mock = MagicMock()
    response_mock.status_code = 200
    response_mock.iter_lines = MagicMock(
        return_value=[
            b'data: {"id":"chat-1","object":"chat.completion.chunk","created":1744238720,"model":"gpt-3","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}',  # noqa: E501
            b'data: {"id":"chat-1","object":"chat.completion.chunk","created":1744238720,"model":"gpt-3","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}',  # noqa: E501
            b"data: [DONE]",
        ]
    )
    mock_post.return_value = response_mock

    mock_openai_user.chat()

    # Verify that ignore_eos was added to the payload
    call_args = mock_post.call_args
    payload = call_args.kwargs["json"]
    assert "ignore_eos" in payload
    assert payload["ignore_eos"] is True  # bool(max_tokens=10) is True


@patch("genai_bench.user.openai_user.requests.post")
def test_ignore_eos_sglang_backend_explicit_false(mock_post, mock_openai_user):
    """Test that explicit ignore_eos=False is preserved for SGLang backend."""
    mock_openai_user.on_start()
    mock_openai_user.api_backend = "sglang"
    mock_openai_user.sample = lambda: UserChatRequest(
        model="gpt-3",
        prompt="Hello",
        num_prefill_tokens=5,
        additional_request_params={"ignore_eos": False},
        max_tokens=10,
    )

    response_mock = MagicMock()
    response_mock.status_code = 200
    response_mock.iter_lines = MagicMock(
        return_value=[
            b'data: {"id":"chat-1","object":"chat.completion.chunk","created":1744238720,"model":"gpt-3","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}',  # noqa: E501
            b'data: {"id":"chat-1","object":"chat.completion.chunk","created":1744238720,"model":"gpt-3","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}',  # noqa: E501
            b"data: [DONE]",
        ]
    )
    mock_post.return_value = response_mock

    mock_openai_user.chat()

    # Verify that user's explicit ignore_eos=False is preserved
    call_args = mock_post.call_args
    payload = call_args.kwargs["json"]
    assert "ignore_eos" in payload
    assert payload["ignore_eos"] is False


@patch("genai_bench.user.openai_user.requests.post")
def test_ignore_eos_openai_backend_removed(mock_post, mock_openai_user):
    """Test that ignore_eos is removed for OpenAI backend."""
    mock_openai_user.on_start()
    mock_openai_user.api_backend = "openai"
    mock_openai_user.sample = lambda: UserChatRequest(
        model="gpt-3",
        prompt="Hello",
        num_prefill_tokens=5,
        additional_request_params={"ignore_eos": True},
        max_tokens=10,
    )

    response_mock = MagicMock()
    response_mock.status_code = 200
    response_mock.iter_lines = MagicMock(
        return_value=[
            b'data: {"id":"chat-1","object":"chat.completion.chunk","created":1744238720,"model":"gpt-3","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}',  # noqa: E501
            b'data: {"id":"chat-1","object":"chat.completion.chunk","created":1744238720,"model":"gpt-3","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}',  # noqa: E501
            b"data: [DONE]",
        ]
    )
    mock_post.return_value = response_mock

    mock_openai_user.chat()

    # Verify that ignore_eos was removed from the payload
    call_args = mock_post.call_args
    payload = call_args.kwargs["json"]
    assert "ignore_eos" not in payload


@patch("genai_bench.user.openai_user.requests.post")
def test_chat_with_system_message(mock_post, mock_openai_user):
    """Test that system message is added to messages array without replacing
    user message."""
    mock_openai_user.on_start()
    mock_openai_user.sample = lambda: UserChatRequest(
        model="gpt-3",
        prompt="Hello",
        num_prefill_tokens=5,
        additional_request_params={"system_message": "You are a helpful assistant."},
        max_tokens=10,
    )

    response_mock = MagicMock()
    response_mock.status_code = 200
    response_mock.iter_lines = MagicMock(
        return_value=[
            b'data: {"id":"chat-1","object":"chat.completion.chunk","created":1744238720,"model":"gpt-3","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}',  # noqa: E501
            b'data: {"id":"chat-1","object":"chat.completion.chunk","created":1744238720,"model":"gpt-3","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":15,"completion_tokens":1,"total_tokens":16}}',  # noqa: E501
            b"data: [DONE]",
        ]
    )
    mock_post.return_value = response_mock

    mock_openai_user.chat()

    # Verify the request payload
    call_args = mock_post.call_args
    payload = call_args.kwargs["json"]

    # Check that messages array has both system and user messages
    assert "messages" in payload
    messages = payload["messages"]
    assert len(messages) == 2

    # First message should be system message
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are a helpful assistant."

    # Second message should be user message with the prompt from traffic scenario
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Hello"

    # Verify system_message is not in the payload (it's been filtered out)
    assert "system_message" not in payload


@patch("genai_bench.user.openai_user.requests.post")
def test_chat_with_system_message_and_vision(mock_post, mock_openai_user):
    """Test that system message works with vision (multimodal) requests."""
    mock_openai_user.on_start()
    mock_openai_user.sample = lambda: UserImageChatRequest(
        model="gpt-4-vision",
        prompt="What's in this image?",
        num_prefill_tokens=10,
        image_content=["data:image/jpeg;base64,base64_image_content"],
        num_images=1,
        additional_request_params={"system_message": "You are an image analyst."},
        max_tokens=20,
    )

    response_mock = MagicMock()
    response_mock.status_code = 200
    response_mock.iter_lines = MagicMock(
        return_value=[
            b'data: {"id":"chat-1","object":"chat.completion.chunk","created":1744238720,"model":"gpt-4-vision","choices":[{"index":0,"delta":{"content":"A cat"},"finish_reason":null}]}',  # noqa: E501
            b'data: {"id":"chat-1","object":"chat.completion.chunk","created":1744238720,"model":"gpt-4-vision","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":20,"completion_tokens":2,"total_tokens":22}}',  # noqa: E501
            b"data: [DONE]",
        ]
    )
    mock_post.return_value = response_mock

    mock_openai_user.chat()

    # Verify the request payload
    call_args = mock_post.call_args
    payload = call_args.kwargs["json"]

    # Check that messages array has both system and user messages
    assert "messages" in payload
    messages = payload["messages"]
    assert len(messages) == 2

    # First message should be system message
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are an image analyst."

    # Second message should be user message with multimodal content
    assert messages[1]["role"] == "user"
    assert isinstance(messages[1]["content"], list)
    # Should have both text and image content
    assert messages[1]["content"][0]["type"] == "text"
    assert messages[1]["content"][0]["text"] == "What's in this image?"
    assert messages[1]["content"][1]["type"] == "image_url"
    assert "base64_image_content" in messages[1]["content"][1]["image_url"]["url"]

    # Verify system_message is not in the payload (it's been filtered out)
    assert "system_message" not in payload
