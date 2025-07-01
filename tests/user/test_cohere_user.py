from unittest.mock import MagicMock, patch

import pytest
import requests

from genai_bench.protocol import (
    UserChatRequest,
    UserChatResponse,
    UserEmbeddingRequest,
    UserImageEmbeddingRequest,
)
from genai_bench.user.cohere_user import CohereUser


@pytest.fixture
def mock_response():
    response = MagicMock(spec=requests.Response)
    response.status_code = 200
    return response


@pytest.fixture
def mock_env():
    # Create a mock environment instead of real Locust Environment
    env = MagicMock()
    env.host = "https://api.cohere.com"
    env.sampler = MagicMock()
    return env


@pytest.fixture
def cohere_user(mock_env):
    # Create the user with the mock environment
    CohereUser.host = "https://api.cohere.com"

    # Set up mock auth provider
    mock_auth = MagicMock()
    mock_auth.get_credentials.return_value = "test-key"
    mock_auth.get_config.return_value = {
        "api_base": "https://api.cohere.com",
        "api_key": "test-key",
    }
    CohereUser.auth_provider = mock_auth

    user = CohereUser(mock_env)
    user.headers = {"Authorization": "Bearer test-key"}
    return user


def test_on_start_missing_config():
    env = MagicMock()
    CohereUser.host = "https://api.cohere.com"
    user = CohereUser(env)
    user.host = None
    user.auth_signer = None
    with pytest.raises(
        ValueError, match="API key and base must be set for CohereUser."
    ):
        user.on_start()


def test_on_start_success(cohere_user):
    cohere_user.on_start()
    assert cohere_user.headers == {
        "Authorization": "Bearer test-key",
        "Content-Type": "application/json",
    }


def test_chat_success(cohere_user, mock_response):
    # Mock the streaming response for chat
    chat_events = [
        'event: message-start\ndata: {"id":"123","type":"message-start","delta":{"message":{"role":"assistant","content":[]}}}\n\n',  # noqa:E501
        'event: content-start\ndata: {"type":"content-start","index":0,"delta":{"message":{"content":{"type":"text","text":""}}}}\n\n',  # noqa:E501
        'event: content-delta\ndata: {"type":"content-delta","index":0,"delta":{"message":{"content":{"text":"Hello"}}}}\n\n',  # noqa:E501
        'event: content-delta\ndata: {"type":"content-delta","index":0,"delta":{"message":{"content":{"text":" world"}}}}\n\n',  # noqa:E501
        'event: content-end\ndata: {"type":"content-end","index":0}\n\n',
        'event: message-end\ndata: {"type":"message-end","delta":{"finish_reason":"COMPLETE","usage":{"tokens":{"input_tokens":68,"output_tokens":22}}}}\n\n',  # noqa:E501
        "data: [DONE]\n\n",
    ]

    mock_response.iter_lines.return_value = [line.encode() for line in chat_events]

    with patch("requests.post", return_value=mock_response):
        # Create a sample chat request
        chat_request = UserChatRequest(
            model="command-r",
            prompt="Hello world!",
            max_tokens=100,
            num_prefill_tokens=5,
            additional_request_params={},
        )
        cohere_user.sample = MagicMock(return_value=chat_request)

        # Execute chat
        response = cohere_user.chat()

        # Verify response
        assert isinstance(response, UserChatResponse)
        assert response.status_code == 200
        assert response.generated_text == "Hello world"
        assert response.tokens_received == 22
        assert response.num_prefill_tokens == 5
        assert response.time_at_first_token is not None


def test_chat_without_usage(cohere_user, mock_response):
    # Mock the streaming response for chat
    chat_events = []

    mock_response.iter_lines.return_value = [line.encode() for line in chat_events]

    with patch("requests.post", return_value=mock_response):
        # Create a sample chat request
        chat_request = UserChatRequest(
            model="command-r",
            prompt="Hello world!",
            max_tokens=100,
            num_prefill_tokens=5,
            additional_request_params={},
        )
        cohere_user.sample = MagicMock(return_value=chat_request)
        with pytest.raises(AssertionError):
            # Execute chat
            response = cohere_user.chat()

            # Verify response
            assert isinstance(response, UserChatResponse)
            assert response.status_code == 200
            assert response.generated_text == "Hello world"
            assert response.tokens_received == 22
            assert response.num_prefill_tokens == 5
            assert response.time_at_first_token is not None


def test_embeddings_text_success(cohere_user, mock_response):
    # Mock the embedding response for text
    embedding_response = {
        "id": "test-id",
        "texts": ["test text"],
        "embeddings": {"float": [[0.1, 0.2, 0.3]]},
        "meta": {"api_version": {"version": "2"}, "billed_units": {"input_tokens": 20}},
    }
    mock_response.json.return_value = embedding_response

    with patch("requests.post", return_value=mock_response):
        # Create a sample embedding request
        embedding_request = UserEmbeddingRequest(
            model="embed-english-v3.0",
            documents=["test text"],
            num_prefill_tokens=3,
            additional_request_params={},
        )
        cohere_user.sample = MagicMock(return_value=embedding_request)

        # Execute embeddings
        response = cohere_user.embeddings()

        # Verify response
        assert response.status_code == 200
        assert response.num_prefill_tokens == 3
        assert response.time_at_first_token is not None


def test_embeddings_text_chatRequest(cohere_user, mock_response):
    with patch("requests.post", return_value=mock_response):
        # Create a sample embedding request
        chat_request = UserChatRequest(
            model="command-r",
            prompt="Hello world!",
            max_tokens=100,
            num_prefill_tokens=5,
            additional_request_params={},
        )
        cohere_user.sample = MagicMock(return_value=chat_request)

        with pytest.raises(AttributeError):
            # Execute embeddings
            cohere_user.embeddings()


def test_embeddings_text_no_documents(cohere_user, mock_response):
    # Mock the embedding response for text
    embedding_response = {
        "id": "test-id",
        "texts": ["test text"],
        "embeddings": {"float": [[0.1, 0.2, 0.3]]},
        "meta": {"api_version": {"version": "2"}, "billed_units": {"input_tokens": 20}},
    }
    mock_response.json.return_value = embedding_response

    with patch("requests.post", return_value=mock_response):
        # Create a sample embedding request
        embedding_request = UserEmbeddingRequest(
            model="embed-english-v3.0",
            documents=["test text"],
            num_prefill_tokens=0,
            additional_request_params={},
        )
        cohere_user.sample = MagicMock(return_value=embedding_request)

        # Execute embeddings
        response = cohere_user.embeddings()

        # Verify response
        assert response.status_code == 200
        assert response.time_at_first_token is not None


def test_embeddings_image_success(cohere_user, mock_response):
    # Mock the embedding response for image
    embedding_response = {
        "id": "test-id",
        "images": [{"width": 400, "height": 400}],
        "embeddings": {"float": [[0.1, 0.2, 0.3]]},
        "meta": {"api_version": {"version": "2"}, "billed_units": {"images": 1}},
    }
    mock_response.json.return_value = embedding_response

    with patch("requests.post", return_value=mock_response):
        # Create a sample image embedding request
        embedding_request = UserImageEmbeddingRequest(
            documents=[],
            model="embed-english-v3.0",
            image_content=["base64_image_content"],
            num_images=1,
            num_prefill_tokens=1,
            additional_request_params={},
        )
        cohere_user.sample = MagicMock(return_value=embedding_request)

        # Execute embeddings
        response = cohere_user.embeddings()

        # Verify response
        assert response.status_code == 200
        assert response.num_prefill_tokens == 1
        assert response.time_at_first_token is not None


def test_embeddings_multiple_images_error(cohere_user):
    # Create a sample image embedding request with multiple images
    embedding_request = UserImageEmbeddingRequest(
        documents=[],
        model="embed-english-v3.0",
        image_content=["image1", "image2"],
        num_images=2,
        num_prefill_tokens=2,
        additional_request_params={},
    )
    cohere_user.sample = MagicMock(return_value=embedding_request)

    # Verify that attempting to embed multiple images raises an error
    with pytest.raises(
        ValueError, match="OCI-Cohere Image embedding supports only 1 image"
    ):
        cohere_user.embeddings()


def test_chat_request_error(cohere_user):
    with patch(
        "requests.post",
        side_effect=requests.exceptions.RequestException("Network error"),
    ):
        chat_request = UserChatRequest(
            model="command-r",
            prompt="Hello",
            max_tokens=100,
            num_prefill_tokens=1,
            additional_request_params={},
        )
        cohere_user.sample = MagicMock(return_value=chat_request)

        user_response = cohere_user.chat()
        assert user_response.status_code == 500


def test_chat_invalid_json_response(cohere_user, mock_response):
    # Mock an invalid JSON response
    mock_response.iter_lines.return_value = [b"data: {invalid json}"]

    with patch("requests.post", return_value=mock_response):
        chat_request = UserChatRequest(
            model="command-r",
            prompt="Hello",
            max_tokens=100,
            num_prefill_tokens=1,
            additional_request_params={},
        )
        cohere_user.sample = MagicMock(return_value=chat_request)

        # The chat should complete but log a warning about the invalid JSON
        response = cohere_user.chat()
        assert isinstance(response, UserChatResponse)
        assert response.generated_text == ""
