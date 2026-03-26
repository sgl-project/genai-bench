import json
from unittest.mock import MagicMock, patch

import pytest
from oci.generative_ai_inference.models import (
    ChatDetails,
    CohereChatRequest,
    CohereChatRequestV2,
    CohereImageContentV2,
    CohereTextContentV2,
    DedicatedServingMode,
    EmbedTextDetails,
    OnDemandServingMode,
    RerankTextDetails,
)

from genai_bench.protocol import (
    UserChatRequest,
    UserEmbeddingRequest,
    UserImageChatRequest,
    UserImageEmbeddingRequest,
    UserReRankRequest,
)
from genai_bench.user.oci_cohere_user import OCICohereUser


@pytest.fixture
def test_cohere_user():
    OCICohereUser.host = "http://example.com"
    OCICohereUser.api_version = "v1"

    # Set up mock auth provider
    mock_auth = MagicMock()
    mock_auth.get_credentials.return_value = "test-key"
    mock_auth.get_config.return_value = {
        "api_base": "http://example.com",
        "compartment_id": "test-compartment",
        "endpoint_id": "test-endpoint",
    }
    OCICohereUser.auth_provider = mock_auth

    user = OCICohereUser(environment=MagicMock())
    try:
        yield user
    finally:
        OCICohereUser.api_version = "v1"


def test_supported_tasks_include_vision_chat():
    assert OCICohereUser.supported_tasks["image-text-to-text"] == "chat"


@patch("genai_bench.user.oci_cohere_user.GenerativeAiInferenceClient")
def test_chat_v2_text(mock_client_class, test_cohere_user):
    mock_client_instance = mock_client_class.return_value
    mock_client_instance.chat.return_value.status = 200
    mock_client_instance.chat.return_value.data.events.return_value = iter(
        [
            MagicMock(
                data=json.dumps(
                    {
                        "message": {
                            "role": "ASSISTANT",
                            "content": [{"type": "TEXT", "text": "Hello"}],
                        }
                    }
                ).encode("utf-8")
            ),
            MagicMock(
                data=json.dumps(
                    {
                        "message": {
                            "role": "ASSISTANT",
                            "content": [
                                {"type": "THINKING", "thinking": " Reason"},
                            ],
                        }
                    }
                ).encode("utf-8")
            ),
            MagicMock(
                data=json.dumps(
                    {
                        "finishReason": "COMPLETE",
                        "usage": {
                            "tokens": {
                                "input_tokens": 5,
                                "output_tokens": 3,
                                "reasoning_tokens": 2,
                            }
                        },
                        "message": {
                            "role": "ASSISTANT",
                            "content": [{"type": "TEXT", "text": "!"}],
                        },
                    }
                ).encode("utf-8")
            ),
        ]
    )

    test_cohere_user.api_version = "v2"
    test_cohere_user.on_start()
    history = [{"role": "assistant", "content": "Previous turn"}]
    documents = [{"text": "Doc"}]
    test_cohere_user.sample = lambda: UserChatRequest(
        model="cohere-command-a",
        prompt="Describe the data",
        num_prefill_tokens=1,
        additional_request_params={
            "compartmentId": "ocid1.compartment.oc1..example",
            "servingType": "ON_DEMAND",
            "system_message": "Be concise",
            "chat_history": history,
            "documents": documents,
            "top_p": 0.9,
            "temperature": 0.3,
        },
        max_tokens=64,
    )

    metrics_collector_mock = MagicMock()
    test_cohere_user.collect_metrics = metrics_collector_mock

    test_cohere_user.chat()

    chat_detail = mock_client_instance.chat.call_args[0][0]
    chat_request = chat_detail.chat_request
    assert isinstance(chat_request, CohereChatRequestV2)
    assert chat_request.top_p == 0.9
    assert chat_request.temperature == 0.3
    assert chat_request.documents == documents
    assert chat_request.stream_options.is_include_usage
    assert chat_request.messages[0].role == "SYSTEM"
    assert isinstance(chat_request.messages[0].content[0], CohereTextContentV2)

    metrics_collector_mock.assert_called_once()
    user_response = metrics_collector_mock.call_args[0][0]
    assert user_response.tokens_received == 3
    assert user_response.reasoning_tokens == 2
    assert user_response.num_prefill_tokens == 5
    assert user_response.status_code == 200


@patch("genai_bench.user.oci_cohere_user.GenerativeAiInferenceClient")
def test_chat_v2_vision(mock_client_class, test_cohere_user):
    mock_client_instance = mock_client_class.return_value
    mock_client_instance.chat.return_value.status = 200
    mock_client_instance.chat.return_value.data.events.return_value = iter(
        [
            MagicMock(
                data=json.dumps(
                    {
                        "message": {
                            "role": "ASSISTANT",
                            "content": [{"type": "TEXT", "text": "Analyzing"}],
                        }
                    }
                ).encode("utf-8")
            ),
            MagicMock(
                data=json.dumps(
                    {
                        "finishReason": "COMPLETE",
                        "message": {
                            "role": "ASSISTANT",
                            "content": [{"type": "TEXT", "text": " image"}],
                        },
                    }
                ).encode("utf-8")
            ),
        ]
    )

    test_cohere_user.api_version = "v2"
    test_cohere_user.on_start()
    image_data = "data:image/png;base64,AAA"
    test_cohere_user.sample = lambda: UserImageChatRequest(
        model="cohere-command-a-vision",
        prompt="What do you see?",
        image_content=[image_data],
        num_images=1,
        num_prefill_tokens=None,
        additional_request_params={
            "compartmentId": "ocid1.compartment.oc1..example",
            "servingType": "ON_DEMAND",
        },
        max_tokens=32,
    )

    metrics_collector_mock = MagicMock()
    test_cohere_user.collect_metrics = metrics_collector_mock

    test_cohere_user.chat()

    chat_detail = mock_client_instance.chat.call_args[0][0]
    chat_request = chat_detail.chat_request
    user_message = chat_request.messages[-1]
    image_contents = [
        content
        for content in user_message.content
        if isinstance(content, CohereImageContentV2)
    ]
    assert len(image_contents) == 1
    assert image_contents[0].image_url.url == image_data
    metrics_collector_mock.assert_called_once()


@patch("genai_bench.user.oci_cohere_user.GenerativeAiInferenceClient")
def test_chat_v2_vision_multiple_images_error(mock_client_class, test_cohere_user):
    mock_client_class.return_value.chat.return_value.status = 200

    test_cohere_user.api_version = "v2"
    test_cohere_user.on_start()
    test_cohere_user.sample = lambda: UserImageChatRequest(
        model="cohere-command-a-vision",
        prompt="Describe both images",
        image_content=["img1", "img2"],
        num_images=2,
        num_prefill_tokens=None,
        additional_request_params={
            "compartmentId": "ocid1.compartment.oc1..example",
            "servingType": "ON_DEMAND",
        },
        max_tokens=32,
    )

    with pytest.raises(ValueError, match="only a single image"):
        test_cohere_user.chat()


def test_embeddings_use_legacy_parse_with_v2_enabled(test_cohere_user):
    test_cohere_user.api_version = "v2"
    mock_send = MagicMock()
    test_cohere_user.send_request = mock_send
    test_cohere_user.sample = lambda: UserEmbeddingRequest(
        documents=["text"],
        num_prefill_tokens=4,
        model="cohere-embed-v3",
        additional_request_params={
            "compartmentId": "ocid1.compartment.oc1..example",
            "servingType": "ON_DEMAND",
        },
    )

    test_cohere_user.embeddings()

    assert (
        mock_send.call_args.kwargs["parse_strategy"]
        == test_cohere_user.parse_embedding_response
    )


def test_rerank_use_legacy_parse_with_v2_enabled(test_cohere_user):
    test_cohere_user.api_version = "v2"
    mock_send = MagicMock()
    test_cohere_user.send_request = mock_send
    test_cohere_user.sample = lambda: UserReRankRequest(
        documents=["doc"],
        query="question",
        num_prefill_tokens=2,
        model="cohere-rerank",
        additional_request_params={
            "compartmentId": "ocid1.compartment.oc1..example",
            "servingType": "ON_DEMAND",
        },
    )

    test_cohere_user.rerank()

    assert (
        mock_send.call_args.kwargs["parse_strategy"]
        == test_cohere_user.parse_rerank_response
    )


@patch("genai_bench.user.oci_cohere_user.GenerativeAiInferenceClient")
def test_chat(mock_client_class, test_cohere_user):
    # Mock the chat method
    mock_client_instance = mock_client_class.return_value
    mock_client_instance.chat.return_value.status = 200
    mock_client_instance.chat.return_value.data.events.return_value = iter(
        [
            MagicMock(data=b'{"apiFormat":"COHERE","text":"The","pad":"aaaaaaaaa"}'),
            MagicMock(
                data=b'{"apiFormat":"COHERE","text":" relational","pad":"aaaaaaaaa"}'
            ),
            MagicMock(
                data=b'{"apiFormat":"COHERE","text":" database","pad":"aaaaaaaaa"}'
            ),
            MagicMock(data=b'{"apiFormat":"COHERE","text":".","pad":"aaaaaaaaa"}'),
            MagicMock(
                data=b'{"apiFormat":"COHERE","finishReason": "tokens", "text":""}'
            ),
            MagicMock(data=b"data: [DONE]"),
        ]
    )

    test_cohere_user.on_start()
    test_cohere_user.sample = lambda: UserChatRequest(
        model="cohere-model",
        prompt="Hello",
        num_prefill_tokens=5,
        additional_request_params={
            "compartmentId": "ocid1.compartment.oc1..example",
            "servingType": "ON_DEMAND",
        },
        max_tokens=10,
    )

    metrics_collector_mock = MagicMock()
    test_cohere_user.collect_metrics = metrics_collector_mock

    test_cohere_user.chat()

    mock_client_instance.chat.assert_called_once_with(
        ChatDetails(
            compartment_id="ocid1.compartment.oc1..example",
            serving_mode=OnDemandServingMode(model_id="cohere-model"),
            chat_request=CohereChatRequest(
                api_format="COHERE",
                message="Hello",
                max_tokens=10,
                is_stream=True,
                temperature=0.1,
                top_p=0,
                top_k=0.75,
                frequency_penalty=0,
                presence_penalty=0,
            ),
        )
    )
    metrics_collector_mock.assert_called_once()
    args, _ = metrics_collector_mock.call_args
    user_response = args[0]
    assert user_response.status_code == 200
    assert user_response.num_prefill_tokens == 5
    assert user_response.reasoning_tokens == 0


@patch("genai_bench.user.oci_cohere_user.GenerativeAiInferenceClient")
def test_chat_json_error(mock_client_class, test_cohere_user):
    # Mock the chat method
    mock_client_instance = mock_client_class.return_value
    mock_client_instance.chat.return_value.status = 200
    mock_client_instance.chat.return_value.data.events.return_value = iter(
        [
            MagicMock(data=b'{"apiFormat":"COHERE","text":"The","pad":"aaaaaaaaa"}'),
            MagicMock(
                data=b'{"apiFormat":"COHERE","text":" relational","pad":"aaaaaaaaa"}'
            ),
            MagicMock(
                data=b'{"apiFormat":"COHERE","text":" database","pad":"aaaaaaaaa"}'
            ),
            MagicMock(data=b'{"apiFormat":"COHERE","text":".","pad":"aaaaaaaaa"}'),
            MagicMock(data=b"data: [DONE]"),
        ]
    )

    test_cohere_user.on_start()
    test_cohere_user.sample = lambda: UserChatRequest(
        model="cohere-model",
        prompt="Hello",
        num_prefill_tokens=5,
        additional_request_params={
            "compartmentId": "ocid1.compartment.oc1..example",
            "servingType": "ON_DEMAND",
        },
        max_tokens=10,
    )

    metrics_collector_mock = MagicMock()
    test_cohere_user.collect_metrics = metrics_collector_mock

    test_cohere_user.chat()

    mock_client_instance.chat.assert_called_once_with(
        ChatDetails(
            compartment_id="ocid1.compartment.oc1..example",
            serving_mode=OnDemandServingMode(model_id="cohere-model"),
            chat_request=CohereChatRequest(
                api_format="COHERE",
                message="Hello",
                max_tokens=10,
                is_stream=True,
                temperature=0.1,
                top_p=0,
                top_k=0.75,
                frequency_penalty=0,
                presence_penalty=0,
            ),
        )
    )
    metrics_collector_mock.assert_called_once()
    args, _ = metrics_collector_mock.call_args
    user_response = args[0]
    assert user_response.status_code == 200
    assert user_response.num_prefill_tokens == 5


@patch("genai_bench.user.oci_cohere_user.GenerativeAiInferenceClient")
def test_chat_with_embedding_request(mock_client_class, test_cohere_user):
    test_cohere_user.on_start()
    test_cohere_user.sample = lambda: UserEmbeddingRequest(
        documents=[],
        num_prefill_tokens=10,
        model="cohere-embed-v3",
        additional_request_params={
            "servingType": "ON_DEMAND",
            "compartmentId": "compartmentId",
        },
    )

    metrics_collector_mock = MagicMock()
    test_cohere_user.collect_metrics = metrics_collector_mock
    with pytest.raises(AttributeError):
        test_cohere_user.chat()


@patch("genai_bench.user.oci_cohere_user.logger")
@patch("genai_bench.user.oci_cohere_user.GenerativeAiInferenceClient")
def test_chat_with_response_error(mock_client_class, mock_logger, test_cohere_user):
    # Mock the chat method
    mock_client_instance = mock_client_class.return_value
    mock_client_instance.chat.return_value.status = 401
    mock_client_instance.chat.return_value.data.events.side_effect = [
        MagicMock(data=b'{"apiFormat":"COHERE","text":"The","pad":"aaaaaaaaa"}'),
        MagicMock(
            data=b'{"apiFormat":"COHERE","text":" relational","pad":"aaaaaaaaa"}'
        ),
        MagicMock(data=b'{"apiFormat":"COHERE","text":" database","pad":"aaaaaaaaa"}'),
        MagicMock(data=b'{"apiFormat":"COHERE","text":".","pad":"aaaaaaaaa"}'),
        MagicMock(data=b"data: [DONE]"),
    ]

    test_cohere_user.on_start()
    test_cohere_user.sample = lambda: UserChatRequest(
        model="cohere-model",
        prompt="Hello",
        num_prefill_tokens=5,
        additional_request_params={
            "compartmentId": "ocid1.compartment.oc1..example",
            "servingType": "ON_DEMAND",
        },
        max_tokens=10,
    )

    metrics_collector_mock = MagicMock()
    test_cohere_user.collect_metrics = metrics_collector_mock

    test_cohere_user.chat()

    mock_client_instance.chat.assert_called_once_with(
        ChatDetails(
            compartment_id="ocid1.compartment.oc1..example",
            serving_mode=OnDemandServingMode(model_id="cohere-model"),
            chat_request=CohereChatRequest(
                api_format="COHERE",
                message="Hello",
                max_tokens=10,
                is_stream=True,
                temperature=0.1,
                top_p=0,
                top_k=0.75,
                frequency_penalty=0,
                presence_penalty=0,
            ),
        )
    )
    metrics_collector_mock.assert_called_once()
    args, _ = metrics_collector_mock.call_args
    user_response = args[0]
    assert user_response.status_code == 401


def test_send_request_collects_metrics_on_exception(test_cohere_user):
    """Test that exceptions are caught and metrics are collected."""
    metrics_collector_mock = MagicMock()
    test_cohere_user.collect_metrics = metrics_collector_mock

    def raise_error():
        raise ValueError("Something went wrong")

    response = test_cohere_user.send_request(
        make_request=raise_error,
        endpoint="chat",
        payload=None,
        parse_strategy=None,
        num_prefill_tokens=5,
    )

    # Verify collect_metrics was called with correct parameters
    metrics_collector_mock.assert_called_once()
    call_args = metrics_collector_mock.call_args
    user_response = call_args[0][0]
    endpoint = call_args[0][1]

    # Verify metrics are properly collected
    assert user_response.status_code == 500
    assert response.status_code == 500
    assert user_response.num_prefill_tokens == 5
    assert endpoint == "chat"
    assert "Something went wrong" in user_response.error_message


@patch("genai_bench.user.oci_cohere_user.GenerativeAiInferenceClient")
def test_embed(mock_client_class, test_cohere_user):
    mock_client_instance = mock_client_class.return_value
    mock_client_instance.embed_text.return_value.status = 200

    test_cohere_user.on_start()
    documents = ["The earth has a radius of 6,454km."]
    model = "cohere-embed-v3"
    test_cohere_user.sample = lambda: UserEmbeddingRequest(
        documents=documents,
        num_prefill_tokens=10,
        model=model,
        additional_request_params={
            "servingType": "ON_DEMAND",
            "compartmentId": "compartmentId",
        },
    )

    metrics_collector_mock = MagicMock()
    test_cohere_user.collect_metrics = metrics_collector_mock

    test_cohere_user.embeddings()

    mock_client_instance.embed_text.assert_called_once_with(
        EmbedTextDetails(
            inputs=documents,
            input_type="SEARCH_DOCUMENT",
            compartment_id="compartmentId",
            serving_mode=OnDemandServingMode(model_id=model),
            truncate="NONE",
        )
    )
    metrics_collector_mock.assert_called_once()
    args, _ = metrics_collector_mock.call_args
    user_response = args[0]
    assert user_response.status_code == 200
    assert user_response.num_prefill_tokens == 10


@patch("genai_bench.user.oci_cohere_user.GenerativeAiInferenceClient")
def test_embed_with_prefill_tokens(mock_client_class, test_cohere_user):
    mock_client_instance = mock_client_class.return_value
    mock_client_instance.embed_text.return_value.status = 200

    test_cohere_user.on_start()
    documents = ["The earth has a radius of 6,454km."]
    model = "cohere-embed-v3"
    test_cohere_user.sample = lambda: UserEmbeddingRequest(
        documents=documents,
        num_prefill_tokens=0,
        model=model,
        additional_request_params={
            "servingType": "ON_DEMAND",
            "compartmentId": "compartmentId",
        },
    )

    metrics_collector_mock = MagicMock()
    test_cohere_user.collect_metrics = metrics_collector_mock

    test_cohere_user.embeddings()

    mock_client_instance.embed_text.assert_called_once_with(
        EmbedTextDetails(
            inputs=documents,
            input_type="SEARCH_DOCUMENT",
            compartment_id="compartmentId",
            serving_mode=OnDemandServingMode(model_id=model),
            truncate="NONE",
        )
    )
    metrics_collector_mock.assert_called_once()
    args, _ = metrics_collector_mock.call_args
    user_response = args[0]
    assert user_response.status_code == 200
    assert user_response.num_prefill_tokens == 0


@patch("genai_bench.user.oci_cohere_user.GenerativeAiInferenceClient")
def test_image_embeddings(mock_client_class, test_cohere_user):
    mock_client_instance = mock_client_class.return_value
    mock_client_instance.embed_text.return_value.status = 200

    test_cohere_user.on_start()
    images = ["data:image/jpeg;base64,BASE64Image1"]
    model = "cohere-embed-v3"
    test_cohere_user.sample = lambda: UserImageEmbeddingRequest(
        documents=[],
        image_content=images,
        num_images=2,
        num_prefill_tokens=10,
        model=model,
        additional_request_params={
            "servingType": "ON_DEMAND",
            "compartmentId": "compartmentId",
        },
    )

    metrics_collector_mock = MagicMock()
    test_cohere_user.collect_metrics = metrics_collector_mock

    test_cohere_user.embeddings()

    mock_client_instance.embed_text.assert_called_once_with(
        EmbedTextDetails(
            inputs=images,
            input_type="IMAGE",
            compartment_id="compartmentId",
            serving_mode=OnDemandServingMode(model_id=model),
            truncate="NONE",
        )
    )
    metrics_collector_mock.assert_called_once()
    args, _ = metrics_collector_mock.call_args
    user_response = args[0]
    assert user_response.status_code == 200
    assert user_response.num_prefill_tokens == 10


@patch("genai_bench.user.oci_cohere_user.GenerativeAiInferenceClient")
def test_image_embeddings_multiple_images(mock_client_class, test_cohere_user):
    mock_client_instance = mock_client_class.return_value
    mock_client_instance.embed_text.return_value.status = 200

    test_cohere_user.on_start()
    images = ["BASE64Image1", "BASE64Image2"]
    model = "cohere-embed-v3"
    test_cohere_user.sample = lambda: UserImageEmbeddingRequest(
        documents=[],
        image_content=images,
        num_images=2,
        num_prefill_tokens=10,
        model=model,
        additional_request_params={
            "servingType": "ON_DEMAND",
            "compartmentId": "compartmentId",
        },
    )

    metrics_collector_mock = MagicMock()
    test_cohere_user.collect_metrics = metrics_collector_mock
    with pytest.raises(ValueError):
        test_cohere_user.embeddings()
        mock_client_instance.embed_text.assert_not_called()
        metrics_collector_mock.embed_text.assert_not_called()


@patch("genai_bench.user.oci_cohere_user.logger")
@patch("genai_bench.user.oci_cohere_user.GenerativeAiInferenceClient")
def test_embed_with_response_error(mock_client_class, mock_logger, test_cohere_user):
    mock_client_instance = mock_client_class.return_value
    mock_client_instance.embed_text.return_value.status = 401

    test_cohere_user.on_start()
    documents = ["The earth has a radius of 6,454km."]
    model = "cohere-embed-v3"
    test_cohere_user.sample = lambda: UserEmbeddingRequest(
        documents=documents,
        num_prefill_tokens=10,
        model=model,
        additional_request_params={
            "servingType": "ON_DEMAND",
            "compartmentId": "compartmentId",
        },
    )

    metrics_collector_mock = MagicMock()
    test_cohere_user.collect_metrics = metrics_collector_mock

    test_cohere_user.embeddings()

    mock_client_instance.embed_text.assert_called_once_with(
        EmbedTextDetails(
            inputs=documents,
            input_type="SEARCH_DOCUMENT",
            compartment_id="compartmentId",
            serving_mode=OnDemandServingMode(model_id=model),
            truncate="NONE",
        )
    )
    metrics_collector_mock.assert_called_once()
    args, _ = metrics_collector_mock.call_args
    user_response = args[0]
    assert user_response.status_code == 401


@patch("genai_bench.user.oci_cohere_user.GenerativeAiInferenceClient")
def test_chat_with_chat_history(mock_client_class, test_cohere_user):
    """Test chat with chat history in additional params."""
    mock_client_instance = mock_client_class.return_value
    mock_client_instance.chat.return_value.status = 200
    mock_client_instance.chat.return_value.data.events.return_value = []

    test_cohere_user.on_start()
    chat_history = [{"role": "user", "content": "Previous message"}]
    test_cohere_user.sample = lambda: UserChatRequest(
        model="cohere-model",
        prompt="Hello",
        num_prefill_tokens=5,
        additional_request_params={
            "compartmentId": "ocid1.compartment.oc1..example",
            "servingType": "ON_DEMAND",
            "chatHistory": chat_history,
        },
        max_tokens=10,
    )

    test_cohere_user.chat()

    chat_request = mock_client_instance.chat.call_args[0][0].chat_request
    assert chat_request.chat_history == chat_history


@patch("genai_bench.user.oci_cohere_user.GenerativeAiInferenceClient")
def test_chat_with_documents(mock_client_class, test_cohere_user):
    """Test chat with documents in additional params."""
    mock_client_instance = mock_client_class.return_value
    mock_client_instance.chat.return_value.status = 200
    mock_client_instance.chat.return_value.data.events.return_value = []

    test_cohere_user.on_start()
    documents = [{"text": "Document content"}]
    test_cohere_user.sample = lambda: UserChatRequest(
        model="cohere-model",
        prompt="Hello",
        num_prefill_tokens=5,
        additional_request_params={
            "compartmentId": "ocid1.compartment.oc1..example",
            "servingType": "ON_DEMAND",
            "documents": documents,
        },
        max_tokens=10,
    )

    test_cohere_user.chat()

    chat_request = mock_client_instance.chat.call_args[0][0].chat_request
    assert chat_request.documents == documents


def test_get_compartment_id_missing(test_cohere_user):
    """Test error when compartmentId is missing."""
    request = UserChatRequest(
        model="model",
        prompt="test",
        additional_request_params={},
        num_prefill_tokens=5,
        max_tokens=10,
    )

    with pytest.raises(
        ValueError, match="compartmentId missing in additional request params"
    ):
        test_cohere_user.get_compartment_id(request)


@patch("genai_bench.user.oci_cohere_user.GenerativeAiInferenceClient")
def test_embeddings_with_missing_prefill_tokens(mock_client_class, test_cohere_user):
    """Test embeddings with missing prefill tokens."""
    mock_client_instance = mock_client_class.return_value
    mock_client_instance.embed_text.return_value.status = 200
    model = "cohere-embed-v3"

    test_cohere_user.on_start()
    test_cohere_user.sample = lambda: UserEmbeddingRequest(
        documents=[],
        num_prefill_tokens=0,  # missing prefill tokens
        model=model,
        additional_request_params={
            "servingType": "ON_DEMAND",
            "compartmentId": "compartmentId",
        },
    )
    metrics_collector_mock = MagicMock()
    test_cohere_user.collect_metrics = metrics_collector_mock

    test_cohere_user.embeddings()

    mock_client_instance.embed_text.assert_called_once_with(
        EmbedTextDetails(
            inputs=[],
            input_type="SEARCH_DOCUMENT",
            compartment_id="compartmentId",
            serving_mode=OnDemandServingMode(model_id=model),
            truncate="NONE",
        )
    )
    metrics_collector_mock.assert_called_once()
    args, _ = metrics_collector_mock.call_args
    user_response = args[0]
    assert user_response.status_code == 200
    assert user_response.num_prefill_tokens == 0


def test_get_serving_mode_missing_endpoint(test_cohere_user):
    """Test error when endpointId is missing for DEDICATED serving type."""
    request = UserChatRequest(
        model="model",
        prompt="test",
        additional_request_params={
            "servingType": "DEDICATED",  # Missing endpointId
            "compartmentId": "test",
        },
        num_prefill_tokens=5,
        max_tokens=10,
    )

    with pytest.raises(
        ValueError, match="endpointId must be provided for DEDICATED servingType"
    ):
        test_cohere_user.get_serving_mode(request)


def test_get_serving_mode_dedicated(test_cohere_user):
    """Test error when endpointId is missing for DEDICATED serving type."""
    request = UserChatRequest(
        model="model",
        prompt="test",
        additional_request_params={
            "servingType": "DEDICATED",
            "endpointId": "endpoint",
            "compartmentId": "test",
        },
        num_prefill_tokens=5,
        max_tokens=10,
    )

    serving_mode = test_cohere_user.get_serving_mode(request)

    assert isinstance(serving_mode, DedicatedServingMode)
    assert serving_mode.endpoint_id == "endpoint"


@patch("genai_bench.user.oci_cohere_user.GenerativeAiInferenceClient")
def test_rerank(mock_client_class, test_cohere_user):
    mock_client_instance = mock_client_class.return_value
    mock_client_instance.rerank_text.return_value.status = 200

    test_cohere_user.on_start()
    documents = [
        "The earth has a radius of 6,454km.",
        "Earth is the 3rd planet in solar system.",
    ]
    query = "What is the order of earth in the solar system?"
    model = "cohere-reank-v3.5"
    test_cohere_user.sample = lambda: UserReRankRequest(
        documents=documents,
        query=query,
        num_prefill_tokens=100,
        model=model,
        additional_request_params={
            "servingType": "ON_DEMAND",
            "compartmentId": "compartmentId",
        },
    )

    metrics_collector_mock = MagicMock()
    test_cohere_user.collect_metrics = metrics_collector_mock

    test_cohere_user.rerank()

    mock_client_instance.rerank_text.assert_called_once_with(
        RerankTextDetails(
            documents=documents,
            input=query,
            compartment_id="compartmentId",
            serving_mode=OnDemandServingMode(model_id=model),
        )
    )
    metrics_collector_mock.assert_called_once()
    args, _ = metrics_collector_mock.call_args
    user_response = args[0]
    assert user_response.status_code == 200
    assert user_response.num_prefill_tokens == 100


def test_rerank_with_embedding_request(test_cohere_user):
    documents = [
        "The earth has a radius of 6,454km.",
        "Earth is the 3rd planet in solar system.",
    ]
    model = "cohere-reank-v3.5"
    test_cohere_user.sample = lambda: UserEmbeddingRequest(
        documents=documents,
        num_prefill_tokens=100,
        model=model,
        additional_request_params={
            "servingType": "ON_DEMAND",
            "compartmentId": "compartmentId",
        },
    )

    with pytest.raises(
        AttributeError, match="user_request should be of type UserReRankRequest"
    ):
        test_cohere_user.rerank()


@patch("genai_bench.user.oci_cohere_user.GenerativeAiInferenceClient")
def test_rerank_with_no_prefill(mock_client_class, test_cohere_user):
    mock_client_instance = mock_client_class.return_value
    mock_client_instance.rerank_text.return_value.status = 200

    test_cohere_user.on_start()
    documents = [
        "The earth has a radius of 6,454km.",
        "Earth is the 3rd planet in solar system.",
    ]
    query = "What is the order of earth in the solar system?"
    model = "cohere-reank-v3.5"
    test_cohere_user.sample = lambda: UserReRankRequest(
        documents=documents,
        query=query,
        num_prefill_tokens=None,
        model=model,
        additional_request_params={
            "servingType": "ON_DEMAND",
            "compartmentId": "compartmentId",
        },
    )

    metrics_collector_mock = MagicMock()
    test_cohere_user.collect_metrics = metrics_collector_mock

    test_cohere_user.rerank()

    mock_client_instance.rerank_text.assert_called_once_with(
        RerankTextDetails(
            documents=documents,
            input=query,
            compartment_id="compartmentId",
            serving_mode=OnDemandServingMode(model_id=model),
        )
    )
    metrics_collector_mock.assert_called_once()
    args, _ = metrics_collector_mock.call_args
    user_response = args[0]
    assert user_response.status_code == 200
    assert user_response.num_prefill_tokens is None
