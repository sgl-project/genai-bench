import json
from unittest.mock import MagicMock, patch

import pytest
from oci.generative_ai_inference.models import (
    CohereChatRequestV2,
    CohereImageContentV2,
    CohereTextContentV2,
)

from genai_bench.protocol import UserChatRequest, UserImageChatRequest
from genai_bench.user.oci_cohere_v2_user import OCICohereV2User


@pytest.fixture
def test_cohere_v2_user():
    OCICohereV2User.host = "http://example.com"

    mock_auth = MagicMock()
    mock_auth.get_credentials.return_value = "test-key"
    mock_auth.get_config.return_value = {
        "api_base": "http://example.com",
        "compartment_id": "test-compartment",
        "endpoint_id": "test-endpoint",
    }
    OCICohereV2User.auth_provider = mock_auth

    user = OCICohereV2User(environment=MagicMock())
    return user


def test_supported_tasks_include_vision_chat():
    assert OCICohereV2User.supported_tasks["image-text-to-text"] == "chat"
    assert "text-to-embeddings" not in OCICohereV2User.supported_tasks


@patch("genai_bench.user.oci_cohere_user.GenerativeAiInferenceClient")
def test_chat_v2_text(mock_client_class, test_cohere_v2_user):
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
                            "promptTokens": 5,
                            "completionTokens": 3,
                            "completionTokensDetails": {
                                "reasoningTokens": 2,
                            },
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

    test_cohere_v2_user.on_start()
    history = [{"role": "assistant", "content": "Previous turn"}]
    documents = [{"text": "Doc"}]
    test_cohere_v2_user.sample = lambda: UserChatRequest(
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
    test_cohere_v2_user.collect_metrics = metrics_collector_mock

    test_cohere_v2_user.chat()

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
def test_chat_v2_text_snake_case_usage(mock_client_class, test_cohere_v2_user):
    mock_client_instance = mock_client_class.return_value
    mock_client_instance.chat.return_value.status = 200
    mock_client_instance.chat.return_value.data.events.return_value = iter(
        [
            MagicMock(
                data=json.dumps(
                    {
                        "message": {
                            "role": "ASSISTANT",
                            "content": [{"type": "TEXT", "text": "Hi"}],
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
                                "input_tokens": 4,
                                "output_tokens": 2,
                                "reasoning_tokens": 1,
                            }
                        },
                        "message": {
                            "role": "ASSISTANT",
                            "content": [{"type": "TEXT", "text": " there"}],
                        },
                    }
                ).encode("utf-8")
            ),
        ]
    )

    test_cohere_v2_user.on_start()
    test_cohere_v2_user.sample = lambda: UserChatRequest(
        model="cohere-command-a",
        prompt="Hi?",
        num_prefill_tokens=None,
        additional_request_params={
            "compartmentId": "ocid1.compartment.oc1..example",
            "servingType": "ON_DEMAND",
        },
        max_tokens=32,
    )

    metrics_collector_mock = MagicMock()
    test_cohere_v2_user.collect_metrics = metrics_collector_mock

    test_cohere_v2_user.chat()

    metrics_collector_mock.assert_called_once()
    user_response = metrics_collector_mock.call_args[0][0]
    assert user_response.tokens_received == 2
    assert user_response.reasoning_tokens == 1
    assert user_response.num_prefill_tokens == 4


@patch("genai_bench.user.oci_cohere_user.GenerativeAiInferenceClient")
def test_chat_v2_vision(mock_client_class, test_cohere_v2_user):
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

    test_cohere_v2_user.on_start()
    image_data = "data:image/png;base64,AAA"
    test_cohere_v2_user.sample = lambda: UserImageChatRequest(
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
    test_cohere_v2_user.collect_metrics = metrics_collector_mock

    test_cohere_v2_user.chat()

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
def test_chat_v2_vision_multiple_images_error(mock_client_class, test_cohere_v2_user):
    mock_client_class.return_value.chat.return_value.status = 200

    test_cohere_v2_user.on_start()
    test_cohere_v2_user.sample = lambda: UserImageChatRequest(
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
        test_cohere_v2_user.chat()


@patch("genai_bench.user.oci_cohere_user.GenerativeAiInferenceClient")
def test_chat_v2_includes_thinking_content(mock_client_class, test_cohere_v2_user):
    mock_client_instance = mock_client_class.return_value
    mock_client_instance.chat.return_value.status = 200
    mock_client_instance.chat.return_value.data.events.return_value = iter(
        [
            MagicMock(
                data=json.dumps(
                    {
                        "message": {
                            "role": "ASSISTANT",
                            "content": [
                                {"type": "TEXT", "text": "Analysis: "},
                                {"type": "THINKING", "thinking": "Step 1 -> Step 2"},
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
                                "input_tokens": 6,
                                "output_tokens": 4,
                                "reasoning_tokens": 8,
                            }
                        },
                        "message": {
                            "role": "ASSISTANT",
                            "content": [{"type": "TEXT", "text": "Final answer."}],
                        },
                    }
                ).encode("utf-8")
            ),
        ]
    )

    test_cohere_v2_user.on_start()
    test_cohere_v2_user.sample = lambda: UserChatRequest(
        model="cohere-command-a",
        prompt="Explain the process.",
        num_prefill_tokens=None,
        additional_request_params={
            "compartmentId": "ocid1.compartment.oc1..example",
            "servingType": "ON_DEMAND",
        },
        max_tokens=128,
    )

    metrics_collector_mock = MagicMock()
    test_cohere_v2_user.collect_metrics = metrics_collector_mock

    test_cohere_v2_user.chat()

    metrics_collector_mock.assert_called_once()
    user_response = metrics_collector_mock.call_args[0][0]
    assert user_response.tokens_received == 4
    assert user_response.reasoning_tokens == 8
    assert user_response.num_prefill_tokens == 6
    # Thinking text should be appended to generated text payload.
    assert "Step 1 -> Step 2" in user_response.generated_text
