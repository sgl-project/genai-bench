import json
from unittest.mock import MagicMock, patch

import pytest

from genai_bench.protocol import (
    UserChatResponse,
    UserChatRequest,
    UserImageGenerationRequest,
    UserImageGenerationResponse,
)
from genai_bench.user.oci_openai_user import (
    COALESCED_STREAM_STATUS_CODE,
    OCI_AUTH_CLASS_MAP,
    OCIOpenAIUser,
)


@pytest.fixture
def mock_oci_openai_user():
    mock_oci_auth = MagicMock()
    mock_oci_auth.profile = "DEFAULT"
    mock_oci_auth.config_path = None

    mock_auth = MagicMock()
    mock_auth.get_auth_type.return_value = "oci_security_token"
    mock_auth.oci_auth = mock_oci_auth
    OCIOpenAIUser.auth_provider = mock_auth
    OCIOpenAIUser.host = (
        "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
    )

    user = OCIOpenAIUser(environment=MagicMock())
    user.user_requests = [
        UserChatRequest(
            model="meta.llama-3.1-70b-instruct",
            prompt="Hello",
            num_prefill_tokens=5,
            additional_request_params={},
            max_tokens=10,
        )
    ] * 5
    return user


@patch("genai_bench.user.oci_openai_user.OpenAI")
@patch.dict(OCI_AUTH_CLASS_MAP, {"oci_security_token": MagicMock()})
def test_on_start_session_auth(mock_openai, mock_oci_openai_user):
    mock_oci_openai_user.on_start()

    mock_auth_cls = OCI_AUTH_CLASS_MAP["oci_security_token"]
    mock_auth_cls.assert_called_once_with(profile_name="DEFAULT")
    mock_openai.assert_called_once()


def test_text_to_text_task_is_supported():
    assert OCIOpenAIUser.supported_tasks["text-to-text"] == "chat"


class FakeStreamingResponse:
    def __init__(self, lines):
        self.lines = lines
        self.chunk_sizes = []
        self._genai_bench_token_event_count = 0

    def iter_lines(self, chunk_size=None, decode_unicode=False, delimiter=None):
        self.chunk_sizes.append(chunk_size)
        return iter(self.lines)


def _chat_response(*, first_token_time=1.0, end_time=2.0, tokens_received=4):
    return UserChatResponse(
        status_code=200,
        generated_text="test",
        tokens_received=tokens_received,
        time_at_first_token=first_token_time,
        num_prefill_tokens=5,
        start_time=0.0,
        end_time=end_time,
    )


def test_low_latency_iter_lines_counts_token_events(monkeypatch):
    monkeypatch.setenv("OCI_OPENAI_STREAM_CHUNK_SIZE", "1")
    response = FakeStreamingResponse(
        [
            b'data: {"choices":[{"delta":{"content":"one"}}]}',
            b'data: {"choices":[{"delta":{"reasoning_content":"two"}}]}',
            b'data: {"choices":[{"delta":{},"finish_reason":"length"}]}',
            b"data: [DONE]",
        ]
    )

    OCIOpenAIUser._use_low_latency_iter_lines(response)

    assert list(response.iter_lines(chunk_size=None)) == response.lines
    assert response.chunk_sizes == [1]
    assert response._genai_bench_token_event_count == 2


def test_coalesced_stream_timing_is_rejected(monkeypatch):
    monkeypatch.setenv("OCI_OPENAI_MAX_PLAUSIBLE_OUTPUT_TPS", "2000")
    response = FakeStreamingResponse([])
    response._genai_bench_token_event_count = 1

    result = OCIOpenAIUser._reject_unreliable_stream_timing(
        _chat_response(first_token_time=1.0, end_time=2.0),
        response,
    )

    assert result.status_code == COALESCED_STREAM_STATUS_CODE
    assert "only 1 token-bearing SSE event" in result.error_message


def test_multi_event_plausible_stream_timing_is_preserved(monkeypatch):
    monkeypatch.setenv("OCI_OPENAI_MAX_PLAUSIBLE_OUTPUT_TPS", "2000")
    response = FakeStreamingResponse([])
    response._genai_bench_token_event_count = 2
    metrics_response = _chat_response(first_token_time=1.0, end_time=2.0)

    result = OCIOpenAIUser._reject_unreliable_stream_timing(
        metrics_response,
        response,
    )

    assert result is metrics_response


def test_multi_event_impossible_stream_timing_is_rejected(monkeypatch):
    monkeypatch.setenv("OCI_OPENAI_MAX_PLAUSIBLE_OUTPUT_TPS", "2000")
    response = FakeStreamingResponse([])
    response._genai_bench_token_event_count = 2

    result = OCIOpenAIUser._reject_unreliable_stream_timing(
        _chat_response(
            first_token_time=1.0,
            end_time=1.001,
            tokens_received=512,
        ),
        response,
    )

    assert result.status_code == COALESCED_STREAM_STATUS_CODE
    assert "tokens/s exceeds 2000.0" in result.error_message


@patch("genai_bench.user.oci_openai_user.requests.Session")
def test_send_request_signs_low_latency_stream(mock_session, mock_oci_openai_user):
    response = FakeStreamingResponse(
        [
            b'data: {"choices":[{"delta":{"content":"one"}}]}',
            b'data: {"choices":[{"delta":{"content":"two"}}]}',
            b"data: [DONE]",
        ]
    )
    response.status_code = 200
    response.close = MagicMock()

    session = MagicMock()
    session.__enter__.return_value = session
    session.send.return_value = response
    mock_session.return_value = session

    mock_oci_openai_user._oci_auth = MagicMock()
    mock_oci_openai_user.collect_metrics = MagicMock()

    def parse_strategy(stream_response, start_time, num_prefill_tokens, _):
        assert list(stream_response.iter_lines(chunk_size=None)) == response.lines
        return UserChatResponse(
            status_code=200,
            generated_text="onetwo",
            tokens_received=2,
            time_at_first_token=start_time + 0.5,
            num_prefill_tokens=num_prefill_tokens,
            start_time=start_time,
            end_time=start_time + 1.0,
        )

    result = mock_oci_openai_user.send_request(
        stream=True,
        endpoint="/v1/chat/completions",
        payload={
            "model": "test-model",
            "stream": True,
            "ignore_eos": True,
            "compartmentId": "ocid1.compartment.oc1..test",
        },
        parse_strategy=parse_strategy,
        num_prefill_tokens=5,
    )

    prepared_request = session.send.call_args.args[0]
    request_body = json.loads(prepared_request.body)
    assert request_body["ignore_eos"] is True
    assert "compartmentId" not in request_body
    assert prepared_request.headers["Accept-Encoding"] == "identity"
    assert prepared_request.headers["CompartmentId"] == "ocid1.compartment.oc1..test"
    assert prepared_request.headers["opc-compartment-id"] == (
        "ocid1.compartment.oc1..test"
    )
    assert session.trust_env is False
    assert response.chunk_sizes == [1]
    assert response._genai_bench_token_event_count == 2
    mock_oci_openai_user._oci_auth.signer.do_request_sign.assert_called_once_with(
        prepared_request
    )
    mock_oci_openai_user.collect_metrics.assert_called_once_with(
        result,
        "/v1/chat/completions",
    )


@patch("genai_bench.user.oci_openai_user.OpenAI")
@patch.dict(OCI_AUTH_CLASS_MAP, {"oci_user_principal": MagicMock()})
def test_on_start_user_principal_auth(mock_openai, mock_oci_openai_user):
    mock_oci_openai_user.auth_provider.get_auth_type.return_value = "oci_user_principal"
    mock_oci_openai_user.auth_provider.oci_auth.profile = "MY_PROFILE"
    mock_oci_openai_user.auth_provider.oci_auth.config_path = "/home/user/.oci/config"

    mock_oci_openai_user.on_start()

    mock_auth_cls = OCI_AUTH_CLASS_MAP["oci_user_principal"]
    mock_auth_cls.assert_called_once_with(
        profile_name="MY_PROFILE", config_file="/home/user/.oci/config"
    )


@patch("genai_bench.user.oci_openai_user.OpenAI")
@patch.dict(OCI_AUTH_CLASS_MAP, {"oci_instance_principal": MagicMock()})
def test_on_start_instance_principal_auth(mock_openai, mock_oci_openai_user):
    mock_oci_openai_user.auth_provider.get_auth_type.return_value = (
        "oci_instance_principal"
    )

    mock_oci_openai_user.on_start()

    mock_auth_cls = OCI_AUTH_CLASS_MAP["oci_instance_principal"]
    mock_auth_cls.assert_called_once_with()


@patch("genai_bench.user.oci_openai_user.OpenAI")
@patch.dict(OCI_AUTH_CLASS_MAP, {"oci_obo_token": MagicMock()})
def test_on_start_resource_principal_auth(mock_openai, mock_oci_openai_user):
    mock_oci_openai_user.auth_provider.get_auth_type.return_value = "oci_obo_token"

    mock_oci_openai_user.on_start()

    mock_auth_cls = OCI_AUTH_CLASS_MAP["oci_obo_token"]
    mock_auth_cls.assert_called_once_with()


def test_on_start_unsupported_auth(mock_oci_openai_user):
    mock_oci_openai_user.auth_provider.get_auth_type.return_value = "unsupported_type"

    with pytest.raises(ValueError, match="Unsupported OCI auth type"):
        mock_oci_openai_user.on_start()


@patch("genai_bench.user.oci_openai_user.OpenAI")
@patch.dict(OCI_AUTH_CLASS_MAP, {"oci_security_token": MagicMock()})
def test_images_generations(mock_openai, mock_oci_openai_user):
    mock_client = MagicMock()
    mock_openai.return_value = mock_client

    mock_image = MagicMock()
    mock_image.url = "https://example.com/generated.png"
    mock_image.b64_json = None
    mock_image.revised_prompt = "A revised test prompt"
    mock_client.images.generate.return_value = MagicMock(data=[mock_image])

    mock_oci_openai_user.on_start()
    mock_oci_openai_user.sample = lambda: UserImageGenerationRequest(
        model="cohere.flux-1.1-pro",
        prompt="A test image",
        size="1024x1024",
        quality="standard",
        num_images=1,
        additional_request_params={"compartmentId": "ocid1.compartment.oc1..test"},
    )

    result = mock_oci_openai_user.images_generations()

    assert isinstance(result, UserImageGenerationResponse)
    assert result.status_code == 200
    assert result.generated_images == ["https://example.com/generated.png"]
    assert result.revised_prompt == "A revised test prompt"

    mock_client.images.generate.assert_called_once_with(
        model="cohere.flux-1.1-pro",
        prompt="A test image",
        n=1,
        size="1024x1024",
        extra_headers={"CompartmentId": "ocid1.compartment.oc1..test"},
    )


@patch("genai_bench.user.oci_openai_user.OpenAI")
@patch.dict(OCI_AUTH_CLASS_MAP, {"oci_security_token": MagicMock()})
def test_images_generations_error(mock_openai, mock_oci_openai_user):
    """Test that errors from OCI server are handled gracefully."""
    mock_client = MagicMock()
    mock_openai.return_value = mock_client

    error = Exception("Service unavailable")
    error.status_code = 503
    mock_client.images.generate.side_effect = error

    mock_oci_openai_user.on_start()
    mock_oci_openai_user.sample = lambda: UserImageGenerationRequest(
        model="cohere.flux-1.1-pro",
        prompt="A test image",
        size="1024x1024",
        quality="standard",
        num_images=1,
        additional_request_params={"compartmentId": "ocid1.compartment.oc1..test"},
    )

    result = mock_oci_openai_user.images_generations()

    assert isinstance(result, UserImageGenerationResponse)
    assert result.status_code == 503
    assert "Service unavailable" in result.error_message
    assert result.generated_images == []
