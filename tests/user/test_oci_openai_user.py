from unittest.mock import MagicMock, patch

import pytest

from genai_bench.protocol import (
    UserChatRequest,
    UserImageGenerationRequest,
    UserImageGenerationResponse,
)
from genai_bench.user.oci_openai_user import OCI_AUTH_CLASS_MAP, OCIOpenAIUser


@pytest.fixture
def mock_oci_openai_user():
    mock_auth = MagicMock()
    mock_auth.get_auth_type.return_value = "oci_security_token"
    OCIOpenAIUser.auth_provider = mock_auth
    OCIOpenAIUser.host = (
        "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
    )
    OCIOpenAIUser.oci_compartment_id = "ocid1.compartment.oc1..test"
    OCIOpenAIUser.oci_profile = "DEFAULT"
    OCIOpenAIUser.oci_config_file = None

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


@patch("genai_bench.user.oci_openai_user.OpenAI")
@patch.dict(OCI_AUTH_CLASS_MAP, {"oci_user_principal": MagicMock()})
def test_on_start_user_principal_auth(mock_openai, mock_oci_openai_user):
    mock_oci_openai_user.auth_provider.get_auth_type.return_value = "oci_user_principal"
    mock_oci_openai_user.oci_profile = "MY_PROFILE"
    mock_oci_openai_user.oci_config_file = "/home/user/.oci/config"

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
def test_image_generation(mock_openai, mock_oci_openai_user):
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
        additional_request_params={},
    )

    result = mock_oci_openai_user.image_generation()

    assert isinstance(result, UserImageGenerationResponse)
    assert result.status_code == 200
    assert result.generated_images == ["https://example.com/generated.png"]
    assert result.revised_prompt == "A revised test prompt"

    mock_client.images.generate.assert_called_once_with(
        model="cohere.flux-1.1-pro",
        prompt="A test image",
        n=1,
        size="1024x1024",
    )


@patch("genai_bench.user.oci_openai_user.OpenAI")
@patch.dict(OCI_AUTH_CLASS_MAP, {"oci_security_token": MagicMock()})
def test_image_generation_error(mock_openai, mock_oci_openai_user):
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
        additional_request_params={},
    )

    result = mock_oci_openai_user.image_generation()

    assert isinstance(result, UserImageGenerationResponse)
    assert result.status_code == 503
    assert "Service unavailable" in result.error_message
    assert result.generated_images == []
