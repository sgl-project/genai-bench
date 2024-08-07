import pytest

from genai_bench.auth.factory import AuthFactory
from genai_bench.auth.oci.instance_principal import OCIInstancePrincipalAuth
from genai_bench.auth.oci.obo_token import OCIOBOTokenAuth
from genai_bench.auth.oci.session import OCISessionAuth
from genai_bench.auth.oci.user_principal import OCIUserPrincipalAuth
from genai_bench.auth.openai.auth import OpenAIAuth

MOCK_API_KEY = "genai-bench-test-123456789"
MOCK_CONFIG_PATH = "~/.oci/config"
MOCK_PROFILE = "TestProfile"
MOCK_TOKEN = "test.security.token"
MOCK_REGION = "us-ashburn-1"


class TestAuthFactory:
    def test_create_openai_auth(self):
        """Test creating OpenAI auth provider."""
        auth = AuthFactory.create_openai_auth(MOCK_API_KEY)
        assert isinstance(auth, OpenAIAuth)
        assert auth.api_key == MOCK_API_KEY

    def test_create_oci_user_principal_auth(self):
        """Test creating OCI user principal auth."""
        auth = AuthFactory.create_oci_auth(
            auth_type="user_principal",
            config_path=MOCK_CONFIG_PATH,
            profile=MOCK_PROFILE,
        )
        assert isinstance(auth, OCIUserPrincipalAuth)
        assert auth.config_path == MOCK_CONFIG_PATH
        assert auth.profile == MOCK_PROFILE

    def test_create_oci_instance_principal_auth(self):
        """Test creating OCI instance principal auth."""
        auth = AuthFactory.create_oci_auth(auth_type="instance_principal")
        assert isinstance(auth, OCIInstancePrincipalAuth)

    def test_create_oci_session_auth(self):
        """Test creating OCI session auth."""
        auth = AuthFactory.create_oci_auth(
            auth_type="security_token",
            config_path=MOCK_CONFIG_PATH,
            profile=MOCK_PROFILE,
        )
        assert isinstance(auth, OCISessionAuth)
        assert auth.config_path == MOCK_CONFIG_PATH
        assert auth.profile == MOCK_PROFILE

    def test_create_oci_obo_token_auth(self):
        """Test creating OCI OBO token auth."""
        auth = AuthFactory.create_oci_auth(
            auth_type="instance_obo_user", token=MOCK_TOKEN, region=MOCK_REGION
        )
        assert isinstance(auth, OCIOBOTokenAuth)
        assert auth.token == MOCK_TOKEN
        assert auth.region == MOCK_REGION

    def test_create_oci_auth_missing_token(self):
        """Test error when OBO token is missing."""
        with pytest.raises(ValueError) as exc:
            AuthFactory.create_oci_auth(
                auth_type="instance_obo_user", region=MOCK_REGION
            )
        assert "token and region are required" in str(exc.value)

    def test_create_oci_auth_missing_region(self):
        """Test error when OBO region is missing."""
        with pytest.raises(ValueError) as exc:
            AuthFactory.create_oci_auth(auth_type="instance_obo_user", token=MOCK_TOKEN)
        assert "token and region are required" in str(exc.value)

    def test_create_oci_auth_invalid_type(self):
        """Test error with invalid auth type."""
        with pytest.raises(ValueError) as exc:
            AuthFactory.create_oci_auth(auth_type="invalid")
        assert "Invalid auth_type" in str(exc.value)
