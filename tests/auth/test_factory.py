"""Tests for auth factory."""

from unittest.mock import MagicMock, patch

import pytest

from genai_bench.auth.factory import AuthFactory


class TestAuthFactory:
    """Test auth factory."""

    def test_create_openai_auth(self):
        """Test creating OpenAI auth provider."""
        auth = AuthFactory.create_openai_auth("test-key")
        assert auth.api_key == "test-key"

    @patch("genai_bench.auth.factory.OCIInstancePrincipalAuth")
    def test_create_oci_instance_principal(self, mock_auth):
        """Test creating OCI instance principal auth."""
        mock_instance = MagicMock()
        mock_auth.return_value = mock_instance

        result = AuthFactory.create_oci_auth("instance_principal")

        mock_auth.assert_called_once_with(security_token=None, region=None)
        assert result == mock_instance

    @patch("genai_bench.auth.factory.OCIOBOTokenAuth")
    def test_create_oci_obo_token(self, mock_auth):
        """Test creating OCI OBO token auth."""
        mock_instance = MagicMock()
        mock_auth.return_value = mock_instance

        result = AuthFactory.create_oci_auth(
            "instance_obo_user", token="test-token", region="us-ashburn-1"
        )

        mock_auth.assert_called_once_with(token="test-token", region="us-ashburn-1")
        assert result == mock_instance

    def test_create_oci_obo_token_missing_params(self):
        """Test creating OCI OBO token auth with missing params."""
        with pytest.raises(ValueError, match="token and region are required"):
            AuthFactory.create_oci_auth("instance_obo_user")

    @patch("genai_bench.auth.factory.OCIUserPrincipalAuth")
    def test_create_oci_user_principal(self, mock_auth):
        """Test creating OCI user principal auth."""
        mock_instance = MagicMock()
        mock_auth.return_value = mock_instance

        result = AuthFactory.create_oci_auth(
            "user_principal", config_path="/path/to/config", profile="TEST"
        )

        mock_auth.assert_called_once_with(config_path="/path/to/config", profile="TEST")
        assert result == mock_instance

    @patch("genai_bench.auth.factory.OCISessionAuth")
    def test_create_oci_security_token(self, mock_auth):
        """Test creating OCI security token auth."""
        mock_instance = MagicMock()
        mock_auth.return_value = mock_instance

        result = AuthFactory.create_oci_auth(
            "security_token", config_path="/path/to/config", profile="TEST"
        )

        mock_auth.assert_called_once_with(config_path="/path/to/config", profile="TEST")
        assert result == mock_instance

    def test_create_oci_invalid_auth_type(self):
        """Test creating OCI auth with invalid type."""
        with pytest.raises(ValueError) as exc_info:
            AuthFactory.create_oci_auth("invalid_type")

        assert "Invalid auth_type: invalid_type" in str(exc_info.value)
        assert "user_principal" in str(exc_info.value)
        assert "instance_principal" in str(exc_info.value)

    @patch("genai_bench.auth.factory.OCIUserPrincipalAuth")
    def test_create_oci_default_values(self, mock_auth):
        """Test creating OCI auth with default values."""
        mock_instance = MagicMock()
        mock_auth.return_value = mock_instance

        result = AuthFactory.create_oci_auth("user_principal")

        # The factory uses hardcoded defaults
        mock_auth.assert_called_once_with(
            config_path="~/.oci/config", profile="DEFAULT"
        )
        assert result == mock_instance
