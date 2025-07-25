"""Unit tests for OCI model authentication adapter."""

from unittest.mock import MagicMock, Mock

import pytest

from genai_bench.auth.model_auth_provider import ModelAuthProvider
from genai_bench.auth.oci.model_auth_adapter import OCIModelAuthAdapter


class TestOCIModelAuthAdapter:
    """Test cases for OCI model authentication adapter."""

    def test_inheritance(self):
        """Test that OCIModelAuthAdapter inherits from ModelAuthProvider."""
        mock_oci_auth = MagicMock()
        adapter = OCIModelAuthAdapter(mock_oci_auth)
        assert isinstance(adapter, ModelAuthProvider)

    def test_init(self):
        """Test initialization with OCI auth provider."""
        mock_oci_auth = MagicMock()
        adapter = OCIModelAuthAdapter(mock_oci_auth)
        assert adapter.oci_auth is mock_oci_auth

    def test_get_headers(self):
        """Test get_headers returns empty dict for OCI."""
        mock_oci_auth = MagicMock()
        adapter = OCIModelAuthAdapter(mock_oci_auth)

        headers = adapter.get_headers()
        assert headers == {}
        assert isinstance(headers, dict)

    def test_get_config(self):
        """Test get_config returns proper configuration."""
        mock_oci_auth = MagicMock()
        # Mock the underlying auth's get_config method
        mock_oci_auth.get_config.return_value = {"region": "us-ashburn-1"}
        adapter = OCIModelAuthAdapter(mock_oci_auth)

        # Mock the get_auth_type to return a known value
        adapter.get_auth_type = Mock(return_value="oci_user_principal")

        config = adapter.get_config()

        assert isinstance(config, dict)
        assert config["auth_type"] == "oci_user_principal"
        assert config["region"] == "us-ashburn-1"
        assert "oci_auth" not in config  # Should not include non-serializable object

    def test_get_credentials(self):
        """Test get_credentials returns the OCI auth provider."""
        mock_oci_auth = MagicMock()
        adapter = OCIModelAuthAdapter(mock_oci_auth)

        credentials = adapter.get_credentials()
        assert credentials is mock_oci_auth.get_credentials()

    @pytest.mark.parametrize(
        "class_name,expected_auth_type",
        [
            ("InstancePrincipalAuth", "oci_instance_principal"),
            ("UserPrincipalAuth", "oci_user_principal"),
            ("OBOTokenAuth", "oci_obo_token"),
            ("SessionTokenAuth", "oci_security_token"),
            ("UnknownAuth", "oci_unknown"),
            ("SomeOtherAuth", "oci_unknown"),
        ],
    )
    def test_get_auth_type(self, class_name, expected_auth_type):
        """Test get_auth_type returns correct auth type based on class name."""
        mock_oci_auth = MagicMock()
        mock_oci_auth.__class__.__name__ = class_name

        adapter = OCIModelAuthAdapter(mock_oci_auth)
        auth_type = adapter.get_auth_type()

        assert auth_type == expected_auth_type

    def test_get_auth_type_with_namespace_class(self):
        """Test get_auth_type with classes that have namespace prefixes."""
        mock_oci_auth = MagicMock()
        mock_oci_auth.__class__.__name__ = "OCIInstancePrincipalAuth"

        adapter = OCIModelAuthAdapter(mock_oci_auth)
        auth_type = adapter.get_auth_type()

        assert auth_type == "oci_instance_principal"

    def test_integration_with_factory(self):
        """Test that adapter works with unified auth factory pattern."""
        from genai_bench.auth.auth_provider import AuthProvider

        # Create a mock OCI auth that implements AuthProvider
        class MockOCIAuth(AuthProvider):
            def get_config(self):
                return {"region": "us-ashburn-1"}

            def get_credentials(self):
                return "mock_signer"

        mock_oci_auth = MockOCIAuth()
        adapter = OCIModelAuthAdapter(mock_oci_auth)

        # Test that all required methods work
        assert adapter.get_headers() == {}
        config = adapter.get_config()
        assert "auth_type" in config
        assert "region" in config
        assert "oci_auth" not in config  # Should not include non-serializable object
        assert adapter.get_credentials() is mock_oci_auth.get_credentials()
        assert adapter.get_auth_type().startswith("oci_")

    def test_error_handling(self):
        """Test error handling when OCI auth provider is None."""
        adapter = OCIModelAuthAdapter(None)

        # Should not raise errors, but handle gracefully
        assert adapter.get_headers() == {}

        config = adapter.get_config()
        assert "auth_type" in config
        assert "oci_auth" not in config  # Should not include non-serializable object

        # get_auth_type should handle None gracefully
        auth_type = adapter.get_auth_type()
        assert auth_type == "oci_unknown"

    def test_multiple_instances(self):
        """Test that multiple adapter instances don't interfere."""
        mock_oci_auth1 = MagicMock()
        mock_oci_auth1.__class__.__name__ = "UserPrincipalAuth"

        mock_oci_auth2 = MagicMock()
        mock_oci_auth2.__class__.__name__ = "InstancePrincipalAuth"

        adapter1 = OCIModelAuthAdapter(mock_oci_auth1)
        adapter2 = OCIModelAuthAdapter(mock_oci_auth2)

        assert adapter1.get_auth_type() == "oci_user_principal"
        assert adapter2.get_auth_type() == "oci_instance_principal"
        assert adapter1.get_credentials() is mock_oci_auth1.get_credentials()
        assert adapter2.get_credentials() is mock_oci_auth2.get_credentials()
