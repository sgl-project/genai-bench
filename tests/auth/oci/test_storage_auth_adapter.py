"""Unit tests for OCI storage authentication adapter."""

from unittest.mock import MagicMock

from genai_bench.auth.oci.storage_auth_adapter import OCIStorageAuthAdapter
from genai_bench.auth.storage_auth_provider import StorageAuthProvider


class TestOCIStorageAuthAdapter:
    """Test cases for OCI storage authentication adapter."""

    def test_inheritance(self):
        """Test that OCIStorageAuthAdapter inherits from StorageAuthProvider."""
        mock_oci_auth = MagicMock()
        adapter = OCIStorageAuthAdapter(mock_oci_auth)
        assert isinstance(adapter, StorageAuthProvider)

    def test_init(self):
        """Test initialization with OCI auth provider."""
        mock_oci_auth = MagicMock()
        adapter = OCIStorageAuthAdapter(mock_oci_auth)
        assert adapter.oci_auth is mock_oci_auth

    def test_get_client_config(self):
        """Test get_client_config returns proper configuration."""
        mock_oci_auth = MagicMock()
        adapter = OCIStorageAuthAdapter(mock_oci_auth)

        config = adapter.get_client_config()

        assert isinstance(config, dict)
        assert "auth_provider" in config
        assert config["auth_provider"] is mock_oci_auth
        assert len(config) == 1

    def test_get_credentials(self):
        """Test get_credentials returns the OCI auth provider."""
        mock_oci_auth = MagicMock()
        adapter = OCIStorageAuthAdapter(mock_oci_auth)

        credentials = adapter.get_credentials()
        assert credentials is mock_oci_auth

    def test_get_storage_type(self):
        """Test get_storage_type returns 'oci'."""
        mock_oci_auth = MagicMock()
        adapter = OCIStorageAuthAdapter(mock_oci_auth)

        storage_type = adapter.get_storage_type()
        assert storage_type == "oci"

    def test_get_region_with_region_attribute(self):
        """Test get_region when OCI auth has region attribute."""
        mock_oci_auth = MagicMock()
        mock_oci_auth.region = "us-phoenix-1"

        adapter = OCIStorageAuthAdapter(mock_oci_auth)
        region = adapter.get_region()

        assert region == "us-phoenix-1"

    def test_get_region_without_region_attribute(self):
        """Test get_region when OCI auth doesn't have region attribute."""
        mock_oci_auth = MagicMock()
        # Remove region attribute if it exists
        if hasattr(mock_oci_auth, "region"):
            delattr(mock_oci_auth, "region")

        adapter = OCIStorageAuthAdapter(mock_oci_auth)
        region = adapter.get_region()

        assert region is None

    def test_get_region_with_none_region(self):
        """Test get_region when OCI auth has region set to None."""
        mock_oci_auth = MagicMock()
        mock_oci_auth.region = None

        adapter = OCIStorageAuthAdapter(mock_oci_auth)
        region = adapter.get_region()

        assert region is None

    def test_integration_with_factory(self):
        """Test that adapter works with unified auth factory pattern."""
        from genai_bench.auth.auth_provider import AuthProvider

        # Create a mock OCI auth that implements AuthProvider
        class MockOCIAuth(AuthProvider):
            def __init__(self):
                self.region = "us-ashburn-1"

            def get_config(self):
                return {"region": self.region}

            def get_credentials(self):
                return "mock_signer"

        mock_oci_auth = MockOCIAuth()
        adapter = OCIStorageAuthAdapter(mock_oci_auth)

        # Test that all required methods work
        client_config = adapter.get_client_config()
        assert client_config["auth_provider"] is mock_oci_auth

        assert adapter.get_credentials() is mock_oci_auth
        assert adapter.get_storage_type() == "oci"
        assert adapter.get_region() == "us-ashburn-1"

    def test_error_handling_none_auth(self):
        """Test error handling when OCI auth provider is None."""
        adapter = OCIStorageAuthAdapter(None)

        # Should not raise errors, but handle gracefully
        config = adapter.get_client_config()
        assert config["auth_provider"] is None

        assert adapter.get_credentials() is None
        assert adapter.get_storage_type() == "oci"
        assert adapter.get_region() is None

    def test_multiple_instances(self):
        """Test that multiple adapter instances don't interfere."""
        mock_oci_auth1 = MagicMock()
        mock_oci_auth1.region = "us-phoenix-1"

        mock_oci_auth2 = MagicMock()
        mock_oci_auth2.region = "eu-frankfurt-1"

        adapter1 = OCIStorageAuthAdapter(mock_oci_auth1)
        adapter2 = OCIStorageAuthAdapter(mock_oci_auth2)

        assert adapter1.get_region() == "us-phoenix-1"
        assert adapter2.get_region() == "eu-frankfurt-1"
        assert adapter1.get_credentials() is mock_oci_auth1
        assert adapter2.get_credentials() is mock_oci_auth2

    def test_get_region_with_different_attribute_types(self):
        """Test get_region with different types of region attributes."""
        # Test with property
        mock_oci_auth = MagicMock()
        mock_oci_auth.region = property(lambda self: "ap-tokyo-1")

        adapter = OCIStorageAuthAdapter(mock_oci_auth)
        # MagicMock handles properties differently, so we check if region exists
        assert hasattr(adapter.oci_auth, "region")

        # Test with method that returns region
        class MockAuthWithMethod:
            def get_region(self):
                return "ca-toronto-1"

        mock_auth = MockAuthWithMethod()
        mock_auth.region = mock_auth.get_region()

        adapter = OCIStorageAuthAdapter(mock_auth)
        assert adapter.get_region() == "ca-toronto-1"

    def test_client_config_immutability(self):
        """Test that modifying returned config doesn't affect adapter."""
        mock_oci_auth = MagicMock()
        adapter = OCIStorageAuthAdapter(mock_oci_auth)

        config1 = adapter.get_client_config()
        config1["extra_key"] = "extra_value"

        config2 = adapter.get_client_config()
        assert "extra_key" not in config2
        assert len(config2) == 1
