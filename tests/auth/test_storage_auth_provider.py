"""Tests for base storage auth provider interface."""

import pytest

from genai_bench.auth.storage_auth_provider import StorageAuthProvider


class TestStorageAuthProvider:
    """Test base storage auth provider interface."""

    def test_abstract_interface(self):
        """Test that StorageAuthProvider cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            StorageAuthProvider()

    def test_abstract_methods(self):
        """Test that abstract methods are defined."""

        # Create a concrete implementation for testing
        class ConcreteStorageAuthProvider(StorageAuthProvider):
            def get_client_config(self):
                return {"region": "us-east-1"}

            def get_credentials(self):
                return {"access_key": "test"}

            def get_storage_type(self):
                return "s3"

            def get_region(self):
                return "us-east-1"

        # Test the concrete implementation
        provider = ConcreteStorageAuthProvider()

        # Test all methods
        assert provider.get_client_config() == {"region": "us-east-1"}
        assert provider.get_credentials() == {"access_key": "test"}
        assert provider.get_storage_type() == "s3"
        assert provider.get_region() == "us-east-1"

    def test_missing_method_implementation(self):
        """Test that missing method implementation raises error."""

        # Create incomplete implementation
        class IncompleteStorageAuthProvider(StorageAuthProvider):
            def get_client_config(self):
                return {}

            def get_credentials(self):
                return {}

            # Missing get_storage_type

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteStorageAuthProvider()

    def test_all_abstract_methods_required(self):
        """Test that all abstract methods must be implemented."""

        # Test with each method missing
        class MissingClientConfig(StorageAuthProvider):
            # Missing get_client_config
            def get_credentials(self):
                return {}

            def get_storage_type(self):
                return "test"

            def get_region(self):
                return "test"

        with pytest.raises(TypeError):
            MissingClientConfig()
