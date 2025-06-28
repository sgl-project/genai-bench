"""Tests for base model auth provider interface."""

import pytest

from genai_bench.auth.model_auth_provider import ModelAuthProvider


class TestModelAuthProvider:
    """Test base model auth provider interface."""

    def test_abstract_interface(self):
        """Test that ModelAuthProvider cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            ModelAuthProvider()

    def test_abstract_methods(self):
        """Test that abstract methods are defined."""

        # Create a concrete implementation for testing
        class ConcreteAuthProvider(ModelAuthProvider):
            def get_headers(self):
                return {"Authorization": "Bearer test"}

            def get_config(self):
                return {"api_key": "test"}

            def get_auth_type(self):
                return "api_key"

            def get_credentials(self):
                return {"api_key": "test"}

        # Test the concrete implementation
        provider = ConcreteAuthProvider()

        # Test all methods
        assert provider.get_headers() == {"Authorization": "Bearer test"}
        assert provider.get_config() == {"api_key": "test"}
        assert provider.get_auth_type() == "api_key"
        assert provider.get_credentials() == {"api_key": "test"}

    def test_missing_method_implementation(self):
        """Test that missing method implementation raises error."""

        # Create incomplete implementation
        class IncompleteAuthProvider(ModelAuthProvider):
            def get_headers(self):
                return {}

            def get_config(self):
                return {}

            # Missing get_auth_type

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteAuthProvider()

    def test_all_abstract_methods_required(self):
        """Test that all abstract methods must be implemented."""

        # Test with each method missing
        class MissingGetHeaders(ModelAuthProvider):
            # Missing get_headers
            def get_config(self):
                return {}

            def get_auth_type(self):
                return "test"

            def get_credentials(self):
                return {}

        with pytest.raises(TypeError):
            MissingGetHeaders()
