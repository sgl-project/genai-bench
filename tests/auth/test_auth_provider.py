"""Tests for base auth provider interface."""

import pytest

from genai_bench.auth.auth_provider import AuthProvider


class TestAuthProvider:
    """Test base auth provider interface."""

    def test_abstract_interface(self):
        """Test that AuthProvider cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            AuthProvider()

    def test_abstract_methods(self):
        """Test that abstract methods are defined."""

        # Create a concrete implementation for testing
        class ConcreteAuthProvider(AuthProvider):
            def get_credentials(self):
                return "test_credentials"

            def get_config(self):
                return {"api_key": "test"}

        # Test the concrete implementation
        provider = ConcreteAuthProvider()

        # Test all methods
        assert provider.get_credentials() == "test_credentials"
        assert provider.get_config() == {"api_key": "test"}

    def test_missing_method_implementation(self):
        """Test that missing method implementation raises error."""

        # Create incomplete implementation
        class IncompleteAuthProvider(AuthProvider):
            def get_credentials(self):
                return {}

            # Missing get_config

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteAuthProvider()
