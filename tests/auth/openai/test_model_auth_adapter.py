"""Unit tests for OpenAI model authentication adapter."""

from unittest.mock import MagicMock, patch

import pytest

from genai_bench.auth.model_auth_provider import ModelAuthProvider
from genai_bench.auth.openai.auth import OpenAIAuth
from genai_bench.auth.openai.model_auth_adapter import OpenAIModelAuthAdapter


class TestOpenAIModelAuthAdapter:
    """Test cases for OpenAI model authentication adapter."""

    def test_inheritance(self):
        """Test that OpenAIModelAuthAdapter inherits from ModelAuthProvider."""
        mock_openai_auth = MagicMock()
        adapter = OpenAIModelAuthAdapter(mock_openai_auth)
        assert isinstance(adapter, ModelAuthProvider)

    def test_init(self):
        """Test initialization with OpenAI auth provider."""
        mock_openai_auth = MagicMock()
        adapter = OpenAIModelAuthAdapter(mock_openai_auth)
        assert adapter.openai_auth is mock_openai_auth

    def test_get_headers_with_mock(self):
        """Test get_headers when OpenAI auth has get_headers method."""
        mock_openai_auth = MagicMock()
        mock_openai_auth.api_key = "test-key"

        adapter = OpenAIModelAuthAdapter(mock_openai_auth)
        headers = adapter.get_headers()

        assert headers == {"Authorization": "Bearer test-key"}

    def test_get_headers_without_method(self):
        """Test get_headers when OpenAI auth doesn't have get_headers method."""
        # Create a real OpenAI auth instance
        openai_auth = OpenAIAuth(api_key="test-api-key")
        adapter = OpenAIModelAuthAdapter(openai_auth)

        # The adapter implements get_headers, so it should work
        headers = adapter.get_headers()
        assert headers == {"Authorization": "Bearer test-api-key"}

    def test_get_headers_implementation_suggestion(self):
        """Test suggested implementation for get_headers."""
        # Mock OpenAI auth with api_key attribute
        mock_openai_auth = MagicMock()
        mock_openai_auth.api_key = "test-api-key"

        # Mock get_headers to return proper OpenAI headers
        mock_openai_auth.get_headers.return_value = {
            "Authorization": f"Bearer {mock_openai_auth.api_key}"
        }

        adapter = OpenAIModelAuthAdapter(mock_openai_auth)
        headers = adapter.get_headers()

        assert headers == {"Authorization": "Bearer test-api-key"}

    def test_get_config(self):
        """Test get_config returns proper configuration."""
        mock_openai_auth = MagicMock()
        mock_openai_auth.api_key = "test-api-key"

        adapter = OpenAIModelAuthAdapter(mock_openai_auth)
        config = adapter.get_config()

        assert isinstance(config, dict)
        assert config["auth_type"] == "api_key"
        assert config["has_api_key"] is True
        assert len(config) == 2

    def test_get_config_no_api_key(self):
        """Test get_config when API key is None."""
        mock_openai_auth = MagicMock()
        mock_openai_auth.api_key = None

        adapter = OpenAIModelAuthAdapter(mock_openai_auth)
        config = adapter.get_config()

        assert config["has_api_key"] is False

    def test_get_config_empty_api_key(self):
        """Test get_config when API key is empty string."""
        mock_openai_auth = MagicMock()
        mock_openai_auth.api_key = ""

        adapter = OpenAIModelAuthAdapter(mock_openai_auth)
        config = adapter.get_config()

        assert config["has_api_key"] is False

    def test_get_auth_type(self):
        """Test get_auth_type returns 'api_key'."""
        mock_openai_auth = MagicMock()
        adapter = OpenAIModelAuthAdapter(mock_openai_auth)

        auth_type = adapter.get_auth_type()
        assert auth_type == "api_key"

    def test_get_credentials(self):
        """Test get_credentials returns dict with API key."""
        mock_openai_auth = MagicMock()
        mock_openai_auth.api_key = "test-api-key-123"

        adapter = OpenAIModelAuthAdapter(mock_openai_auth)
        credentials = adapter.get_credentials()

        assert isinstance(credentials, dict)
        assert credentials["api_key"] == "test-api-key-123"
        assert len(credentials) == 1

    def test_get_credentials_with_none_key(self):
        """Test get_credentials when API key is None."""
        mock_openai_auth = MagicMock()
        mock_openai_auth.api_key = None

        adapter = OpenAIModelAuthAdapter(mock_openai_auth)
        credentials = adapter.get_credentials()

        assert credentials == {}

    def test_integration_with_real_openai_auth(self):
        """Test adapter with real OpenAIAuth instance."""
        openai_auth = OpenAIAuth(api_key="sk-test-key-123")
        adapter = OpenAIModelAuthAdapter(openai_auth)

        # Test config
        config = adapter.get_config()
        assert config["auth_type"] == "api_key"
        assert config["has_api_key"] is True

        # Test credentials
        credentials = adapter.get_credentials()
        assert credentials["api_key"] == "sk-test-key-123"

        # Test auth type
        assert adapter.get_auth_type() == "api_key"

    def test_integration_with_factory(self):
        """Test that adapter works with unified auth factory pattern."""
        mock_openai_auth = MagicMock()
        mock_openai_auth.api_key = "factory-test-key"
        mock_openai_auth.get_headers.return_value = {
            "Authorization": "Bearer factory-test-key"
        }

        adapter = OpenAIModelAuthAdapter(mock_openai_auth)

        # Test all required methods
        assert adapter.get_auth_type() == "api_key"
        assert adapter.get_config()["has_api_key"] is True
        assert adapter.get_credentials()["api_key"] == "factory-test-key"
        headers = adapter.get_headers()
        assert headers["Authorization"] == "Bearer factory-test-key"

    def test_multiple_instances(self):
        """Test that multiple adapter instances don't interfere."""
        mock_auth1 = MagicMock()
        mock_auth1.api_key = "key1"

        mock_auth2 = MagicMock()
        mock_auth2.api_key = "key2"

        adapter1 = OpenAIModelAuthAdapter(mock_auth1)
        adapter2 = OpenAIModelAuthAdapter(mock_auth2)

        assert adapter1.get_credentials()["api_key"] == "key1"
        assert adapter2.get_credentials()["api_key"] == "key2"
        assert adapter1.openai_auth is mock_auth1
        assert adapter2.openai_auth is mock_auth2

    def test_error_handling_missing_attributes(self):
        """Test error handling when OpenAI auth is missing expected attributes."""
        mock_openai_auth = MagicMock()
        # Remove api_key attribute
        del mock_openai_auth.api_key

        adapter = OpenAIModelAuthAdapter(mock_openai_auth)

        # Should raise AttributeError when trying to access api_key
        with pytest.raises(AttributeError):
            adapter.get_config()

        with pytest.raises(AttributeError):
            adapter.get_credentials()

    @patch("genai_bench.auth.openai.auth.OpenAIAuth")
    def test_with_patched_openai_auth(self, mock_openai_class):
        """Test adapter with patched OpenAIAuth class."""
        # Configure the mock instance
        mock_instance = MagicMock()
        mock_instance.api_key = "patched-key"
        mock_instance.get_headers.return_value = {"Authorization": "Bearer patched-key"}
        mock_openai_class.return_value = mock_instance

        # Create adapter with the mocked instance
        adapter = OpenAIModelAuthAdapter(mock_instance)

        # Verify behavior
        assert adapter.get_credentials()["api_key"] == "patched-key"
        assert adapter.get_headers()["Authorization"] == "Bearer patched-key"
