"""Unit tests for Azure OpenAI authentication provider."""

import os
from unittest.mock import patch

from genai_bench.auth.azure.openai_auth import AzureOpenAIAuth


class TestAzureOpenAIAuth:
    """Test cases for Azure OpenAI authentication."""

    def test_init_with_api_key(self):
        """Test initialization with API key authentication."""
        auth = AzureOpenAIAuth(
            api_key="test_key",
            api_version="2024-02-01",
            azure_endpoint="https://test.openai.azure.com",
            azure_deployment="gpt-4",
        )

        assert auth.api_key == "test_key"
        assert auth.api_version == "2024-02-01"
        assert auth.azure_endpoint == "https://test.openai.azure.com"
        assert auth.azure_deployment == "gpt-4"
        assert auth.use_azure_ad is False
        assert auth.azure_ad_token is None

    def test_init_with_azure_ad(self):
        """Test initialization with Azure AD authentication."""
        auth = AzureOpenAIAuth(
            use_azure_ad=True,
            azure_ad_token="test_token",
            azure_endpoint="https://test.openai.azure.com",
            azure_deployment="gpt-35-turbo",
        )

        assert auth.api_key is None
        assert auth.use_azure_ad is True
        assert auth.azure_ad_token == "test_token"
        assert auth.azure_endpoint == "https://test.openai.azure.com"
        assert auth.azure_deployment == "gpt-35-turbo"

    def test_init_with_env_vars(self):
        """Test initialization with environment variables."""
        with patch.dict(
            os.environ,
            {
                "AZURE_OPENAI_API_KEY": "env_key",
                "AZURE_OPENAI_ENDPOINT": "https://env.openai.azure.com",
                "AZURE_OPENAI_DEPLOYMENT": "env_deployment",
                "AZURE_AD_TOKEN": "env_token",
            },
        ):
            auth = AzureOpenAIAuth()

            assert auth.api_key == "env_key"
            assert auth.azure_endpoint == "https://env.openai.azure.com"
            assert auth.azure_deployment == "env_deployment"
            assert auth.azure_ad_token == "env_token"

    def test_default_api_version(self):
        """Test default API version."""
        auth = AzureOpenAIAuth()
        assert auth.api_version == "2024-02-01"

    def test_get_headers_with_api_key(self):
        """Test headers when using API key authentication."""
        auth = AzureOpenAIAuth(api_key="test_key")
        headers = auth.get_headers()

        assert headers == {"api-key": "test_key"}

    def test_get_headers_with_azure_ad(self):
        """Test headers when using Azure AD authentication."""
        auth = AzureOpenAIAuth(use_azure_ad=True, azure_ad_token="bearer_token")
        headers = auth.get_headers()

        assert headers == {"Authorization": "Bearer bearer_token"}

    def test_get_headers_no_auth(self):
        """Test headers when no authentication is provided."""
        auth = AzureOpenAIAuth()
        headers = auth.get_headers()

        assert headers == {}

    def test_get_headers_azure_ad_without_token(self):
        """Test headers when Azure AD is enabled but no token provided."""
        auth = AzureOpenAIAuth(use_azure_ad=True)
        headers = auth.get_headers()

        assert headers == {}

    def test_get_config(self):
        """Test configuration dictionary."""
        auth = AzureOpenAIAuth(
            api_key="key",
            api_version="2023-12-01",
            azure_endpoint="https://test.openai.azure.com",
            azure_deployment="gpt-4",
            use_azure_ad=False,
        )
        config = auth.get_config()

        assert config["api_version"] == "2023-12-01"
        assert config["auth_type"] == "api_key"
        assert config["azure_endpoint"] == "https://test.openai.azure.com"
        assert config["azure_deployment"] == "gpt-4"
        assert "use_azure_ad" not in config  # False is not included

    def test_get_config_with_azure_ad(self):
        """Test configuration with Azure AD enabled."""
        auth = AzureOpenAIAuth(use_azure_ad=True)
        config = auth.get_config()

        assert config["auth_type"] == "azure_ad"
        assert config["use_azure_ad"] is True

    def test_get_auth_type_api_key(self):
        """Test auth type for API key authentication."""
        auth = AzureOpenAIAuth(api_key="key")
        assert auth.get_auth_type() == "api_key"

    def test_get_auth_type_azure_ad(self):
        """Test auth type for Azure AD authentication."""
        auth = AzureOpenAIAuth(use_azure_ad=True)
        assert auth.get_auth_type() == "azure_ad"

    def test_get_credentials_api_key(self):
        """Test credentials for API key authentication."""
        auth = AzureOpenAIAuth(api_key="test_key")
        creds = auth.get_credentials()

        assert creds == {"api_key": "test_key"}

    def test_get_credentials_azure_ad(self):
        """Test credentials for Azure AD authentication."""
        auth = AzureOpenAIAuth(use_azure_ad=True, azure_ad_token="test_token")
        creds = auth.get_credentials()

        assert creds == {"azure_ad_token": "test_token"}

    def test_get_credentials_no_auth(self):
        """Test credentials when no authentication is set."""
        auth = AzureOpenAIAuth()
        creds = auth.get_credentials()

        assert creds == {"api_key": None}

    def test_precedence_explicit_over_env(self):
        """Test that explicit values take precedence over env vars."""
        with patch.dict(
            os.environ,
            {
                "AZURE_OPENAI_API_KEY": "env_key",
                "AZURE_OPENAI_ENDPOINT": "https://env.openai.azure.com",
            },
        ):
            auth = AzureOpenAIAuth(
                api_key="explicit_key",
                azure_endpoint="https://explicit.openai.azure.com",
            )

            assert auth.api_key == "explicit_key"
            assert auth.azure_endpoint == "https://explicit.openai.azure.com"
