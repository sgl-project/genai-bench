"""Tests for Baseten authentication provider."""

import os
from unittest.mock import patch

import pytest

from genai_bench.auth.baseten.auth import BasetenAuth
from genai_bench.auth.baseten.model_auth_adapter import BasetenModelAuthAdapter


class TestBasetenAuth:
    """Test Baseten authentication provider."""

    def test_init_with_api_key(self):
        """Test initialization with explicit API key."""
        auth = BasetenAuth(api_key="test-api-key")
        assert auth.api_key == "test-api-key"

    def test_init_with_env_var(self):
        """Test initialization with environment variable."""
        with patch.dict(os.environ, {"BASETEN_API_KEY": "env-api-key"}):
            auth = BasetenAuth()
            assert auth.api_key == "env-api-key"

    def test_init_with_no_api_key(self):
        """Test initialization without API key raises error."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Baseten API key must be provided"):
                BasetenAuth()

    def test_init_with_empty_api_key(self):
        """Test initialization with empty API key raises error."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Baseten API key must be provided"):
                BasetenAuth(api_key="")

    def test_get_config(self):
        """Test get_config returns empty dict."""
        auth = BasetenAuth(api_key="test-key")
        config = auth.get_config()
        assert config == {}

    def test_get_credentials(self):
        """Test get_credentials returns API key."""
        auth = BasetenAuth(api_key="test-key")
        credentials = auth.get_credentials()
        assert credentials == "test-key"

    def test_get_credentials_no_api_key(self):
        """Test get_credentials without API key raises error."""
        auth = BasetenAuth(api_key="test-key")
        auth.api_key = None
        with pytest.raises(ValueError, match="Baseten API key is not set"):
            auth.get_credentials()


class TestBasetenModelAuthAdapter:
    """Test Baseten model auth adapter."""

    def test_init(self):
        """Test adapter initialization."""
        baseten_auth = BasetenAuth(api_key="test-key")
        adapter = BasetenModelAuthAdapter(baseten_auth)
        assert adapter.baseten_auth == baseten_auth

    def test_get_headers(self):
        """Test get_headers returns correct Authorization header."""
        baseten_auth = BasetenAuth(api_key="test-key")
        adapter = BasetenModelAuthAdapter(baseten_auth)
        headers = adapter.get_headers()
        assert headers == {"Authorization": "Api-Key test-key"}

    def test_get_config(self):
        """Test get_config returns baseten config."""
        baseten_auth = BasetenAuth(api_key="test-key")
        adapter = BasetenModelAuthAdapter(baseten_auth)
        config = adapter.get_config()
        assert config == {}

    def test_get_auth_type(self):
        """Test get_auth_type returns 'api_key'."""
        baseten_auth = BasetenAuth(api_key="test-key")
        adapter = BasetenModelAuthAdapter(baseten_auth)
        auth_type = adapter.get_auth_type()
        assert auth_type == "api_key"

    def test_get_credentials(self):
        """Test get_credentials returns API key."""
        baseten_auth = BasetenAuth(api_key="test-key")
        adapter = BasetenModelAuthAdapter(baseten_auth)
        credentials = adapter.get_credentials()
        assert credentials == "test-key"
