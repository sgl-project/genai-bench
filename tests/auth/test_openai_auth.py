import pytest

from genai_bench.auth.auth_provider import AuthProvider
from genai_bench.auth.openai.auth import OpenAIAuth

MOCK_API_KEY = "genai-bench-test-123456789"


class MockAuthProvider(AuthProvider):
    """Mock implementation of AuthProvider for testing."""

    def get_config(self):
        return {}

    def get_credentials(self):
        return "mock-credentials"


def test_auth_provider_abstract():
    """Test that AuthProvider cannot be instantiated directly."""
    with pytest.raises(TypeError):
        AuthProvider()


class TestOpenAIAuth:
    def test_init_with_key(self):
        """Test initialization with API key."""
        auth = OpenAIAuth(api_key=MOCK_API_KEY)
        assert auth.api_key == MOCK_API_KEY

    def test_init_with_env(self, monkeypatch):
        """Test initialization with environment variable."""
        monkeypatch.setenv("OPENAI_API_KEY", MOCK_API_KEY)
        auth = OpenAIAuth()
        assert auth.api_key == MOCK_API_KEY

    def test_init_no_key(self, monkeypatch):
        """Test initialization with no API key."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError):
            OpenAIAuth()

    def test_init_empty_key(self):
        """Test initialization with empty API key."""
        with pytest.raises(ValueError):
            OpenAIAuth(api_key="")

    def test_init_whitespace_key(self):
        """Test initialization with whitespace API key."""
        with pytest.raises(ValueError):
            OpenAIAuth(api_key="   ")

    def test_get_config(self):
        """Test getting OpenAI config."""
        auth = OpenAIAuth(api_key=MOCK_API_KEY)
        assert auth.get_config() == {}

    def test_get_credentials(self):
        """Test getting OpenAI credentials."""
        auth = OpenAIAuth(api_key=MOCK_API_KEY)
        assert auth.get_credentials() == MOCK_API_KEY
