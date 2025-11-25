import pytest

from genai_bench.auth.auth_provider import AuthProvider
from genai_bench.auth.together.auth import TogetherAuth

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


class TestTogetherAuth:
    def test_init_with_key(self):
        """Test initialization with API key."""
        auth = TogetherAuth(api_key=MOCK_API_KEY)
        assert auth.api_key == MOCK_API_KEY

    def test_init_with_env(self, monkeypatch):
        """Test initialization with environment variable."""
        monkeypatch.setenv("TOGETHER_API_KEY", MOCK_API_KEY)
        auth = TogetherAuth()
        assert auth.api_key == MOCK_API_KEY

    def test_init_no_key(self, monkeypatch):
        """Test initialization with no API key."""
        monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
        with pytest.raises(ValueError):
            TogetherAuth()

    def test_init_empty_key(self):
        """Test initialization with empty API key."""
        with pytest.raises(ValueError):
            TogetherAuth(api_key="")

    def test_init_whitespace_key(self):
        """Test initialization with whitespace API key."""
        with pytest.raises(ValueError):
            TogetherAuth(api_key="   ")

    def test_get_config(self):
        """Test getting Together config."""
        auth = TogetherAuth(api_key=MOCK_API_KEY)
        assert auth.get_config() == {}

    def test_get_credentials(self):
        """Test getting Together credentials."""
        auth = TogetherAuth(api_key=MOCK_API_KEY)
        assert auth.get_credentials() == MOCK_API_KEY
