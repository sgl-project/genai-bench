from unittest.mock import MagicMock

import click
import pytest

from genai_bench.cli.validation import validate_api_backend, validate_api_key
from genai_bench.user.fast_openai_user import FastOpenAIUser
from genai_bench.user.openai_user import OpenAIUser


class TestValidateApiBackend:
    def test_validate_api_backend_fast_openai_hyphen(self):
        """Test validation with 'fast-openai' (hyphen)."""
        ctx = MagicMock()
        ctx.obj = {}

        result = validate_api_backend(ctx, None, "fast-openai")

        assert result == "fast-openai"
        assert ctx.obj["user_class"] == FastOpenAIUser

    def test_validate_api_backend_openai(self):
        """Test validation with standard 'openai'."""
        ctx = MagicMock()
        ctx.obj = {}

        result = validate_api_backend(ctx, None, "openai")

        assert result == "openai"
        assert ctx.obj["user_class"] == OpenAIUser

    def test_validate_api_backend_invalid(self):
        """Test validation with invalid backend."""
        ctx = MagicMock()

        with pytest.raises(click.BadParameter):
            validate_api_backend(ctx, None, "invalid-backend")

    def test_validate_api_key_fast_openai_required(self):
        """Test that api_key is required for fast-openai."""
        ctx = MagicMock()
        ctx.params = {"api_backend": "fast-openai"}

        with pytest.raises(click.BadParameter, match="API key is required"):
            validate_api_key(ctx, None, None)

    def test_validate_api_key_fast_openai_provided(self):
        """Test that api_key validation passes when provided for fast-openai."""
        ctx = MagicMock()
        ctx.params = {"api_backend": "fast-openai"}

        result = validate_api_key(ctx, None, "sk-12345")
        assert result == "sk-12345"
