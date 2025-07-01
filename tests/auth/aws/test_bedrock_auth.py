"""Unit tests for AWS Bedrock authentication provider."""

import os
from unittest.mock import patch

from genai_bench.auth.aws.bedrock_auth import AWSBedrockAuth


class TestAWSBedrockAuth:
    """Test cases for AWS Bedrock authentication."""

    def test_init_with_explicit_credentials(self):
        """Test initialization with explicitly provided credentials."""
        auth = AWSBedrockAuth(
            access_key_id="test_key",
            secret_access_key="test_secret",
            session_token="test_token",
            region="us-west-2",
            profile="test_profile",
        )

        assert auth.access_key_id == "test_key"
        assert auth.secret_access_key == "test_secret"
        assert auth.session_token == "test_token"
        assert auth.region == "us-west-2"
        assert auth.profile == "test_profile"

    def test_init_with_env_vars(self):
        """Test initialization with environment variables."""
        with patch.dict(
            os.environ,
            {
                "AWS_ACCESS_KEY_ID": "env_key",
                "AWS_SECRET_ACCESS_KEY": "env_secret",
                "AWS_SESSION_TOKEN": "env_token",
                "AWS_DEFAULT_REGION": "eu-west-1",
                "AWS_PROFILE": "env_profile",
            },
        ):
            auth = AWSBedrockAuth()

            assert auth.access_key_id == "env_key"
            assert auth.secret_access_key == "env_secret"
            assert auth.session_token == "env_token"
            assert auth.region == "eu-west-1"
            assert auth.profile == "env_profile"

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        with patch.dict(os.environ, {}, clear=True):
            auth = AWSBedrockAuth()

            assert auth.access_key_id is None
            assert auth.secret_access_key is None
            assert auth.session_token is None
            assert auth.region == "us-east-1"  # Default region
            assert auth.profile is None

    def test_get_headers(self):
        """Test that headers are empty for AWS (auth handled by SDK)."""
        auth = AWSBedrockAuth(access_key_id="test_key")
        headers = auth.get_headers()

        assert headers == {}

    def test_get_config_with_profile(self):
        """Test configuration when using profile."""
        auth = AWSBedrockAuth(profile="test_profile", region="us-west-2")
        config = auth.get_config()

        assert config["region"] == "us-west-2"
        assert config["auth_type"] == "aws_bedrock"
        assert config["profile"] == "test_profile"
        assert "access_key_id" not in config

    def test_get_config_with_credentials(self):
        """Test configuration when using explicit credentials."""
        auth = AWSBedrockAuth(
            access_key_id="test_key",
            secret_access_key="test_secret",
            session_token="test_token",
            region="us-east-1",
        )
        config = auth.get_config()

        assert config["region"] == "us-east-1"
        assert config["auth_type"] == "aws_bedrock"
        assert config["access_key_id"] == "test_key"
        assert config["session_token"] == "test_token"
        assert "profile" not in config

    def test_get_config_without_session_token(self):
        """Test configuration without session token."""
        auth = AWSBedrockAuth(access_key_id="test_key", secret_access_key="test_secret")
        config = auth.get_config()

        assert "session_token" not in config

    def test_get_auth_type(self):
        """Test auth type identifier."""
        auth = AWSBedrockAuth()
        assert auth.get_auth_type() == "aws_bedrock"

    def test_get_credentials(self):
        """Test getting raw credentials."""
        auth = AWSBedrockAuth(
            access_key_id="test_key",
            secret_access_key="test_secret",
            session_token="test_token",
            region="us-west-2",
            profile="test_profile",
        )
        creds = auth.get_credentials()

        assert creds["aws_access_key_id"] == "test_key"
        assert creds["aws_secret_access_key"] == "test_secret"
        assert creds["aws_session_token"] == "test_token"
        assert creds["region_name"] == "us-west-2"
        assert creds["profile_name"] == "test_profile"

    def test_precedence_explicit_over_env(self):
        """Test that explicit credentials take precedence over env vars."""
        with patch.dict(
            os.environ, {"AWS_ACCESS_KEY_ID": "env_key", "AWS_PROFILE": "env_profile"}
        ):
            auth = AWSBedrockAuth(
                access_key_id="explicit_key", profile="explicit_profile"
            )

            assert auth.access_key_id == "explicit_key"
            assert auth.profile == "explicit_profile"
