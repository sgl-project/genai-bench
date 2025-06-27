"""Unit tests for AWS S3 storage authentication provider."""

import os
from unittest.mock import patch

from genai_bench.auth.aws.s3_auth import AWSS3Auth


class TestAWSS3Auth:
    """Test cases for AWS S3 authentication."""

    def test_init_with_explicit_credentials(self):
        """Test initialization with explicitly provided credentials."""
        auth = AWSS3Auth(
            access_key_id="s3_key",
            secret_access_key="s3_secret",
            session_token="s3_token",
            region="eu-central-1",
            profile="s3_profile",
        )

        assert auth.access_key_id == "s3_key"
        assert auth.secret_access_key == "s3_secret"
        assert auth.session_token == "s3_token"
        assert auth.region == "eu-central-1"
        assert auth.profile == "s3_profile"

    def test_get_client_config_with_profile(self):
        """Test client configuration when using profile."""
        auth = AWSS3Auth(profile="test_profile", region="ap-southeast-1")
        config = auth.get_client_config()

        assert config["region_name"] == "ap-southeast-1"
        assert config["profile_name"] == "test_profile"
        assert "aws_access_key_id" not in config

    def test_get_client_config_with_credentials(self):
        """Test client configuration when using explicit credentials."""
        auth = AWSS3Auth(
            access_key_id="key",
            secret_access_key="secret",
            session_token="token",
            region="us-west-1",
        )
        config = auth.get_client_config()

        assert config["region_name"] == "us-west-1"
        assert config["aws_access_key_id"] == "key"
        assert config["aws_secret_access_key"] == "secret"
        assert config["aws_session_token"] == "token"
        assert "profile_name" not in config

    def test_get_client_config_partial_credentials(self):
        """Test client configuration with partial credentials."""
        auth = AWSS3Auth(
            access_key_id="key",
            secret_access_key="secret",
            # No session token
        )
        config = auth.get_client_config()

        assert config["aws_access_key_id"] == "key"
        assert config["aws_secret_access_key"] == "secret"
        assert "aws_session_token" not in config

    def test_get_client_config_no_region(self):
        """Test client configuration without region."""
        auth = AWSS3Auth(profile="test")
        config = auth.get_client_config()

        assert config["profile_name"] == "test"
        assert "region_name" not in config

    def test_get_credentials(self):
        """Test getting auth credentials."""
        auth = AWSS3Auth(
            access_key_id="key", secret_access_key="secret", region="us-east-2"
        )
        creds = auth.get_credentials()

        assert creds["aws_access_key_id"] == "key"
        assert creds["aws_secret_access_key"] == "secret"
        assert creds["region_name"] == "us-east-2"
        assert creds["aws_session_token"] is None
        assert creds["profile_name"] is None

    def test_get_storage_type(self):
        """Test storage type identifier."""
        auth = AWSS3Auth()
        assert auth.get_storage_type() == "aws"

    def test_get_region(self):
        """Test getting region."""
        auth = AWSS3Auth(region="sa-east-1")
        assert auth.get_region() == "sa-east-1"

    def test_get_region_none(self):
        """Test getting region when not set."""
        with patch.dict(os.environ, {}, clear=True):
            auth = AWSS3Auth()
            assert auth.get_region() == "us-east-1"  # Default

    def test_env_var_fallback(self):
        """Test fallback to environment variables."""
        with patch.dict(
            os.environ,
            {
                "AWS_ACCESS_KEY_ID": "env_key",
                "AWS_SECRET_ACCESS_KEY": "env_secret",
                "AWS_SESSION_TOKEN": "env_token",
                "AWS_DEFAULT_REGION": "env_region",
                "AWS_PROFILE": "env_profile",
            },
        ):
            auth = AWSS3Auth()

            assert auth.access_key_id == "env_key"
            assert auth.secret_access_key == "env_secret"
            assert auth.session_token == "env_token"
            assert auth.region == "env_region"
            assert auth.profile == "env_profile"

    def test_cross_credential_usage(self):
        """Test that S3 auth can use different credentials than model auth."""
        # This tests the separation of concerns - S3 auth is independent
        model_auth = AWSS3Auth(access_key_id="model_key", profile="model_profile")
        storage_auth = AWSS3Auth(access_key_id="storage_key", profile="storage_profile")

        assert model_auth.access_key_id != storage_auth.access_key_id
        assert model_auth.profile != storage_auth.profile
