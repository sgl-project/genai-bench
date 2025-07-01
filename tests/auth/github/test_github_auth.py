"""Unit tests for GitHub storage authentication provider."""

import os
from unittest.mock import patch

from genai_bench.auth.github.github_auth import GitHubAuth


class TestGitHubAuth:
    """Test cases for GitHub storage authentication."""

    def test_init_with_all_params(self):
        """Test initialization with all parameters."""
        auth = GitHubAuth(token="ghp_test_token", owner="test-owner", repo="test-repo")

        assert auth.token == "ghp_test_token"
        assert auth.owner == "test-owner"
        assert auth.repo == "test-repo"

    def test_init_with_env_vars(self):
        """Test initialization with environment variables."""
        with patch.dict(
            os.environ,
            {
                "GITHUB_TOKEN": "env_token",
                "GITHUB_OWNER": "env_owner",
                "GITHUB_REPO": "env_repo",
            },
        ):
            auth = GitHubAuth()

            assert auth.token == "env_token"
            assert auth.owner == "env_owner"
            assert auth.repo == "env_repo"

    def test_init_minimal(self):
        """Test initialization with no parameters."""
        with patch.dict(os.environ, {}, clear=True):
            auth = GitHubAuth()

            assert auth.token is None
            assert auth.owner is None
            assert auth.repo is None

    def test_get_client_config_full(self):
        """Test client config with all parameters."""
        auth = GitHubAuth(token="test_token", owner="test_owner", repo="test_repo")
        config = auth.get_client_config()

        assert config == {
            "owner": "test_owner",
            "repo": "test_repo",
            "token": "test_token",
        }

    def test_get_client_config_partial(self):
        """Test client config with partial parameters."""
        auth = GitHubAuth(
            token="test_token",
            owner="test_owner",
            # No repo
        )
        config = auth.get_client_config()

        assert config == {"owner": "test_owner", "token": "test_token"}

    def test_get_client_config_empty(self):
        """Test client config with no parameters."""
        auth = GitHubAuth()
        config = auth.get_client_config()

        assert config == {}

    def test_get_credentials(self):
        """Test getting auth credentials."""
        auth = GitHubAuth(token="test_token", owner="test_owner", repo="test_repo")
        creds = auth.get_credentials()

        assert creds == {
            "token": "test_token",
            "owner": "test_owner",
            "repo": "test_repo",
        }

    def test_get_credentials_no_token(self):
        """Test auth credentials without token."""
        auth = GitHubAuth(owner="test_owner", repo="test_repo")
        creds = auth.get_credentials()

        assert creds == {"owner": "test_owner", "repo": "test_repo"}

    def test_get_storage_type(self):
        """Test storage type identifier."""
        auth = GitHubAuth()
        assert auth.get_storage_type() == "github"

    def test_precedence_explicit_over_env(self):
        """Test that explicit values take precedence over env vars."""
        with patch.dict(
            os.environ,
            {
                "GITHUB_TOKEN": "env_token",
                "GITHUB_OWNER": "env_owner",
                "GITHUB_REPO": "env_repo",
            },
        ):
            auth = GitHubAuth(
                token="explicit_token", owner="explicit_owner", repo="explicit_repo"
            )

            assert auth.token == "explicit_token"
            assert auth.owner == "explicit_owner"
            assert auth.repo == "explicit_repo"

    def test_partial_env_override(self):
        """Test partial override of environment variables."""
        with patch.dict(
            os.environ,
            {
                "GITHUB_TOKEN": "env_token",
                "GITHUB_OWNER": "env_owner",
                "GITHUB_REPO": "env_repo",
            },
        ):
            auth = GitHubAuth(
                owner="explicit_owner"
                # Use env token and repo
            )

            assert auth.token == "env_token"  # From env
            assert auth.owner == "explicit_owner"  # Explicit
            assert auth.repo == "env_repo"  # From env

    def test_github_enterprise_format(self):
        """Test that owner/repo format is supported."""
        auth = GitHubAuth(
            token="token",
            owner="myorg/myrepo",  # Combined format
        )

        # The implementation should handle splitting this
        assert auth.owner == "myorg/myrepo"  # Kept as-is for now
        assert auth.repo is None
