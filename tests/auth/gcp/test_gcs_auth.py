"""Unit tests for GCP Cloud Storage authentication provider."""

import os
from unittest.mock import patch

from genai_bench.auth.gcp.gcs_auth import GCPStorageAuth


class TestGCPStorageAuth:
    """Test cases for GCP Cloud Storage authentication."""

    def test_init_with_service_account(self):
        """Test initialization with service account credentials."""
        auth = GCPStorageAuth(
            project_id="storage-project", credentials_path="/path/to/storage-creds.json"
        )

        assert auth.project_id == "storage-project"
        assert auth.credentials_path == "/path/to/storage-creds.json"
        assert auth.access_token is None

    def test_init_with_access_token(self):
        """Test initialization with access token."""
        auth = GCPStorageAuth(
            project_id="storage-project", access_token="ya29.test_token"
        )

        assert auth.project_id == "storage-project"
        assert auth.access_token == "ya29.test_token"
        assert auth.credentials_path is None

    def test_init_with_env_vars(self):
        """Test initialization with environment variables."""
        with patch.dict(
            os.environ,
            {
                "GCP_PROJECT_ID": "env-storage-project",
                "GOOGLE_APPLICATION_CREDENTIALS": "/env/storage-creds.json",
                "GCP_ACCESS_TOKEN": "env_access_token",
            },
        ):
            auth = GCPStorageAuth()

            assert auth.project_id == "env-storage-project"
            assert auth.credentials_path == "/env/storage-creds.json"
            assert auth.access_token == "env_access_token"

    def test_init_minimal(self):
        """Test initialization with minimal configuration."""
        with patch.dict(os.environ, {}, clear=True):
            auth = GCPStorageAuth()

            assert auth.project_id is None
            assert auth.credentials_path is None
            assert auth.access_token is None

    def test_get_client_config_with_credentials_path(self):
        """Test client config with service account credentials."""
        auth = GCPStorageAuth(
            project_id="test-project", credentials_path="/path/to/creds.json"
        )
        config = auth.get_client_config()

        assert config == {
            "project": "test-project",
            "credentials_path": "/path/to/creds.json",
        }

    def test_get_client_config_with_access_token(self):
        """Test client config with access token."""
        auth = GCPStorageAuth(project_id="test-project", access_token="ya29.token")
        config = auth.get_client_config()

        assert config == {"project": "test-project", "access_token": "ya29.token"}

    def test_get_client_config_no_project(self):
        """Test client config without project ID."""
        auth = GCPStorageAuth(credentials_path="/path/to/creds.json")
        config = auth.get_client_config()

        assert config == {"credentials_path": "/path/to/creds.json"}

    def test_get_client_config_empty(self):
        """Test client config with no configuration."""
        auth = GCPStorageAuth()
        config = auth.get_client_config()

        assert config == {}

    def test_get_credentials_same_format(self):
        """Test auth credentials format."""
        auth = GCPStorageAuth(
            project_id="test-project", credentials_path="/path/to/creds.json"
        )
        creds = auth.get_credentials()

        assert creds == {
            "project_id": "test-project",
            "credentials_path": "/path/to/creds.json",
        }

    def test_get_credentials_access_token(self):
        """Test auth credentials with access token."""
        auth = GCPStorageAuth(project_id="test-project", access_token="token")
        creds = auth.get_credentials()

        assert creds == {"project_id": "test-project", "access_token": "token"}

    def test_get_storage_type(self):
        """Test storage type identifier."""
        auth = GCPStorageAuth()
        assert auth.get_storage_type() == "gcp"

    def test_precedence_credentials_over_token(self):
        """Test that credentials path takes precedence over access token."""
        auth = GCPStorageAuth(
            credentials_path="/path/to/creds.json", access_token="ignored_token"
        )
        config = auth.get_client_config()

        assert "credentials_path" in config
        assert "access_token" not in config

    def test_precedence_explicit_over_env(self):
        """Test that explicit values take precedence over env vars."""
        with patch.dict(
            os.environ,
            {
                "GCP_PROJECT_ID": "env-project",
                "GOOGLE_APPLICATION_CREDENTIALS": "/env/creds.json",
                "GCP_ACCESS_TOKEN": "env_token",
            },
        ):
            auth = GCPStorageAuth(
                project_id="explicit-project",
                credentials_path="/explicit/creds.json",
                access_token="explicit_token",
            )

            assert auth.project_id == "explicit-project"
            assert auth.credentials_path == "/explicit/creds.json"
            assert auth.access_token == "explicit_token"

    def test_cross_service_auth_separation(self):
        """Test that storage auth can differ from Vertex AI auth."""
        # This verifies the separation of concerns
        vertex_auth = GCPStorageAuth(
            project_id="vertex-project", credentials_path="/vertex/creds.json"
        )
        storage_auth = GCPStorageAuth(
            project_id="storage-project", access_token="storage_token"
        )

        assert vertex_auth.project_id != storage_auth.project_id
        assert vertex_auth.credentials_path != storage_auth.access_token
