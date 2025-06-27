"""Unit tests for GCP Vertex AI authentication provider."""

import os
from unittest.mock import patch

from genai_bench.auth.gcp.vertex_auth import GCPVertexAuth


class TestGCPVertexAuth:
    """Test cases for GCP Vertex AI authentication."""

    def test_init_with_service_account(self):
        """Test initialization with service account credentials."""
        auth = GCPVertexAuth(
            project_id="test-project",
            location="us-central1",
            credentials_path="/path/to/creds.json",
        )

        assert auth.project_id == "test-project"
        assert auth.location == "us-central1"
        assert auth.credentials_path == "/path/to/creds.json"
        assert auth.api_key is None

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        auth = GCPVertexAuth(
            project_id="test-project", location="europe-west1", api_key="test_api_key"
        )

        assert auth.project_id == "test-project"
        assert auth.location == "europe-west1"
        assert auth.api_key == "test_api_key"
        assert auth.credentials_path is None

    def test_init_with_env_vars(self):
        """Test initialization with environment variables."""
        with patch.dict(
            os.environ,
            {
                "GCP_PROJECT_ID": "env-project",
                "GCP_LOCATION": "asia-east1",
                "GOOGLE_APPLICATION_CREDENTIALS": "/env/path/creds.json",
                "GCP_API_KEY": "env_api_key",
            },
        ):
            auth = GCPVertexAuth()

            assert auth.project_id == "env-project"
            assert auth.location == "asia-east1"
            assert auth.credentials_path == "/env/path/creds.json"
            assert auth.api_key == "env_api_key"

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        with patch.dict(os.environ, {}, clear=True):
            auth = GCPVertexAuth(project_id="test-project")

            assert auth.project_id == "test-project"
            assert auth.location == "us-central1"  # Default location
            assert auth.credentials_path is None
            assert auth.api_key is None

    def test_get_headers_with_api_key(self):
        """Test headers when using API key authentication."""
        auth = GCPVertexAuth(api_key="test_key")
        headers = auth.get_headers()

        assert headers == {"x-goog-api-key": "test_key"}

    def test_get_headers_with_service_account(self):
        """Test headers when using service account (empty headers)."""
        auth = GCPVertexAuth(credentials_path="/path/to/creds.json")
        headers = auth.get_headers()

        assert headers == {}

    def test_get_headers_no_auth(self):
        """Test headers when no authentication is provided."""
        auth = GCPVertexAuth()
        headers = auth.get_headers()

        assert headers == {}

    def test_get_config(self):
        """Test configuration dictionary."""
        auth = GCPVertexAuth(
            project_id="test-project",
            location="us-west1",
            credentials_path="/path/to/creds.json",
        )
        config = auth.get_config()

        assert config["project_id"] == "test-project"
        assert config["location"] == "us-west1"
        assert config["auth_type"] == "service_account"
        assert config["credentials_path"] == "/path/to/creds.json"

    def test_get_config_with_api_key(self):
        """Test configuration with API key."""
        auth = GCPVertexAuth(project_id="test-project", api_key="test_key")
        config = auth.get_config()

        assert config["auth_type"] == "api_key"
        assert "credentials_path" not in config

    def test_get_config_missing_project(self):
        """Test configuration without project ID."""
        auth = GCPVertexAuth(location="us-central1")
        config = auth.get_config()

        assert config["project_id"] is None
        assert config["location"] == "us-central1"

    def test_get_auth_type_service_account(self):
        """Test auth type for service account authentication."""
        auth = GCPVertexAuth(credentials_path="/path/to/creds.json")
        assert auth.get_auth_type() == "service_account"

    def test_get_auth_type_api_key(self):
        """Test auth type for API key authentication."""
        auth = GCPVertexAuth(api_key="test_key")
        assert auth.get_auth_type() == "api_key"

    def test_get_auth_type_default(self):
        """Test auth type when no explicit auth is provided."""
        auth = GCPVertexAuth()
        assert auth.get_auth_type() == "service_account"  # Default

    def test_get_credentials_full(self):
        """Test getting full credentials."""
        auth = GCPVertexAuth(
            project_id="test-project",
            location="us-central1",
            credentials_path="/path/to/creds.json",
        )
        creds = auth.get_credentials()

        assert creds == {
            "project_id": "test-project",
            "location": "us-central1",
            "credentials_path": "/path/to/creds.json",
        }

    def test_get_credentials_api_key(self):
        """Test getting credentials with API key."""
        auth = GCPVertexAuth(project_id="test-project", api_key="test_key")
        creds = auth.get_credentials()

        assert creds == {
            "project_id": "test-project",
            "location": "us-central1",
            "api_key": "test_key",
        }

    def test_get_credentials_empty(self):
        """Test getting credentials when nothing is set."""
        auth = GCPVertexAuth()
        creds = auth.get_credentials()

        assert creds is None

    def test_precedence_explicit_over_env(self):
        """Test that explicit values take precedence over env vars."""
        with patch.dict(
            os.environ,
            {
                "GCP_PROJECT_ID": "env-project",
                "GCP_LOCATION": "env-location",
                "GCP_API_KEY": "env_key",
            },
        ):
            auth = GCPVertexAuth(
                project_id="explicit-project",
                location="explicit-location",
                api_key="explicit_key",
            )

            assert auth.project_id == "explicit-project"
            assert auth.location == "explicit-location"
            assert auth.api_key == "explicit_key"

    def test_both_auth_methods(self):
        """Test when both auth methods are provided."""
        auth = GCPVertexAuth(credentials_path="/path/to/creds.json", api_key="test_key")

        # API key should be used in headers
        assert auth.get_headers() == {"x-goog-api-key": "test_key"}
        # But auth type should reflect API key
        assert auth.get_auth_type() == "api_key"
