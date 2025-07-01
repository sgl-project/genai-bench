"""GCP Vertex AI authentication provider."""

import os
from typing import Any, Dict, Optional

from genai_bench.auth.model_auth_provider import ModelAuthProvider


class GCPVertexAuth(ModelAuthProvider):
    """GCP Vertex AI authentication provider."""

    def __init__(
        self,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        credentials_path: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """Initialize GCP Vertex AI authentication.

        Args:
            project_id: GCP project ID
            location: GCP region/location for Vertex AI
            credentials_path: Path to service account JSON file
            api_key: API key for authentication (if using API key auth)
        """
        self.project_id = project_id or os.getenv("GCP_PROJECT_ID")
        self.location = location or os.getenv("GCP_LOCATION") or "us-central1"
        self.credentials_path = credentials_path or os.getenv(
            "GOOGLE_APPLICATION_CREDENTIALS"
        )
        self.api_key = api_key or os.getenv("GCP_API_KEY")

    def get_headers(self) -> Dict[str, str]:
        """Get authentication headers for Vertex AI API requests.

        Returns:
            Dict[str, str]: Headers, empty for service account auth
        """
        headers = {}

        if self.api_key:
            headers["x-goog-api-key"] = self.api_key

        return headers

    def get_config(self) -> Dict[str, Any]:
        """Get GCP Vertex AI configuration.

        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        config = {
            "project_id": self.project_id,
            "location": self.location,
            "auth_type": self.get_auth_type(),
        }

        if self.credentials_path:
            config["credentials_path"] = self.credentials_path

        return config

    def get_auth_type(self) -> str:
        """Get the authentication type identifier.

        Returns:
            'api_key' if using API key, otherwise 'service_account'
        """
        return "api_key" if self.api_key else "service_account"

    def get_credentials(self) -> Optional[Dict[str, Any]]:
        """Get GCP credentials.

        Returns:
            Dict with credentials or None
        """
        # Only return credentials if we have actual auth info
        if not self.project_id and not self.credentials_path and not self.api_key:
            return None

        creds = {}

        if self.project_id:
            creds["project_id"] = self.project_id

        if self.location:
            creds["location"] = self.location

        if self.credentials_path:
            creds["credentials_path"] = self.credentials_path
        elif self.api_key:
            creds["api_key"] = self.api_key

        return creds
