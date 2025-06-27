"""GCP Cloud Storage authentication provider."""

import os
from typing import Any, Dict, Optional

from genai_bench.auth.storage_auth_provider import StorageAuthProvider


class GCPStorageAuth(StorageAuthProvider):
    """GCP Cloud Storage authentication provider."""

    def __init__(
        self,
        project_id: Optional[str] = None,
        credentials_path: Optional[str] = None,
        access_token: Optional[str] = None,
    ):
        """Initialize GCP Cloud Storage authentication.

        Args:
            project_id: GCP project ID
            credentials_path: Path to service account JSON file
            access_token: OAuth2 access token for authentication
        """
        self.project_id = project_id or os.getenv("GCP_PROJECT_ID")
        self.credentials_path = credentials_path or os.getenv(
            "GOOGLE_APPLICATION_CREDENTIALS"
        )
        self.access_token = access_token or os.getenv("GCP_ACCESS_TOKEN")

    def get_client_config(self) -> Dict[str, Any]:
        """Get GCS client configuration.

        Returns:
            Dict[str, Any]: Configuration for storage client
        """
        config = {}

        if self.project_id:
            config["project"] = self.project_id

        if self.credentials_path:
            config["credentials_path"] = self.credentials_path
        elif self.access_token:
            config["access_token"] = self.access_token

        return config

    def get_credentials(self) -> Dict[str, Any]:
        """Get GCP credentials for storage operations.

        Returns:
            Dict with GCP credentials
        """
        creds = {}

        if self.project_id:
            creds["project_id"] = self.project_id

        if self.credentials_path:
            creds["credentials_path"] = self.credentials_path
        elif self.access_token:
            creds["access_token"] = self.access_token

        return creds

    def get_storage_type(self) -> str:
        """Get the storage provider type.

        Returns:
            'gcp'
        """
        return "gcp"
