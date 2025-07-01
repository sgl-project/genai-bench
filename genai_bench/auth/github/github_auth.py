"""GitHub storage authentication provider."""

import os
from typing import Any, Dict, Optional

from genai_bench.auth.storage_auth_provider import StorageAuthProvider


class GitHubAuth(StorageAuthProvider):
    """GitHub storage authentication provider for releases/artifacts."""

    def __init__(
        self,
        token: Optional[str] = None,
        owner: Optional[str] = None,
        repo: Optional[str] = None,
    ):
        """Initialize GitHub authentication.

        Args:
            token: GitHub personal access token or GITHUB_TOKEN
            owner: Repository owner (user or organization)
            repo: Repository name
        """
        self.token = token or os.getenv("GITHUB_TOKEN")
        self.owner = owner or os.getenv("GITHUB_OWNER")
        self.repo = repo or os.getenv("GITHUB_REPO")

    def get_client_config(self) -> Dict[str, Any]:
        """Get GitHub client configuration.

        Returns:
            Dict[str, Any]: Configuration for GitHub client
        """
        config = {}

        if self.owner:
            config["owner"] = self.owner

        if self.repo:
            config["repo"] = self.repo

        if self.token:
            config["token"] = self.token

        return config

    def get_credentials(self) -> Dict[str, Any]:
        """Get GitHub credentials for storage operations.

        Returns:
            Dict with GitHub credentials
        """
        creds = {}

        if self.token:
            creds["token"] = self.token

        if self.owner:
            creds["owner"] = self.owner

        if self.repo:
            creds["repo"] = self.repo

        return creds

    def get_storage_type(self) -> str:
        """Get the storage provider type.

        Returns:
            'github'
        """
        return "github"
