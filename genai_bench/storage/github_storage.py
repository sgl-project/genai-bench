"""GitHub storage implementation for releases and artifacts."""

from pathlib import Path
from typing import Generator, Optional, Union

from genai_bench.auth.storage_auth_provider import StorageAuthProvider
from genai_bench.logging import init_logger
from genai_bench.storage.base import BaseStorage

logger = init_logger(__name__)


class GitHubStorage(BaseStorage):
    """GitHub storage implementation using releases/artifacts."""

    def __init__(self, auth: StorageAuthProvider, **kwargs):
        """Initialize GitHub storage.

        Args:
            auth: Storage authentication provider
            **kwargs: Additional configuration
        """
        if auth.get_storage_type() != "github":
            raise ValueError("Auth provider must be for GitHub")

        self.auth = auth
        self.config = auth.get_client_config()

        # Extract configuration
        self.token = self.config.get("token")
        self.owner = self.config.get("owner")
        self.repo = self.config.get("repo")

        # Parse owner/repo if provided as single string
        if self.owner and "/" in str(self.owner) and not self.repo:
            self.owner, self.repo = self.owner.split("/", 1)

        if not all([self.token, self.owner, self.repo]):
            raise ValueError("GitHub token, owner, and repo are required")

        # API endpoints
        self.api_base = "https://api.github.com"
        self.upload_base = "https://uploads.github.com"

        # Headers
        self.headers = {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json",
        }

    def _make_request(self, method: str, url: str, **kwargs):
        """Make a request to GitHub API.

        Args:
            method: HTTP method
            url: URL to request
            **kwargs: Additional request arguments

        Returns:
            Response object
        """
        import requests

        # Add headers
        if "headers" in kwargs:
            kwargs["headers"].update(self.headers)
        else:
            kwargs["headers"] = self.headers

        response = requests.request(method, url, **kwargs)
        response.raise_for_status()
        return response

    def _get_or_create_release(self, tag_name: str) -> dict:
        """Get existing release or create a new one.

        Args:
            tag_name: Release tag name (used as bucket)

        Returns:
            Release data dict
        """
        # Try to get existing release
        url = f"{self.api_base}/repos/{self.owner}/{self.repo}/releases/tags/{tag_name}"

        try:
            response = self._make_request("GET", url)
            return response.json()
        except Exception:
            # Create new release
            url = f"{self.api_base}/repos/{self.owner}/{self.repo}/releases"
            data = {
                "tag_name": tag_name,
                "name": f"genai-bench results - {tag_name}",
                "body": "Automated upload from genai-bench",
                "draft": False,
                "prerelease": False,
            }
            response = self._make_request("POST", url, json=data)
            return response.json()

    def upload_file(
        self, local_path: Union[str, Path], remote_path: str, bucket: str, **kwargs
    ) -> None:
        """Upload a file to GitHub release.

        Args:
            local_path: Local file path
            remote_path: Asset name in release
            bucket: Release tag name
            **kwargs: Additional options
        """
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")

        try:
            # Get or create release
            release = self._get_or_create_release(bucket)
            upload_url = release["upload_url"].replace("{?name,label}", "")

            # Check if asset already exists
            existing_assets = release.get("assets", [])
            for asset in existing_assets:
                if asset["name"] == remote_path:
                    # Delete existing asset
                    delete_url = (
                        f"{self.api_base}/repos/{self.owner}/{self.repo}/"
                        f"releases/assets/{asset['id']}"
                    )
                    self._make_request("DELETE", delete_url)
                    logger.info(f"Deleted existing asset: {remote_path}")

            # Upload new asset
            with open(local_path, "rb") as f:
                headers = {"Content-Type": "application/octet-stream"}
                self._make_request(
                    "POST", f"{upload_url}?name={remote_path}", headers=headers, data=f
                )

            logger.info(
                f"Uploaded {local_path} to github://{self.owner}/{self.repo}/releases/{bucket}/{remote_path}"
            )

        except Exception as e:
            logger.error(f"Failed to upload {local_path}: {e}")
            raise

    def upload_folder(
        self, local_folder: Union[str, Path], bucket: str, prefix: str = "", **kwargs
    ) -> None:
        """Upload all files in a folder to GitHub release.

        Args:
            local_folder: Local folder path
            bucket: Release tag name
            prefix: Optional prefix for all uploaded files
            **kwargs: Additional options
        """
        local_folder = Path(local_folder)
        if not local_folder.exists() or not local_folder.is_dir():
            raise ValueError(
                f"Local folder not found or not a directory: {local_folder}"
            )

        # Upload all files in the folder
        for file_path in local_folder.rglob("*"):
            if file_path.is_file():
                # Calculate relative path
                relative_path = file_path.relative_to(local_folder)

                # Construct asset name with prefix
                if prefix:
                    asset_name = f"{prefix}-{relative_path}".replace("/", "-")
                else:
                    asset_name = str(relative_path).replace("/", "-")

                # Upload file
                self.upload_file(file_path, asset_name, bucket, **kwargs)

    def download_file(
        self, remote_path: str, local_path: Union[str, Path], bucket: str, **kwargs
    ) -> None:
        """Download a file from GitHub release.

        Args:
            remote_path: Asset name in release
            local_path: Local file path to save to
            bucket: Release tag name
            **kwargs: Additional options
        """
        local_path = Path(local_path)

        # Create parent directories if needed
        local_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Get release
            url = (
                f"{self.api_base}/repos/{self.owner}/{self.repo}/releases/tags/{bucket}"
            )
            response = self._make_request("GET", url)
            release = response.json()

            # Find asset
            asset_found = None
            for asset in release.get("assets", []):
                if asset["name"] == remote_path:
                    asset_found = asset
                    break

            if not asset_found:
                raise FileNotFoundError(f"Asset not found: {remote_path}")

            # Download asset
            download_url = asset_found["browser_download_url"]
            response = self._make_request("GET", download_url, stream=True)

            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info(
                f"Downloaded github://{self.owner}/{self.repo}/releases/"
                f"{bucket}/{remote_path} to {local_path}"
            )

        except Exception as e:
            logger.error(f"Failed to download {remote_path}: {e}")
            raise

    def list_objects(
        self, bucket: str, prefix: Optional[str] = None, **kwargs
    ) -> Generator[str, None, None]:
        """List assets in a GitHub release.

        Args:
            bucket: Release tag name
            prefix: Optional prefix to filter assets
            **kwargs: Additional options

        Yields:
            Asset names
        """
        try:
            # Get release
            url = (
                f"{self.api_base}/repos/{self.owner}/{self.repo}/releases/tags/{bucket}"
            )
            response = self._make_request("GET", url)
            release = response.json()

            # List assets
            for asset in release.get("assets", []):
                name = asset["name"]
                if not prefix or name.startswith(prefix):
                    yield name

        except Exception as e:
            logger.error(f"Failed to list assets in release {bucket}: {e}")
            raise

    def delete_object(self, remote_path: str, bucket: str, **kwargs) -> None:
        """Delete an asset from GitHub release.

        Args:
            remote_path: Asset name in release
            bucket: Release tag name
            **kwargs: Additional options
        """
        try:
            # Get release
            url = (
                f"{self.api_base}/repos/{self.owner}/{self.repo}/releases/tags/{bucket}"
            )
            response = self._make_request("GET", url)
            release = response.json()

            # Find asset
            asset_found = None
            for asset in release.get("assets", []):
                if asset["name"] == remote_path:
                    asset_found = asset
                    break

            if not asset_found:
                logger.warning(f"Asset not found (already deleted?): {remote_path}")
                return

            # Delete asset
            delete_url = (
                f"{self.api_base}/repos/{self.owner}/{self.repo}/"
                f"releases/assets/{asset_found['id']}"
            )
            self._make_request("DELETE", delete_url)

            logger.info(
                f"Deleted github://{self.owner}/{self.repo}/releases/{bucket}/{remote_path}"
            )

        except Exception as e:
            logger.error(f"Failed to delete {remote_path}: {e}")
            raise

    def get_storage_type(self) -> str:
        """Get the storage provider type.

        Returns:
            'github'
        """
        return "github"
