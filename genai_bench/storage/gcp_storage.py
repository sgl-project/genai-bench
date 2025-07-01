"""GCP Cloud Storage implementation."""

import os
from pathlib import Path
from typing import Generator, Optional, Union

from genai_bench.auth.storage_auth_provider import StorageAuthProvider
from genai_bench.logging import init_logger
from genai_bench.storage.base import BaseStorage

logger = init_logger(__name__)


class GCPCloudStorage(BaseStorage):
    """GCP Cloud Storage implementation."""

    def __init__(self, auth: StorageAuthProvider, **kwargs):
        """Initialize GCP Cloud Storage.

        Args:
            auth: Storage authentication provider
            **kwargs: Additional configuration
        """
        if auth.get_storage_type() != "gcp":
            raise ValueError("Auth provider must be for GCP")

        self.auth = auth
        self.config = auth.get_client_config()

        # Lazy import to avoid dependency if not using GCP
        try:
            from google.api_core import exceptions
            from google.cloud import storage
        except ImportError as e:
            raise ImportError(
                "google-cloud-storage is required for GCP Cloud Storage. "
                "Install it with: pip install google-cloud-storage"
            ) from e

        self.storage = storage
        self.exceptions = exceptions

        # Create storage client
        self.client = self._create_client()

    def _create_client(self):
        """Create GCS client with appropriate credentials."""
        config = self.config

        if "credentials_path" in config:
            # Use service account credentials
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = config["credentials_path"]
            return self.storage.Client(project=config.get("project"))
        elif "access_token" in config:
            # Use access token
            try:
                from google.oauth2.credentials import Credentials
            except ImportError as e:
                raise ImportError(
                    "google-auth is required for token authentication. "
                    "Install it with: pip install google-auth"
                ) from e

            credentials = Credentials(token=config["access_token"])
            return self.storage.Client(
                project=config.get("project"), credentials=credentials
            )
        else:
            # Use default credentials
            return self.storage.Client(project=config.get("project"))

    def upload_file(
        self, local_path: Union[str, Path], remote_path: str, bucket: str, **kwargs
    ) -> None:
        """Upload a file to GCS.

        Args:
            local_path: Local file path
            remote_path: GCS object name
            bucket: GCS bucket name
            **kwargs: Additional options (e.g., content_type, metadata)
        """
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")

        try:
            # Get bucket
            bucket_obj = self.client.bucket(bucket)

            # Create blob
            blob = bucket_obj.blob(remote_path)

            # Set optional metadata
            if "content_type" in kwargs:
                blob.content_type = kwargs["content_type"]
            if "metadata" in kwargs:
                blob.metadata = kwargs["metadata"]

            # Upload file
            blob.upload_from_filename(
                str(local_path), timeout=kwargs.get("timeout", 300)
            )

            logger.info(f"Uploaded {local_path} to gs://{bucket}/{remote_path}")

        except Exception as e:
            logger.error(f"Failed to upload {local_path}: {e}")
            raise

    def upload_folder(
        self, local_folder: Union[str, Path], bucket: str, prefix: str = "", **kwargs
    ) -> None:
        """Upload all files in a folder to GCS.

        Args:
            local_folder: Local folder path
            bucket: GCS bucket name
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

                # Construct GCS object name with prefix
                if prefix:
                    object_name = f"{prefix}/{relative_path}"
                else:
                    object_name = str(relative_path)

                # Upload file
                self.upload_file(file_path, object_name, bucket, **kwargs)

    def download_file(
        self, remote_path: str, local_path: Union[str, Path], bucket: str, **kwargs
    ) -> None:
        """Download a file from GCS.

        Args:
            remote_path: GCS object name
            local_path: Local file path to save to
            bucket: GCS bucket name
            **kwargs: Additional options
        """
        local_path = Path(local_path)

        # Create parent directories if needed
        local_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Get bucket
            bucket_obj = self.client.bucket(bucket)

            # Get blob
            blob = bucket_obj.blob(remote_path)

            # Download file
            blob.download_to_filename(
                str(local_path), timeout=kwargs.get("timeout", 300)
            )

            logger.info(f"Downloaded gs://{bucket}/{remote_path} to {local_path}")

        except self.exceptions.NotFound as e:
            raise FileNotFoundError(
                f"Object not found: gs://{bucket}/{remote_path}"
            ) from e
        except Exception as e:
            logger.error(f"Failed to download {remote_path}: {e}")
            raise

    def list_objects(
        self, bucket: str, prefix: Optional[str] = None, **kwargs
    ) -> Generator[str, None, None]:
        """List objects in a GCS bucket with optional prefix.

        Args:
            bucket: GCS bucket name
            prefix: Optional prefix to filter objects
            **kwargs: Additional options

        Yields:
            Object names
        """
        try:
            # Get bucket
            bucket_obj = self.client.bucket(bucket)

            # List blobs
            blobs = bucket_obj.list_blobs(prefix=prefix)

            for blob in blobs:
                yield blob.name

        except Exception as e:
            logger.error(f"Failed to list objects in {bucket}: {e}")
            raise

    def delete_object(self, remote_path: str, bucket: str, **kwargs) -> None:
        """Delete an object from GCS.

        Args:
            remote_path: GCS object name
            bucket: GCS bucket name
            **kwargs: Additional options
        """
        try:
            # Get bucket
            bucket_obj = self.client.bucket(bucket)

            # Get blob
            blob = bucket_obj.blob(remote_path)

            # Delete blob
            blob.delete()

            logger.info(f"Deleted gs://{bucket}/{remote_path}")

        except self.exceptions.NotFound:
            logger.warning(
                f"Object not found (already deleted?): gs://{bucket}/{remote_path}"
            )
        except Exception as e:
            logger.error(f"Failed to delete {remote_path}: {e}")
            raise

    def get_storage_type(self) -> str:
        """Get the storage provider type.

        Returns:
            'gcp'
        """
        return "gcp"
