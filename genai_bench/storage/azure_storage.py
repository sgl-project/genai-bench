"""Azure Blob Storage implementation."""

from pathlib import Path
from typing import Generator, Optional, Union

from genai_bench.auth.storage_auth_provider import StorageAuthProvider
from genai_bench.logging import init_logger
from genai_bench.storage.base import BaseStorage

logger = init_logger(__name__)


class AzureBlobStorage(BaseStorage):
    """Azure Blob Storage implementation."""

    def __init__(self, auth: StorageAuthProvider, **kwargs):
        """Initialize Azure Blob Storage.

        Args:
            auth: Storage authentication provider
            **kwargs: Additional configuration
        """
        if auth.get_storage_type() != "azure":
            raise ValueError("Auth provider must be for Azure")

        self.auth = auth
        self.config = auth.get_client_config()

        # Lazy import to avoid dependency if not using Azure
        try:
            from azure.core.exceptions import ResourceNotFoundError
            from azure.storage.blob import BlobServiceClient
        except ImportError as e:
            raise ImportError(
                "azure-storage-blob is required for Azure Blob Storage. "
                "Install it with: pip install azure-storage-blob"
            ) from e

        self.BlobServiceClient = BlobServiceClient
        self.ResourceNotFoundError = ResourceNotFoundError

        # Create blob service client
        self.client = self._create_client()

    def _create_client(self):
        """Create Azure Blob service client with appropriate credentials."""
        config = self.config

        if "connection_string" in config:
            # Use connection string
            return self.BlobServiceClient.from_connection_string(
                config["connection_string"]
            )
        elif "account_name" in config:
            # Build account URL
            account_url = f"https://{config['account_name']}.blob.core.windows.net"

            if "account_key" in config:
                # Use account key
                return self.BlobServiceClient(
                    account_url=account_url, credential=config["account_key"]
                )
            elif "sas_token" in config:
                # Use SAS token
                return self.BlobServiceClient(
                    account_url=account_url, credential=config["sas_token"]
                )
            elif config.get("use_azure_ad"):
                # Use Azure AD authentication
                try:
                    from azure.identity import (
                        ClientSecretCredential,
                        DefaultAzureCredential,
                    )
                except ImportError as e:
                    raise ImportError(
                        "azure-identity is required for Azure AD authentication. "
                        "Install it with: pip install azure-identity"
                    ) from e

                if all(
                    k in config for k in ["tenant_id", "client_id", "client_secret"]
                ):
                    # Use service principal
                    credential = ClientSecretCredential(
                        tenant_id=config["tenant_id"],
                        client_id=config["client_id"],
                        client_secret=config["client_secret"],
                    )
                else:
                    # Use default credential chain
                    credential = DefaultAzureCredential()

                return self.BlobServiceClient(
                    account_url=account_url, credential=credential
                )
            else:
                raise ValueError("No valid Azure credentials provided")
        else:
            raise ValueError("Azure account name or connection string required")

    def upload_file(
        self, local_path: Union[str, Path], remote_path: str, bucket: str, **kwargs
    ) -> None:
        """Upload a file to Azure Blob Storage.

        Args:
            local_path: Local file path
            remote_path: Blob name
            bucket: Container name
            **kwargs: Additional options (e.g., content_settings, metadata)
        """
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")

        try:
            # Get container client
            container_client = self.client.get_container_client(bucket)

            # Get blob client
            blob_client = container_client.get_blob_client(remote_path)

            # Upload file
            with open(local_path, "rb") as data:
                blob_client.upload_blob(
                    data,
                    overwrite=kwargs.get("overwrite", True),
                    content_settings=kwargs.get("content_settings"),
                    metadata=kwargs.get("metadata"),
                )

            logger.info(f"Uploaded {local_path} to azure://{bucket}/{remote_path}")

        except Exception as e:
            logger.error(f"Failed to upload {local_path}: {e}")
            raise

    def upload_folder(
        self, local_folder: Union[str, Path], bucket: str, prefix: str = "", **kwargs
    ) -> None:
        """Upload all files in a folder to Azure Blob Storage.

        Args:
            local_folder: Local folder path
            bucket: Container name
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

                # Construct blob name with prefix
                if prefix:
                    blob_name = f"{prefix}/{relative_path}"
                else:
                    blob_name = str(relative_path)

                # Upload file
                self.upload_file(file_path, blob_name, bucket, **kwargs)

    def download_file(
        self, remote_path: str, local_path: Union[str, Path], bucket: str, **kwargs
    ) -> None:
        """Download a file from Azure Blob Storage.

        Args:
            remote_path: Blob name
            local_path: Local file path to save to
            bucket: Container name
            **kwargs: Additional options
        """
        local_path = Path(local_path)

        # Create parent directories if needed
        local_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Get container client
            container_client = self.client.get_container_client(bucket)

            # Download blob
            blob_client = container_client.get_blob_client(remote_path)

            with open(local_path, "wb") as file:
                download_stream = blob_client.download_blob()
                file.write(download_stream.readall())

            logger.info(f"Downloaded azure://{bucket}/{remote_path} to {local_path}")

        except self.ResourceNotFoundError as e:
            raise FileNotFoundError(f"Blob not found: {bucket}/{remote_path}") from e
        except Exception as e:
            logger.error(f"Failed to download {remote_path}: {e}")
            raise

    def list_objects(
        self, bucket: str, prefix: Optional[str] = None, **kwargs
    ) -> Generator[str, None, None]:
        """List objects in an Azure container with optional prefix.

        Args:
            bucket: Container name
            prefix: Optional prefix to filter objects
            **kwargs: Additional options

        Yields:
            Blob names
        """
        try:
            # Get container client
            container_client = self.client.get_container_client(bucket)

            # List blobs
            list_kwargs = {}
            if prefix:
                list_kwargs["name_starts_with"] = prefix

            for blob in container_client.list_blobs(**list_kwargs):
                yield blob.name

        except Exception as e:
            logger.error(f"Failed to list objects in {bucket}: {e}")
            raise

    def delete_object(self, remote_path: str, bucket: str, **kwargs) -> None:
        """Delete an object from Azure Blob Storage.

        Args:
            remote_path: Blob name
            bucket: Container name
            **kwargs: Additional options
        """
        try:
            # Get container client
            container_client = self.client.get_container_client(bucket)

            # Delete blob
            blob_client = container_client.get_blob_client(remote_path)
            blob_client.delete_blob()

            logger.info(f"Deleted azure://{bucket}/{remote_path}")

        except self.ResourceNotFoundError:
            logger.warning(f"Blob not found (already deleted?): {bucket}/{remote_path}")
        except Exception as e:
            logger.error(f"Failed to delete {remote_path}: {e}")
            raise

    def get_storage_type(self) -> str:
        """Get the storage provider type.

        Returns:
            'azure'
        """
        return "azure"
