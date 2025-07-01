"""OCI Object Storage implementation using existing OSDataStore."""

from pathlib import Path
from typing import Generator, Optional, Union

from genai_bench.auth.storage_auth_provider import StorageAuthProvider
from genai_bench.logging import init_logger
from genai_bench.storage.base import BaseStorage
from genai_bench.storage.oci_object_storage.object_uri import ObjectURI
from genai_bench.storage.oci_object_storage.os_datastore import OSDataStore

logger = init_logger(__name__)


class OCIObjectStorage(BaseStorage):
    """OCI Object Storage implementation wrapping existing OSDataStore."""

    def __init__(self, auth: StorageAuthProvider, **kwargs):
        """Initialize OCI Object Storage.

        Args:
            auth: Storage authentication provider
            **kwargs: Additional configuration
        """
        if auth.get_storage_type() != "oci":
            raise ValueError("Auth provider must be for OCI")

        self.auth = auth
        # Get the OCI auth provider from the adapter
        oci_auth = auth.get_credentials()
        self.datastore = OSDataStore(oci_auth)

        # Get namespace if provided
        self.namespace = kwargs.get("namespace")
        if not self.namespace:
            # Get namespace from service
            self.namespace = self.datastore.get_namespace()

    def upload_file(
        self, local_path: Union[str, Path], remote_path: str, bucket: str, **kwargs
    ) -> None:
        """Upload a file to OCI Object Storage.

        Args:
            local_path: Local file path
            remote_path: Remote path/key in the bucket
            bucket: Bucket name
            **kwargs: Additional options
        """
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")

        # Create ObjectURI
        target_uri = ObjectURI(
            namespace=self.namespace or "",
            bucket_name=bucket,
            object_name=remote_path,
            region=self.auth.get_region() or None,
            prefix=None,
        )

        # Upload using datastore
        self.datastore.upload(str(local_path), target_uri)
        logger.info(f"Uploaded {local_path} to oci://{bucket}/{remote_path}")

    def upload_folder(
        self, local_folder: Union[str, Path], bucket: str, prefix: str = "", **kwargs
    ) -> None:
        """Upload all files in a folder to OCI Object Storage.

        Args:
            local_folder: Local folder path
            bucket: Bucket name
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

                # Construct remote path with prefix
                if prefix:
                    remote_path = f"{prefix}/{relative_path}"
                else:
                    remote_path = str(relative_path)

                # Upload file
                self.upload_file(file_path, remote_path, bucket, **kwargs)

    def download_file(
        self, remote_path: str, local_path: Union[str, Path], bucket: str, **kwargs
    ) -> None:
        """Download a file from OCI Object Storage.

        Args:
            remote_path: Remote path/key in the bucket
            local_path: Local file path to save to
            bucket: Bucket name
            **kwargs: Additional options
        """
        local_path = Path(local_path)

        # Create parent directories if needed
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Create ObjectURI
        source_uri = ObjectURI(
            namespace=self.namespace or "",
            bucket_name=bucket,
            object_name=remote_path,
            region=self.auth.get_region() or None,
            prefix=None,
        )

        # Download using datastore
        self.datastore.download(source_uri, str(local_path))
        logger.info(f"Downloaded oci://{bucket}/{remote_path} to {local_path}")

    def list_objects(
        self, bucket: str, prefix: Optional[str] = None, **kwargs
    ) -> Generator[str, None, None]:
        """List objects in a bucket with optional prefix.

        Args:
            bucket: Bucket name
            prefix: Optional prefix to filter objects
            **kwargs: Additional options

        Yields:
            Object names/keys
        """
        # Create ObjectURI for listing
        list_uri = ObjectURI(
            namespace=self.namespace or "",
            bucket_name=bucket,
            object_name=prefix or "",
            region=self.auth.get_region() or None,
            prefix=prefix,
        )

        # List objects using datastore
        for object_name in self.datastore.list_objects(list_uri):
            yield object_name

    def delete_object(self, remote_path: str, bucket: str, **kwargs) -> None:
        """Delete an object from OCI Object Storage.

        Args:
            remote_path: Remote path/key in the bucket
            bucket: Bucket name
            **kwargs: Additional options
        """
        # Create ObjectURI
        target_uri = ObjectURI(
            namespace=self.namespace or "",
            bucket_name=bucket,
            object_name=remote_path,
            region=self.auth.get_region() or None,
            prefix=None,
        )

        # Delete using datastore
        self.datastore.delete_object(target_uri)
        logger.info(f"Deleted oci://{bucket}/{remote_path}")

    def get_storage_type(self) -> str:
        """Get the storage provider type.

        Returns:
            'oci'
        """
        return "oci"
