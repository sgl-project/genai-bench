"""OCI Object Storage client implementation."""

import os
from pathlib import Path
from typing import Any, BinaryIO, Generator, Optional, Union

from oci.object_storage import ObjectStorageClient, UploadManager

from genai_bench.auth.auth_provider import AuthProvider
from genai_bench.logging import init_logger
from genai_bench.oci_object_storage.datastore import DataStore
from genai_bench.oci_object_storage.object_uri import ObjectURI

MB = 1024 * 1024  # 1 MB in bytes
logger = init_logger(__name__)


class OSDataStore(DataStore):
    """Client for interacting with OCI Object Storage."""

    def __init__(self, auth: AuthProvider):
        """Initialize Casper client.

        Args:
            auth: Authentication provider
        """
        self.auth = auth
        self.config = auth.get_config()
        self.client = ObjectStorageClient(
            config=self.config, signer=auth.get_auth_credentials()
        )

    def set_region(self, region: str) -> None:
        """Set the region for the client.

        Args:
            region: OCI region
        """
        logger.info(f"Setting region to {region}")
        self.config["region"] = region
        self.client.base_client.set_region(self.config["region"])

    def download(
        self, source: ObjectURI, target: str, retries: Optional[int] = 3
    ) -> None:
        """Download an object from OCI Object Storage.

        Args:
            source: Source object URI
            target: Target local path
            retries: Number of retry attempts
        """
        logger.info(f"Downloading {source} to {target}")

        if not source.namespace:
            namespace = self.get_namespace()
            source.namespace = namespace
            logger.debug(f"Using namespace: {namespace}")

        response = self.get_object(source)
        with open(target, "wb") as f:
            for chunk in response.data.raw.stream(1024 * 1024, decode_content=False):
                f.write(chunk)
        logger.info(f"Successfully downloaded {source} to {target}")

    def get_object(self, source: ObjectURI) -> Any:
        """Get an object from OCI Object Storage.

        Args:
            source: Source object URI

        Returns:
            Object response
        """
        return self.client.get_object(
            namespace_name=source.namespace,
            bucket_name=source.bucket_name,
            object_name=source.object_name,
        )

    def put_object(self, target: ObjectURI, data: BinaryIO) -> None:
        """Put an object to OCI Object Storage.

        Args:
            target: Target object URI
            data: Object data
        """
        self.client.put_object(
            namespace_name=target.namespace,
            bucket_name=target.bucket_name,
            object_name=target.object_name,
            put_object_body=data,
        )

    def list_objects(self, uri: ObjectURI) -> Generator[str, None, None]:
        """List objects in a bucket with optional prefix.

        Args:
            uri: Storage URI containing namespace and bucket

        Yields:
            Object names
        """
        logger.info(f"Listing objects in {uri}")
        if not uri.namespace:
            namespace = self.get_namespace()
            uri.namespace = namespace
            logger.debug(f"Using namespace: {namespace}")

        kwargs = {
            "namespace_name": uri.namespace,
            "bucket_name": uri.bucket_name,
        }

        if uri.prefix:
            kwargs["prefix"] = uri.prefix
            logger.debug(f"Using prefix filter: {uri.prefix}")

        response = self.client.list_objects(**kwargs)
        for obj in response.data.objects:
            yield obj.name

    def get_namespace(self) -> str:
        """Get the namespace for the current compartment.

        Returns:
            Namespace string
        """
        response = self.client.get_namespace()
        return response.data

    def upload(
        self, source: str, target: ObjectURI, retries: Optional[int] = 3
    ) -> None:
        """Upload an object to OCI Object Storage.

        Args:
            source: Source local path
            target: Target object URI
            retries: Number of retry attempts
        """
        logger.info(f"Uploading {source} to {target}")

        if not target.namespace:
            namespace = self.get_namespace()
            target.namespace = namespace
            logger.debug(f"Using namespace: {namespace}")

        file_size = os.path.getsize(source)
        if file_size > 128 * MB:  # Use multipart upload for files larger than 128MB
            logger.info(
                f"Using multipart upload for {source} ({file_size / MB:.2f} MB)"
            )
            self.multipart_upload(source, target)
        else:
            logger.info(
                f"Using single-part upload for {source} ({file_size / MB:.2f} MB)"
            )
            with open(source, "rb") as f:
                self.put_object(target, f)
        logger.info(f"Successfully uploaded {source} to {target}")

    def multipart_upload(
        self,
        source: str,
        target: ObjectURI,
        part_size: int = 128 * MB,
        max_workers: int = 3,
    ) -> None:
        """Upload a large file using multipart upload.

        Args:
            source: Source local path
            target: Target object URI
            part_size: Size of each part in bytes
            max_workers: Maximum number of concurrent upload threads
        """
        logger.info(f"Starting multipart upload for {source}")
        upload_manager = UploadManager(self.client, allow_multipart_uploads=True)

        kwargs = {
            "part_size": part_size,
            "parallel_process_count": max_workers,
        }

        response = upload_manager.upload_file(
            namespace_name=target.namespace,
            bucket_name=target.bucket_name,
            object_name=target.object_name,
            file_path=source,
            **kwargs,
        )

        if response.status != 200:
            raise Exception(f"Multipart upload failed: {response.data.error_message}")
        logger.info(f"Successfully completed multipart upload for {source}")

    def upload_folder(
        self,
        folder_path: Union[str, Path],
        bucket: str,
        namespace: str,
        prefix: str = "",
    ) -> None:
        """Upload all files in a folder to object storage.

        Args:
            folder_path: Path to the folder to upload
            bucket: Name of the bucket to upload to
            namespace: Object Storage namespace
            prefix: Optional prefix to add to object names (default: "")
        """
        folder_path = Path(folder_path)
        if not folder_path.is_dir():
            raise ValueError(f"Path {folder_path} is not a directory")

        # Upload all files in the folder
        for file_path in folder_path.glob("**/*"):
            if file_path.is_file():
                # Create object name with prefix
                rel_path = file_path.relative_to(folder_path)
                object_name = str(Path(prefix) / rel_path) if prefix else str(rel_path)

                target = ObjectURI(
                    namespace=namespace,
                    bucket_name=bucket,
                    object_name=object_name,
                    prefix=prefix,
                    region=self.config["region"],
                )

                logger.info(f"Uploading {file_path} to {bucket}/{object_name}")
                self.upload(str(file_path), target)
