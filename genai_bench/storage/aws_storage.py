"""AWS S3 storage implementation."""

from pathlib import Path
from typing import Generator, Optional, Union

from genai_bench.auth.storage_auth_provider import StorageAuthProvider
from genai_bench.logging import init_logger
from genai_bench.storage.base import BaseStorage

logger = init_logger(__name__)


class AWSS3Storage(BaseStorage):
    """AWS S3 storage implementation."""

    def __init__(self, auth: StorageAuthProvider, **kwargs):
        """Initialize AWS S3 storage.

        Args:
            auth: Storage authentication provider
            **kwargs: Additional configuration
        """
        if auth.get_storage_type() != "aws":
            raise ValueError("Auth provider must be for AWS")

        self.auth = auth
        self.config = auth.get_client_config()

        # Lazy import to avoid dependency if not using AWS
        try:
            import boto3
            from botocore.exceptions import NoCredentialsError
        except ImportError as e:
            raise ImportError(
                "boto3 is required for AWS S3 storage. "
                "Install it with: pip install boto3"
            ) from e

        self.boto3 = boto3
        self.NoCredentialsError = NoCredentialsError

        # Create S3 client
        self.client = self._create_client()

    def _create_client(self):
        """Create S3 client with appropriate credentials."""
        client_kwargs = {}

        # Add region if provided
        if "region_name" in self.config:
            client_kwargs["region_name"] = self.config["region_name"]

        # Add profile if provided
        if "profile_name" in self.config:
            # Use session with profile
            session = self.boto3.Session(profile_name=self.config["profile_name"])
            return session.client("s3", **client_kwargs)
        else:
            # Add explicit credentials if provided
            if "aws_access_key_id" in self.config:
                client_kwargs["aws_access_key_id"] = self.config["aws_access_key_id"]
            if "aws_secret_access_key" in self.config:
                client_kwargs["aws_secret_access_key"] = self.config[
                    "aws_secret_access_key"
                ]
            if "aws_session_token" in self.config:
                client_kwargs["aws_session_token"] = self.config["aws_session_token"]

            return self.boto3.client("s3", **client_kwargs)

    def upload_file(
        self, local_path: Union[str, Path], remote_path: str, bucket: str, **kwargs
    ) -> None:
        """Upload a file to S3.

        Args:
            local_path: Local file path
            remote_path: S3 key
            bucket: S3 bucket name
            **kwargs: Additional options (e.g., ExtraArgs)
        """
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")

        try:
            # Check if multipart upload is needed (files > 100MB)
            file_size = local_path.stat().st_size
            if file_size > 100 * 1024 * 1024:  # 100MB
                # Use multipart upload
                logger.info(
                    f"Using multipart upload for large file "
                    f"({file_size / 1024 / 1024:.2f} MB)"
                )
                self._multipart_upload(local_path, remote_path, bucket, **kwargs)
            else:
                # Regular upload
                extra_args = kwargs.get("ExtraArgs", {})
                self.client.upload_file(
                    str(local_path), bucket, remote_path, ExtraArgs=extra_args
                )

            logger.info(f"Uploaded {local_path} to s3://{bucket}/{remote_path}")

        except self.NoCredentialsError as e:
            raise ValueError("AWS credentials not found") from e
        except Exception as e:
            logger.error(f"Failed to upload {local_path}: {e}")
            raise

    def _multipart_upload(
        self, local_path: Path, remote_path: str, bucket: str, **kwargs
    ) -> None:
        """Perform multipart upload for large files."""
        from boto3.s3.transfer import TransferConfig

        # Configure multipart upload
        config = TransferConfig(
            multipart_threshold=1024 * 25,  # 25MB
            max_concurrency=10,
            multipart_chunksize=1024 * 25,
            use_threads=True,
        )

        extra_args = kwargs.get("ExtraArgs", {})
        self.client.upload_file(
            str(local_path), bucket, remote_path, ExtraArgs=extra_args, Config=config
        )

    def upload_folder(
        self, local_folder: Union[str, Path], bucket: str, prefix: str = "", **kwargs
    ) -> None:
        """Upload all files in a folder to S3.

        Args:
            local_folder: Local folder path
            bucket: S3 bucket name
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

                # Construct S3 key with prefix
                s3_key = f"{prefix}/{relative_path}" if prefix else str(relative_path)

                # Upload file
                self.upload_file(file_path, s3_key, bucket, **kwargs)

    def download_file(
        self, remote_path: str, local_path: Union[str, Path], bucket: str, **kwargs
    ) -> None:
        """Download a file from S3.

        Args:
            remote_path: S3 key
            local_path: Local file path to save to
            bucket: S3 bucket name
            **kwargs: Additional options
        """
        local_path = Path(local_path)

        # Create parent directories if needed
        local_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self.client.download_file(bucket, remote_path, str(local_path))
            logger.info(f"Downloaded s3://{bucket}/{remote_path} to {local_path}")
        except self.NoCredentialsError as e:
            raise ValueError("AWS credentials not found") from e
        except Exception as e:
            logger.error(f"Failed to download {remote_path}: {e}")
            raise

    def list_objects(
        self, bucket: str, prefix: Optional[str] = None, **kwargs
    ) -> Generator[str, None, None]:
        """List objects in an S3 bucket with optional prefix.

        Args:
            bucket: S3 bucket name
            prefix: Optional prefix to filter objects
            **kwargs: Additional options

        Yields:
            Object keys
        """
        try:
            paginator = self.client.get_paginator("list_objects_v2")

            page_kwargs = {"Bucket": bucket}
            if prefix:
                page_kwargs["Prefix"] = prefix

            for page in paginator.paginate(**page_kwargs):
                if "Contents" in page:
                    for obj in page["Contents"]:
                        yield obj["Key"]

        except self.NoCredentialsError as e:
            raise ValueError("AWS credentials not found") from e
        except Exception as e:
            logger.error(f"Failed to list objects in {bucket}: {e}")
            raise

    def delete_object(self, remote_path: str, bucket: str, **kwargs) -> None:
        """Delete an object from S3.

        Args:
            remote_path: S3 key
            bucket: S3 bucket name
            **kwargs: Additional options
        """
        try:
            self.client.delete_object(Bucket=bucket, Key=remote_path)
            logger.info(f"Deleted s3://{bucket}/{remote_path}")
        except self.NoCredentialsError as e:
            raise ValueError("AWS credentials not found") from e
        except Exception as e:
            logger.error(f"Failed to delete {remote_path}: {e}")
            raise

    def get_storage_type(self) -> str:
        """Get the storage provider type.

        Returns:
            'aws'
        """
        return "aws"
