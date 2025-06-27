"""Base storage interface for multi-cloud support."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator, Optional, Union


class BaseStorage(ABC):
    """Abstract base class for storage providers."""

    @abstractmethod
    def upload_file(
        self, local_path: Union[str, Path], remote_path: str, bucket: str, **kwargs
    ) -> None:
        """Upload a file to the storage provider.

        Args:
            local_path: Local file path
            remote_path: Remote path/key in the bucket
            bucket: Bucket/container name
            **kwargs: Provider-specific options
        """
        pass

    @abstractmethod
    def upload_folder(
        self, local_folder: Union[str, Path], bucket: str, prefix: str = "", **kwargs
    ) -> None:
        """Upload all files in a folder to the storage provider.

        Args:
            local_folder: Local folder path
            bucket: Bucket/container name
            prefix: Optional prefix for all uploaded files
            **kwargs: Provider-specific options
        """
        pass

    @abstractmethod
    def download_file(
        self, remote_path: str, local_path: Union[str, Path], bucket: str, **kwargs
    ) -> None:
        """Download a file from the storage provider.

        Args:
            remote_path: Remote path/key in the bucket
            local_path: Local file path to save to
            bucket: Bucket/container name
            **kwargs: Provider-specific options
        """
        pass

    @abstractmethod
    def list_objects(
        self, bucket: str, prefix: Optional[str] = None, **kwargs
    ) -> Generator[str, None, None]:
        """List objects in a bucket with optional prefix.

        Args:
            bucket: Bucket/container name
            prefix: Optional prefix to filter objects
            **kwargs: Provider-specific options

        Yields:
            Object names/keys
        """
        pass

    @abstractmethod
    def delete_object(self, remote_path: str, bucket: str, **kwargs) -> None:
        """Delete an object from the storage provider.

        Args:
            remote_path: Remote path/key in the bucket
            bucket: Bucket/container name
            **kwargs: Provider-specific options
        """
        pass

    @abstractmethod
    def get_storage_type(self) -> str:
        """Get the storage provider type.

        Returns:
            Storage provider type (e.g., 'aws', 'azure', 'gcp', 'oci', 'github')
        """
        pass
