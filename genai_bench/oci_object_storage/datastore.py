"""Interface definitions for data store operations."""

from abc import ABC, abstractmethod
from typing import Optional

from genai_bench.oci_object_storage.object_uri import ObjectURI


class DataStore(ABC):
    """Interface for data store operations."""

    @abstractmethod
    def download(
        self, source: ObjectURI, target: str, retries: Optional[int] = 3
    ) -> None:
        """Download an object from its source path to the target path.

        Args:
            source: Source object URI
            target: Target local path
            retries: Number of retry attempts

        Raises:
            ServiceError: If object retrieval fails
            OSError: If file operations fail
        """
        pass

    @abstractmethod
    def upload(
        self, source: str, target: ObjectURI, retries: Optional[int] = 3
    ) -> None:
        """Upload an object from its source path to the target path.

        Args:
            source: Source local path
            target: Target object URI
            retries: Number of retry attempts

        Raises:
            ServiceError: If upload fails
            OSError: If file operations fail
        """
        pass
