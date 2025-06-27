"""Storage authentication provider interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class StorageAuthProvider(ABC):
    """Base class for storage authentication providers."""

    @abstractmethod
    def get_client_config(self) -> Dict[str, Any]:
        """Get storage service-specific client configuration.

        Returns:
            Dict[str, Any]: Configuration for storage client initialization
        """
        pass

    @abstractmethod
    def get_credentials(self) -> Any:
        """Get authentication credentials for storage operations.

        Returns:
            Any: Authentication credentials (e.g., access keys, signers)
        """
        pass

    @abstractmethod
    def get_storage_type(self) -> str:
        """Get the storage provider type.

        Returns:
            str: Storage type (e.g., 'oci', 'aws', 'azure', 'gcp', 'github')
        """
        pass

    def get_region(self) -> Optional[str]:
        """Get the region for storage operations.

        Returns:
            Optional[str]: Region or None if not applicable
        """
        return None
