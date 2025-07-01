"""Model endpoint authentication provider interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class ModelAuthProvider(ABC):
    """Base class for model endpoint authentication providers."""

    @abstractmethod
    def get_headers(self) -> Dict[str, str]:
        """Get authentication headers for model API requests.

        Returns:
            Dict[str, str]: Headers to include in API requests
        """
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get model service-specific configuration.

        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        pass

    @abstractmethod
    def get_auth_type(self) -> str:
        """Get the authentication type identifier.

        Returns:
            str: Authentication type (e.g., 'api_key', 'iam', 'oauth')
        """
        pass

    def get_credentials(self) -> Optional[Any]:
        """Get raw authentication credentials if needed.

        Returns:
            Optional[Any]: Authentication credentials or None
        """
        return None
