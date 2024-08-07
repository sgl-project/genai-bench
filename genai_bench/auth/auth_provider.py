from abc import ABC, abstractmethod
from typing import Any, Dict


class AuthProvider(ABC):
    """Base class for authentication providers."""

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get service-specific configuration.

        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        pass

    @abstractmethod
    def get_auth_credentials(self) -> Any:
        """Get authentication credentials.

        Returns:
            Any: Authentication credentials (e.g., API key, signer)
        """
        pass
