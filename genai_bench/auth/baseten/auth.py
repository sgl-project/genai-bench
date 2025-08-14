import os
from typing import Any, Dict, Optional

from genai_bench.auth.auth_provider import AuthProvider


class BasetenAuth(AuthProvider):
    """Baseten Authentication Provider."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Baseten Auth Provider.

        Args:
            api_key (Optional[str]): Baseten API key. If None, will try to get from
            BASETEN_API_KEY environment variable.

        Raises:
            ValueError: If no API key is provided or found in environment
        """
        self.api_key = api_key or os.getenv("BASETEN_API_KEY")
        if not self.api_key or not self.api_key.strip():
            raise ValueError(
                "Baseten API key must be provided or set in "
                "BASETEN_API_KEY environment variable"
            )

    def get_config(self) -> Dict[str, Any]:
        """Get Baseten configuration.

        Returns:
            Dict[str, Any]: Empty configuration dictionary
            as Baseten doesn't need additional config
        """
        return {}

    def get_credentials(self) -> str:
        """Get Baseten API key.

        Returns:
            str: Baseten API key
        """
        if not self.api_key:
            raise ValueError("Baseten API key is not set")
        return self.api_key 