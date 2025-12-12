import os
from typing import Any, Dict, Optional

from genai_bench.auth.auth_provider import AuthProvider


class TogetherAuth(AuthProvider):
    """Together.ai Authentication Provider."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Together Auth Provider.

        Args:
            api_key (Optional[str]): Together API key. If None, will try to get from
            TOGETHER_API_KEY environment variable.

        Raises:
            ValueError: If no API key is provided or found in environment
        """
        self.api_key = api_key or os.getenv("TOGETHER_API_KEY")
        if not self.api_key or not self.api_key.strip():
            raise ValueError(
                "Together API key must be provided or set in "
                "TOGETHER_API_KEY environment variable"
            )

    def get_config(self) -> Dict[str, Any]:
        """Get Together configuration.

        Returns:
            Dict[str, Any]: Empty configuration dictionary
            as OpenAI doesn't need additional config
        """
        return {}

    def get_credentials(self) -> str:
        """Get Together API key.

        Returns:
            str: Together API key
        """
        if not self.api_key:
            raise ValueError("Together API key is not set")
        return self.api_key
