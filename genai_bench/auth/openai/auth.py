import os
from typing import Any, Dict, Optional

from genai_bench.auth.auth_provider import AuthProvider


class OpenAIAuth(AuthProvider):
    """OpenAI Authentication Provider."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize OpenAI Auth Provider.

        Args:
            api_key (Optional[str]): OpenAI API key. If None, will try to get from
            OPENAI_API_KEY environment variable.

        Raises:
            ValueError: If no API key is provided or found in environment
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key or not self.api_key.strip():
            raise ValueError(
                "OpenAI API key must be provided or set in "
                "OPENAI_API_KEY environment variable"
            )

    def get_config(self) -> Dict[str, Any]:
        """Get OpenAI configuration.

        Returns:
            Dict[str, Any]: Empty configuration dictionary
            as OpenAI doesn't need additional config
        """
        return {}

    def get_credentials(self) -> str:
        """Get OpenAI API key.

        Returns:
            str: OpenAI API key
        """
        if not self.api_key:
            raise ValueError("OpenAI API key is not set")
        return self.api_key
