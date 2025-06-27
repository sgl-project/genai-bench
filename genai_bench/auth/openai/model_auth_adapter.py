"""OpenAI model authentication adapter for backward compatibility."""

from typing import Any, Dict

from genai_bench.auth.model_auth_provider import ModelAuthProvider
from genai_bench.auth.openai.auth import OpenAIAuth


class OpenAIModelAuthAdapter(ModelAuthProvider):
    """Adapter to use existing OpenAI auth as model auth provider."""

    def __init__(self, openai_auth: OpenAIAuth):
        """Initialize OpenAI model auth adapter.

        Args:
            openai_auth: Existing OpenAI auth instance
        """
        self.openai_auth = openai_auth

    def get_headers(self) -> Dict[str, str]:
        """Get authentication headers for OpenAI API requests.

        Returns:
            Dict[str, str]: Headers with Authorization
        """
        # OpenAI uses Bearer token in Authorization header
        if self.openai_auth.api_key:
            return {"Authorization": f"Bearer {self.openai_auth.api_key}"}
        return {}

    def get_config(self) -> Dict[str, Any]:
        """Get OpenAI model service configuration.

        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        return {
            "auth_type": self.get_auth_type(),
            "has_api_key": bool(self.openai_auth.api_key),
        }

    def get_auth_type(self) -> str:
        """Get the authentication type identifier.

        Returns:
            'api_key'
        """
        return "api_key"

    def get_credentials(self) -> Dict[str, str]:
        """Get OpenAI credentials.

        Returns:
            Dict with API key
        """
        if self.openai_auth.api_key:
            return {"api_key": self.openai_auth.api_key}
        return {}
