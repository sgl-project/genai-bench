"""Together model authentication adapter for backward compatibility."""

from typing import Any, Dict

from genai_bench.auth.model_auth_provider import ModelAuthProvider
from genai_bench.auth.together.auth import TogetherAuth


class TogetherModelAuthAdapter(ModelAuthProvider):
    """Adapter to use existing Together auth as model auth provider."""

    def __init__(self, together_auth: TogetherAuth):
        """Initialize Together model auth adapter.

        Args:
            together_auth: Existing Together auth instance
        """
        self.together_auth = together_auth

    def get_headers(self) -> Dict[str, str]:
        """Get authentication headers for Together API requests.

        Returns:
            Dict[str, str]: Headers with Authorization
        """
        # Together uses Bearer token in Authorization header
        if self.together_auth.api_key:
            return {"Authorization": f"Bearer {self.together_auth.api_key}"}
        return {}

    def get_config(self) -> Dict[str, Any]:
        """Get Together model service configuration.

        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        return {
            "auth_type": self.get_auth_type(),
            "has_api_key": bool(self.together_auth.api_key),
        }

    def get_auth_type(self) -> str:
        """Get the authentication type identifier.

        Returns:
            'api_key'
        """
        return "api_key"

    def get_credentials(self) -> Dict[str, str]:
        """Get Together credentials.

        Returns:
            Dict with API key
        """
        if self.together_auth.api_key:
            return {"api_key": self.together_auth.api_key}
        return {}
