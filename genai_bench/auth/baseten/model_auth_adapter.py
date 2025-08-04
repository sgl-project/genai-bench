from typing import Any, Dict, Optional

from genai_bench.auth.baseten.auth import BasetenAuth
from genai_bench.auth.model_auth_provider import ModelAuthProvider


class BasetenModelAuthAdapter(ModelAuthProvider):
    """Adapter for Baseten authentication to model endpoint interface."""

    def __init__(self, baseten_auth: BasetenAuth):
        """Initialize the adapter.

        Args:
            baseten_auth (BasetenAuth): Baseten authentication provider
        """
        self.baseten_auth = baseten_auth

    def get_headers(self) -> Dict[str, str]:
        """Get authentication headers for Baseten API requests.

        Returns:
            Dict[str, str]: Headers with Authorization: Api-Key
        """
        api_key = self.baseten_auth.get_credentials()
        return {"Authorization": f"Api-Key {api_key}"}

    def get_config(self) -> Dict[str, Any]:
        """Get Baseten configuration.

        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        return self.baseten_auth.get_config()

    def get_auth_type(self) -> str:
        """Get the authentication type identifier.

        Returns:
            str: Authentication type ('api_key')
        """
        return "api_key"

    def get_credentials(self) -> Optional[str]:
        """Get raw authentication credentials.

        Returns:
            Optional[str]: Baseten API key
        """
        return self.baseten_auth.get_credentials() 