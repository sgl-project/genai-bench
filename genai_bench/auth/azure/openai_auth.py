"""Azure OpenAI authentication provider."""

import os
from typing import Any, Dict, Optional

from genai_bench.auth.model_auth_provider import ModelAuthProvider


class AzureOpenAIAuth(ModelAuthProvider):
    """Azure OpenAI authentication provider."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_version: str = "2024-02-01",
        azure_endpoint: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        use_azure_ad: bool = False,
        azure_ad_token: Optional[str] = None,
    ):
        """Initialize Azure OpenAI authentication.

        Args:
            api_key: Azure OpenAI API key
            api_version: API version to use
            azure_endpoint: Azure OpenAI endpoint URL
            azure_deployment: Azure OpenAI deployment name
            use_azure_ad: Whether to use Azure AD authentication
            azure_ad_token: Azure AD token for authentication
        """
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = api_version
        self.azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_deployment = azure_deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        self.use_azure_ad = use_azure_ad
        self.azure_ad_token = azure_ad_token or os.getenv("AZURE_AD_TOKEN")

    def get_headers(self) -> Dict[str, str]:
        """Get authentication headers for Azure OpenAI API requests.

        Returns:
            Dict[str, str]: Headers with API key or Bearer token
        """
        headers = {}

        if self.use_azure_ad and self.azure_ad_token:
            headers["Authorization"] = f"Bearer {self.azure_ad_token}"
        elif self.api_key:
            headers["api-key"] = self.api_key

        return headers

    def get_config(self) -> Dict[str, Any]:
        """Get Azure OpenAI configuration.

        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        config: Dict[str, Any] = {
            "api_version": self.api_version,
            "auth_type": self.get_auth_type(),
        }

        if self.azure_endpoint:
            config["azure_endpoint"] = self.azure_endpoint

        if self.azure_deployment:
            config["azure_deployment"] = self.azure_deployment

        if self.use_azure_ad:
            config["use_azure_ad"] = self.use_azure_ad

        return config

    def get_auth_type(self) -> str:
        """Get the authentication type identifier.

        Returns:
            'azure_ad' if using Azure AD, otherwise 'api_key'
        """
        return "azure_ad" if self.use_azure_ad else "api_key"

    def get_credentials(self) -> Optional[Dict[str, Any]]:
        """Get Azure OpenAI credentials.

        Returns:
            Dict with credentials or None
        """
        if self.use_azure_ad:
            return {"azure_ad_token": self.azure_ad_token}
        else:
            return {"api_key": self.api_key}
