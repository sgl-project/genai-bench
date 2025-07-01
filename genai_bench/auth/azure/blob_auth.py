"""Azure Blob Storage authentication provider."""

import os
from typing import Any, Dict, Optional

from genai_bench.auth.storage_auth_provider import StorageAuthProvider


class AzureBlobAuth(StorageAuthProvider):
    """Azure Blob Storage authentication provider."""

    def __init__(
        self,
        account_name: Optional[str] = None,
        account_key: Optional[str] = None,
        connection_string: Optional[str] = None,
        sas_token: Optional[str] = None,
        use_azure_ad: bool = False,
        tenant_id: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
    ):
        """Initialize Azure Blob Storage authentication.

        Args:
            account_name: Storage account name
            account_key: Storage account key
            connection_string: Full connection string (overrides other params)
            sas_token: Shared Access Signature token
            use_azure_ad: Whether to use Azure AD authentication
            tenant_id: Azure AD tenant ID
            client_id: Azure AD client/application ID
            client_secret: Azure AD client secret
        """
        self.account_name = account_name or os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
        self.account_key = account_key or os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
        self.connection_string = connection_string or os.getenv(
            "AZURE_STORAGE_CONNECTION_STRING"
        )
        self.sas_token = sas_token or os.getenv("AZURE_STORAGE_SAS_TOKEN")
        self.use_azure_ad = use_azure_ad
        self.tenant_id = tenant_id or os.getenv("AZURE_TENANT_ID")
        self.client_id = client_id or os.getenv("AZURE_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("AZURE_CLIENT_SECRET")

    def get_client_config(self) -> Dict[str, Any]:
        """Get Azure Blob Storage client configuration.

        Returns:
            Dict[str, Any]: Configuration for storage client
        """
        config: Dict[str, Any] = {}

        if self.connection_string:
            config["connection_string"] = self.connection_string
        else:
            if self.account_name:
                config["account_name"] = self.account_name

            if self.account_key:
                config["account_key"] = self.account_key
            elif self.sas_token:
                config["sas_token"] = self.sas_token
            elif self.use_azure_ad:
                config["use_azure_ad"] = self.use_azure_ad
                if self.tenant_id:
                    config["tenant_id"] = self.tenant_id
                if self.client_id:
                    config["client_id"] = self.client_id
                if self.client_secret:
                    config["client_secret"] = self.client_secret

        return config

    def get_credentials(self) -> Dict[str, Any]:
        """Get Azure credentials for storage operations.

        Returns:
            Dict with Azure credentials
        """
        creds: Dict[str, Any] = {}

        if self.connection_string:
            creds["connection_string"] = self.connection_string
        else:
            if self.account_name:
                creds["account_name"] = self.account_name

            if self.account_key:
                creds["account_key"] = self.account_key
            elif self.sas_token:
                creds["sas_token"] = self.sas_token
            elif self.use_azure_ad:
                creds["use_azure_ad"] = self.use_azure_ad
                if self.tenant_id:
                    creds["tenant_id"] = self.tenant_id
                if self.client_id:
                    creds["client_id"] = self.client_id
                if self.client_secret:
                    creds["client_secret"] = self.client_secret

        return creds

    def get_storage_type(self) -> str:
        """Get the storage provider type.

        Returns:
            'azure'
        """
        return "azure"
