"""Unit tests for Azure Blob Storage authentication provider."""

import os
from unittest.mock import patch

from genai_bench.auth.azure.blob_auth import AzureBlobAuth


class TestAzureBlobAuth:
    """Test cases for Azure Blob Storage authentication."""

    def test_init_with_account_key(self):
        """Test initialization with account key authentication."""
        auth = AzureBlobAuth(account_name="testaccount", account_key="test_key")

        assert auth.account_name == "testaccount"
        assert auth.account_key == "test_key"
        assert auth.connection_string is None
        assert auth.sas_token is None
        assert auth.use_azure_ad is False

    def test_init_with_connection_string(self):
        """Test initialization with connection string."""
        conn_str = (
            "DefaultEndpointsProtocol=https;AccountName=test;"
            "AccountKey=key;EndpointSuffix=core.windows.net"
        )
        auth = AzureBlobAuth(connection_string=conn_str)

        assert auth.connection_string == conn_str
        assert auth.account_name is None
        assert auth.account_key is None

    def test_init_with_sas_token(self):
        """Test initialization with SAS token."""
        auth = AzureBlobAuth(
            account_name="testaccount",
            sas_token="?sv=2020-08-04&ss=b&srt=sco&sp=rwdlac",
        )

        assert auth.account_name == "testaccount"
        assert auth.sas_token == "?sv=2020-08-04&ss=b&srt=sco&sp=rwdlac"
        assert auth.account_key is None

    def test_init_with_azure_ad(self):
        """Test initialization with Azure AD authentication."""
        auth = AzureBlobAuth(
            account_name="testaccount",
            use_azure_ad=True,
            tenant_id="tenant-123",
            client_id="client-456",
            client_secret="secret-789",
        )

        assert auth.account_name == "testaccount"
        assert auth.use_azure_ad is True
        assert auth.tenant_id == "tenant-123"
        assert auth.client_id == "client-456"
        assert auth.client_secret == "secret-789"

    def test_init_with_env_vars(self):
        """Test initialization with environment variables."""
        with patch.dict(
            os.environ,
            {
                "AZURE_STORAGE_ACCOUNT_NAME": "env_account",
                "AZURE_STORAGE_ACCOUNT_KEY": "env_key",
                "AZURE_STORAGE_CONNECTION_STRING": "env_conn_str",
                "AZURE_STORAGE_SAS_TOKEN": "env_sas",
                "AZURE_TENANT_ID": "env_tenant",
                "AZURE_CLIENT_ID": "env_client",
                "AZURE_CLIENT_SECRET": "env_secret",
            },
        ):
            auth = AzureBlobAuth()

            assert auth.account_name == "env_account"
            assert auth.account_key == "env_key"
            assert auth.connection_string == "env_conn_str"
            assert auth.sas_token == "env_sas"
            assert auth.tenant_id == "env_tenant"
            assert auth.client_id == "env_client"
            assert auth.client_secret == "env_secret"

    def test_get_client_config_connection_string(self):
        """Test client config when using connection string."""
        conn_str = "DefaultEndpointsProtocol=https;AccountName=test"
        auth = AzureBlobAuth(connection_string=conn_str)
        config = auth.get_client_config()

        assert config == {"connection_string": conn_str}

    def test_get_client_config_account_key(self):
        """Test client config when using account key."""
        auth = AzureBlobAuth(account_name="testaccount", account_key="testkey")
        config = auth.get_client_config()

        assert config == {"account_name": "testaccount", "account_key": "testkey"}

    def test_get_client_config_sas_token(self):
        """Test client config when using SAS token."""
        auth = AzureBlobAuth(account_name="testaccount", sas_token="?sv=2020")
        config = auth.get_client_config()

        assert config == {"account_name": "testaccount", "sas_token": "?sv=2020"}

    def test_get_client_config_azure_ad(self):
        """Test client config when using Azure AD."""
        auth = AzureBlobAuth(
            account_name="testaccount",
            use_azure_ad=True,
            tenant_id="tenant",
            client_id="client",
            client_secret="secret",
        )
        config = auth.get_client_config()

        assert config == {
            "account_name": "testaccount",
            "use_azure_ad": True,
            "tenant_id": "tenant",
            "client_id": "client",
            "client_secret": "secret",
        }

    def test_get_client_config_azure_ad_partial(self):
        """Test client config with partial Azure AD credentials."""
        auth = AzureBlobAuth(
            account_name="testaccount",
            use_azure_ad=True,
            tenant_id="tenant",
            # Missing client_id and client_secret
        )
        config = auth.get_client_config()

        assert config == {
            "account_name": "testaccount",
            "use_azure_ad": True,
            "tenant_id": "tenant",
        }

    def test_get_credentials_same_as_config(self):
        """Test that auth credentials match client config."""
        auth = AzureBlobAuth(account_name="test", account_key="key")

        assert auth.get_credentials() == auth.get_client_config()

    def test_get_storage_type(self):
        """Test storage type identifier."""
        auth = AzureBlobAuth()
        assert auth.get_storage_type() == "azure"

    def test_precedence_connection_string(self):
        """Test that connection string takes precedence."""
        auth = AzureBlobAuth(
            connection_string="conn_str", account_name="ignored", account_key="ignored"
        )
        config = auth.get_client_config()

        assert config == {"connection_string": "conn_str"}

    def test_precedence_explicit_over_env(self):
        """Test that explicit values take precedence over env vars."""
        with patch.dict(
            os.environ,
            {
                "AZURE_STORAGE_ACCOUNT_NAME": "env_account",
                "AZURE_STORAGE_ACCOUNT_KEY": "env_key",
            },
        ):
            auth = AzureBlobAuth(
                account_name="explicit_account", account_key="explicit_key"
            )

            assert auth.account_name == "explicit_account"
            assert auth.account_key == "explicit_key"

    def test_multiple_auth_methods_priority(self):
        """Test priority when multiple auth methods are provided."""
        # Connection string should take precedence
        auth = AzureBlobAuth(
            connection_string="conn_str",
            account_name="account",
            account_key="key",
            sas_token="sas",
            use_azure_ad=True,
        )
        config = auth.get_client_config()

        # Only connection string should be in config
        assert config == {"connection_string": "conn_str"}

    def test_get_region_none(self):
        """Test getting region when none is set."""
        auth = AzureBlobAuth(account_name="test", account_key="key")
        assert auth.get_region() is None

    def test_auth_with_service_principal(self):
        """Test auth configuration with service principal."""
        auth = AzureBlobAuth(
            account_name="test",
            use_azure_ad=True,
            tenant_id="test-tenant",
            client_id="test-client",
            client_secret="test-secret",
        )

        creds = auth.get_credentials()
        assert creds["use_azure_ad"] is True
        assert creds["tenant_id"] == "test-tenant"
        assert creds["client_id"] == "test-client"
        assert creds["client_secret"] == "test-secret"

    def test_auth_with_partial_service_principal(self):
        """Test auth configuration with partial service principal info."""
        auth = AzureBlobAuth(
            account_name="test",
            use_azure_ad=True,
            tenant_id="test-tenant",
            # Missing client_id and client_secret
        )

        creds = auth.get_credentials()
        assert creds["use_azure_ad"] is True
        assert creds["tenant_id"] == "test-tenant"
        assert "client_id" not in creds
        assert "client_secret" not in creds

    def test_auth_no_credentials(self):
        """Test auth with no credentials provided."""
        auth = AzureBlobAuth(account_name="test")
        creds = auth.get_credentials()
        assert creds == {"account_name": "test"}

    def test_auth_credentials_with_sas_token(self):
        """Test auth credentials with SAS token take precedence over Azure AD."""
        auth = AzureBlobAuth(
            account_name="test", sas_token="?sv=2020", use_azure_ad=True
        )
        creds = auth.get_credentials()
        assert creds == {"account_name": "test", "sas_token": "?sv=2020"}
        assert "use_azure_ad" not in creds
