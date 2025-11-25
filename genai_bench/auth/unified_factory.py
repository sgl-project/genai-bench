"""Unified factory for creating model and storage authentication providers."""

import oci.config

# Model auth providers
from genai_bench.auth.aws.bedrock_auth import AWSBedrockAuth

# Storage auth providers
from genai_bench.auth.aws.s3_auth import AWSS3Auth
from genai_bench.auth.azure.blob_auth import AzureBlobAuth
from genai_bench.auth.azure.openai_auth import AzureOpenAIAuth

# Import existing auth providers
from genai_bench.auth.factory import AuthFactory
from genai_bench.auth.gcp.gcs_auth import GCPStorageAuth
from genai_bench.auth.gcp.vertex_auth import GCPVertexAuth
from genai_bench.auth.github.github_auth import GitHubAuth
from genai_bench.auth.model_auth_provider import ModelAuthProvider
from genai_bench.auth.oci.model_auth_adapter import OCIModelAuthAdapter
from genai_bench.auth.oci.storage_auth_adapter import OCIStorageAuthAdapter
from genai_bench.auth.openai.auth import OpenAIAuth
from genai_bench.auth.openai.model_auth_adapter import OpenAIModelAuthAdapter
from genai_bench.auth.storage_auth_provider import StorageAuthProvider
from genai_bench.auth.together.auth import TogetherAuth
from genai_bench.auth.together.model_auth_adapter import TogetherModelAuthAdapter


class UnifiedAuthFactory:
    """Factory for creating model and storage authentication providers."""

    @staticmethod
    def create_model_auth(provider: str, **kwargs) -> ModelAuthProvider:
        """Create a model endpoint authentication provider.

        Args:
            provider: Provider type ('openai', 'oci', 'aws-bedrock',
                'azure-openai', 'gcp-vertex', 'together')
            **kwargs: Provider-specific arguments

        Returns:
            ModelAuthProvider instance

        Raises:
            ValueError: If provider is not supported
        """
        if provider == "openai":
            api_key = kwargs.get("api_key")
            openai_auth = OpenAIAuth(api_key=api_key)
            return OpenAIModelAuthAdapter(openai_auth)

        elif provider == "oci":
            # Use existing OCI auth factory
            oci_auth = AuthFactory.create_oci_auth(
                auth_type=kwargs.get("auth_type", "user_principal"),
                config_path=kwargs.get("config_path", oci.config.DEFAULT_LOCATION),
                profile=kwargs.get("profile", oci.config.DEFAULT_PROFILE),
                token=kwargs.get("token"),
                region=kwargs.get("region"),
            )
            return OCIModelAuthAdapter(oci_auth)

        elif provider == "aws-bedrock":
            return AWSBedrockAuth(
                access_key_id=kwargs.get("access_key_id"),
                secret_access_key=kwargs.get("secret_access_key"),
                session_token=kwargs.get("session_token"),
                region=kwargs.get("region"),
                profile=kwargs.get("profile"),
            )

        elif provider == "azure-openai":
            return AzureOpenAIAuth(
                api_key=kwargs.get("api_key"),
                api_version=kwargs.get("api_version", "2024-02-01"),  # Provide default
                azure_endpoint=kwargs.get("azure_endpoint"),
                azure_deployment=kwargs.get("azure_deployment"),
                use_azure_ad=kwargs.get("use_azure_ad", False),
                azure_ad_token=kwargs.get("azure_ad_token"),
            )

        elif provider == "gcp-vertex":
            return GCPVertexAuth(
                project_id=kwargs.get("project_id"),
                location=kwargs.get("location"),
                credentials_path=kwargs.get("credentials_path"),
                api_key=kwargs.get("api_key"),
            )

        elif provider == "together":
            api_key = kwargs.get("api_key")
            together_auth = TogetherAuth(api_key=api_key)
            return TogetherModelAuthAdapter(together_auth)

        else:
            raise ValueError(
                f"Unsupported model provider: {provider}. "
                f"Supported: openai, oci, aws-bedrock, azure-openai, gcp-vertex, "
                "together"
            )

    @staticmethod
    def create_storage_auth(provider: str, **kwargs) -> StorageAuthProvider:
        """Create a storage authentication provider.

        Args:
            provider: Provider type ('oci', 'aws', 'azure', 'gcp', 'github')
            **kwargs: Provider-specific arguments

        Returns:
            StorageAuthProvider instance

        Raises:
            ValueError: If provider is not supported
        """
        if provider == "oci":
            # Use existing OCI auth factory
            oci_auth = AuthFactory.create_oci_auth(
                auth_type=kwargs.get("auth_type", "user_principal"),
                config_path=kwargs.get("config_path", oci.config.DEFAULT_LOCATION),
                profile=kwargs.get("profile", oci.config.DEFAULT_PROFILE),
                token=kwargs.get("token"),
                region=kwargs.get("region"),
            )
            return OCIStorageAuthAdapter(oci_auth)

        elif provider == "aws":
            return AWSS3Auth(
                access_key_id=kwargs.get("access_key_id"),
                secret_access_key=kwargs.get("secret_access_key"),
                session_token=kwargs.get("session_token"),
                region=kwargs.get("region"),
                profile=kwargs.get("profile"),
            )

        elif provider == "azure":
            return AzureBlobAuth(
                account_name=kwargs.get("account_name"),
                account_key=kwargs.get("account_key"),
                connection_string=kwargs.get("connection_string"),
                sas_token=kwargs.get("sas_token"),
                use_azure_ad=kwargs.get("use_azure_ad", False),
                tenant_id=kwargs.get("tenant_id"),
                client_id=kwargs.get("client_id"),
                client_secret=kwargs.get("client_secret"),
            )

        elif provider == "gcp":
            return GCPStorageAuth(
                project_id=kwargs.get("project_id"),
                credentials_path=kwargs.get("credentials_path"),
                access_token=kwargs.get("access_token"),
            )

        elif provider == "github":
            return GitHubAuth(
                token=kwargs.get("token"),
                owner=kwargs.get("owner"),
                repo=kwargs.get("repo"),
            )

        else:
            raise ValueError(
                f"Unsupported storage provider: {provider}. "
                f"Supported: oci, aws, azure, gcp, github"
            )
