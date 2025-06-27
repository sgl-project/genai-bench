"""Unit tests for the unified authentication factory."""

from unittest.mock import ANY, MagicMock, patch

import pytest

from genai_bench.auth.aws.bedrock_auth import AWSBedrockAuth
from genai_bench.auth.aws.s3_auth import AWSS3Auth
from genai_bench.auth.azure.blob_auth import AzureBlobAuth
from genai_bench.auth.azure.openai_auth import AzureOpenAIAuth
from genai_bench.auth.gcp.gcs_auth import GCPStorageAuth
from genai_bench.auth.gcp.vertex_auth import GCPVertexAuth
from genai_bench.auth.github.github_auth import GitHubAuth
from genai_bench.auth.model_auth_provider import ModelAuthProvider
from genai_bench.auth.storage_auth_provider import StorageAuthProvider
from genai_bench.auth.unified_factory import UnifiedAuthFactory


class TestUnifiedAuthFactory:
    """Test cases for the unified authentication factory."""

    # Model Auth Provider Tests

    def test_create_openai_model_auth(self):
        """Test creating OpenAI model auth provider."""
        auth = UnifiedAuthFactory.create_model_auth("openai", api_key="test_key")

        assert isinstance(auth, ModelAuthProvider)
        assert auth.get_auth_type() == "api_key"
        creds = auth.get_credentials()
        assert creds["api_key"] == "test_key"

    def test_create_oci_model_auth(self):
        """Test creating OCI model auth provider."""
        with patch(
            "genai_bench.auth.factory.AuthFactory.create_oci_auth"
        ) as mock_create:
            mock_oci_auth = MagicMock()
            mock_create.return_value = mock_oci_auth

            auth = UnifiedAuthFactory.create_model_auth(
                "oci",
                auth_type="user_principal",
                profile="test_profile",
                region="us-ashburn-1",
            )

            assert isinstance(auth, ModelAuthProvider)
            mock_create.assert_called_once_with(
                auth_type="user_principal",
                config_path=ANY,
                profile="test_profile",
                token=None,
                region="us-ashburn-1",
            )

    def test_create_aws_bedrock_model_auth(self):
        """Test creating AWS Bedrock model auth provider."""
        auth = UnifiedAuthFactory.create_model_auth(
            "aws-bedrock",
            access_key_id="aws_key",
            secret_access_key="aws_secret",
            region="us-east-1",
        )

        assert isinstance(auth, AWSBedrockAuth)
        assert auth.access_key_id == "aws_key"
        assert auth.secret_access_key == "aws_secret"
        assert auth.region == "us-east-1"

    def test_create_azure_openai_model_auth(self):
        """Test creating Azure OpenAI model auth provider."""
        auth = UnifiedAuthFactory.create_model_auth(
            "azure-openai",
            api_key="azure_key",
            azure_endpoint="https://test.openai.azure.com",
            azure_deployment="gpt-4",
        )

        assert isinstance(auth, AzureOpenAIAuth)
        assert auth.api_key == "azure_key"
        assert auth.azure_endpoint == "https://test.openai.azure.com"
        assert auth.azure_deployment == "gpt-4"

    def test_create_gcp_vertex_model_auth(self):
        """Test creating GCP Vertex AI model auth provider."""
        auth = UnifiedAuthFactory.create_model_auth(
            "gcp-vertex",
            project_id="test-project",
            location="us-central1",
            credentials_path="/path/to/creds.json",
        )

        assert isinstance(auth, GCPVertexAuth)
        assert auth.project_id == "test-project"
        assert auth.location == "us-central1"
        assert auth.credentials_path == "/path/to/creds.json"

    def test_create_model_auth_unsupported(self):
        """Test creating model auth with unsupported provider."""
        with pytest.raises(ValueError) as exc_info:
            UnifiedAuthFactory.create_model_auth("unsupported-provider")

        assert "Unsupported model provider" in str(exc_info.value)
        assert "openai, oci, aws-bedrock, azure-openai, gcp-vertex" in str(
            exc_info.value
        )

    # Storage Auth Provider Tests

    def test_create_oci_storage_auth(self):
        """Test creating OCI storage auth provider."""
        with patch(
            "genai_bench.auth.factory.AuthFactory.create_oci_auth"
        ) as mock_create:
            mock_oci_auth = MagicMock()
            mock_create.return_value = mock_oci_auth

            auth = UnifiedAuthFactory.create_storage_auth(
                "oci", auth_type="instance_principal", region="us-phoenix-1"
            )

            assert isinstance(auth, StorageAuthProvider)
            assert auth.get_storage_type() == "oci"

    def test_create_aws_s3_storage_auth(self):
        """Test creating AWS S3 storage auth provider."""
        auth = UnifiedAuthFactory.create_storage_auth(
            "aws",
            access_key_id="s3_key",
            secret_access_key="s3_secret",
            region="eu-west-1",
        )

        assert isinstance(auth, AWSS3Auth)
        assert auth.access_key_id == "s3_key"
        assert auth.secret_access_key == "s3_secret"
        assert auth.region == "eu-west-1"

    def test_create_azure_blob_storage_auth(self):
        """Test creating Azure Blob storage auth provider."""
        auth = UnifiedAuthFactory.create_storage_auth(
            "azure",
            account_name="testaccount",
            account_key="testkey",
            use_azure_ad=False,
        )

        assert isinstance(auth, AzureBlobAuth)
        assert auth.account_name == "testaccount"
        assert auth.account_key == "testkey"
        assert auth.use_azure_ad is False

    def test_create_gcp_storage_auth(self):
        """Test creating GCP Cloud Storage auth provider."""
        auth = UnifiedAuthFactory.create_storage_auth(
            "gcp",
            project_id="storage-project",
            credentials_path="/path/to/storage-creds.json",
        )

        assert isinstance(auth, GCPStorageAuth)
        assert auth.project_id == "storage-project"
        assert auth.credentials_path == "/path/to/storage-creds.json"

    def test_create_github_storage_auth(self):
        """Test creating GitHub storage auth provider."""
        auth = UnifiedAuthFactory.create_storage_auth(
            "github", token="ghp_test_token", owner="test-owner", repo="test-repo"
        )

        assert isinstance(auth, GitHubAuth)
        assert auth.token == "ghp_test_token"
        assert auth.owner == "test-owner"
        assert auth.repo == "test-repo"

    def test_create_storage_auth_unsupported(self):
        """Test creating storage auth with unsupported provider."""
        with pytest.raises(ValueError) as exc_info:
            UnifiedAuthFactory.create_storage_auth("unsupported-storage")

        assert "Unsupported storage provider" in str(exc_info.value)
        assert "oci, aws, azure, gcp, github" in str(exc_info.value)

    # Cross-provider Tests

    def test_create_different_model_and_storage_providers(self):
        """Test creating different providers for model and storage."""
        # AWS Bedrock for model
        model_auth = UnifiedAuthFactory.create_model_auth(
            "aws-bedrock", access_key_id="bedrock_key", region="us-east-1"
        )

        # Azure Blob for storage
        storage_auth = UnifiedAuthFactory.create_storage_auth(
            "azure", account_name="storageaccount", account_key="storagekey"
        )

        assert isinstance(model_auth, AWSBedrockAuth)
        assert isinstance(storage_auth, AzureBlobAuth)
        assert model_auth.get_auth_type() == "aws_bedrock"
        assert storage_auth.get_storage_type() == "azure"

    def test_kwargs_isolation(self):
        """Test that kwargs don't leak between different auth creations."""
        # Create with specific kwargs
        auth1 = UnifiedAuthFactory.create_model_auth(
            "aws-bedrock", access_key_id="key1", profile="profile1"
        )

        # Create another with different kwargs
        auth2 = UnifiedAuthFactory.create_model_auth(
            "aws-bedrock", access_key_id="key2", session_token="token2"
        )

        assert auth1.access_key_id == "key1"
        assert auth1.profile == "profile1"
        assert auth1.session_token is None

        assert auth2.access_key_id == "key2"
        assert auth2.profile is None
        assert auth2.session_token == "token2"

    def test_empty_kwargs(self):
        """Test creating auth providers with empty kwargs."""
        # These should work but may have None values
        auth = UnifiedAuthFactory.create_model_auth("aws-bedrock")
        assert isinstance(auth, AWSBedrockAuth)

        storage_auth = UnifiedAuthFactory.create_storage_auth("github")
        assert isinstance(storage_auth, GitHubAuth)
