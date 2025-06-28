"""Integration tests for cross-cloud scenarios.

These tests verify that different cloud providers can be used together,
such as using AWS Bedrock for the model and Azure Blob for storage.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

from genai_bench.auth.unified_factory import UnifiedAuthFactory
from genai_bench.storage.factory import StorageFactory


class TestCrossCloudIntegration:
    """Test cases for cross-cloud integration scenarios."""

    def test_aws_bedrock_with_azure_storage(self):
        """Test using AWS Bedrock for model with Azure Blob for storage."""
        # Create AWS Bedrock model auth
        model_auth = UnifiedAuthFactory.create_model_auth(
            "aws-bedrock",
            access_key_id="aws_key",
            secret_access_key="aws_secret",
            region="us-east-1",
        )

        # Create Azure Blob storage auth
        storage_auth = UnifiedAuthFactory.create_storage_auth(
            "azure", account_name="test_account", account_key="test_key"
        )

        # Verify they are different providers
        assert model_auth.get_auth_type() == "aws_bedrock"
        assert storage_auth.get_storage_type() == "azure"

        # Mock the storage class
        mock_azure_storage = MagicMock()
        mock_azure_class = MagicMock(return_value=mock_azure_storage)

        with patch.dict(
            sys.modules,
            {
                "genai_bench.storage.azure_storage": MagicMock(
                    AzureBlobStorage=mock_azure_class
                )
            },
        ):
            # Create storage instance
            storage = StorageFactory.create_storage("azure", storage_auth)
            assert storage == mock_azure_storage

    def test_gcp_vertex_with_aws_s3(self):
        """Test using GCP Vertex AI for model with AWS S3 for storage."""
        # Create GCP Vertex model auth
        model_auth = UnifiedAuthFactory.create_model_auth(
            "gcp-vertex",
            project_id="test-project",
            location="us-central1",
            api_key="gcp_api_key",
        )

        # Create AWS S3 storage auth
        storage_auth = UnifiedAuthFactory.create_storage_auth(
            "aws",
            access_key_id="s3_key",
            secret_access_key="s3_secret",
            region="us-west-2",
        )

        # Verify they are different providers
        assert model_auth.get_auth_type() == "api_key"
        assert storage_auth.get_storage_type() == "aws"

        # Mock the storage class
        mock_s3_storage = MagicMock()
        mock_s3_class = MagicMock(return_value=mock_s3_storage)

        with patch.dict(
            sys.modules,
            {"genai_bench.storage.aws_storage": MagicMock(AWSS3Storage=mock_s3_class)},
        ):
            # Create storage instance
            storage = StorageFactory.create_storage("aws", storage_auth)
            assert storage == mock_s3_storage

    def test_azure_openai_with_github_storage(self):
        """Test using Azure OpenAI for model with GitHub for storage."""
        # Create Azure OpenAI model auth
        model_auth = UnifiedAuthFactory.create_model_auth(
            "azure-openai",
            api_key="azure_key",
            azure_endpoint="https://test.openai.azure.com",
            azure_deployment="gpt-4",
        )

        # Create GitHub storage auth
        storage_auth = UnifiedAuthFactory.create_storage_auth(
            "github", token="ghp_test_token", owner="test-owner", repo="test-repo"
        )

        # Verify they are different providers
        assert model_auth.get_auth_type() == "api_key"
        assert storage_auth.get_storage_type() == "github"

        # Mock the storage class
        mock_github_storage = MagicMock()
        mock_github_class = MagicMock(return_value=mock_github_storage)

        with patch.dict(
            sys.modules,
            {
                "genai_bench.storage.github_storage": MagicMock(
                    GitHubStorage=mock_github_class
                )
            },
        ):
            # Create storage instance
            storage = StorageFactory.create_storage("github", storage_auth)
            assert storage == mock_github_storage

    def test_openai_with_gcp_storage(self):
        """Test using OpenAI for model with GCP Cloud Storage."""
        # Create OpenAI model auth
        model_auth = UnifiedAuthFactory.create_model_auth(
            "openai", api_key="sk-test-key"
        )

        # Create GCP storage auth
        storage_auth = UnifiedAuthFactory.create_storage_auth(
            "gcp", project_id="storage-project", credentials_path="/path/to/creds.json"
        )

        # Verify they are different providers
        assert model_auth.get_auth_type() == "api_key"
        assert storage_auth.get_storage_type() == "gcp"

    def test_oci_with_aws_storage(self):
        """Test using OCI for model with AWS S3 storage."""
        # Mock OCI auth creation
        mock_oci_auth = MagicMock()

        with patch(
            "genai_bench.auth.factory.AuthFactory.create_oci_auth"
        ) as mock_create:
            mock_create.return_value = mock_oci_auth

            # Create OCI model auth
            model_auth = UnifiedAuthFactory.create_model_auth(
                "oci", auth_type="user_principal", profile="DEFAULT"
            )

        # Create AWS storage auth
        storage_auth = UnifiedAuthFactory.create_storage_auth(
            "aws", access_key_id="aws_key", secret_access_key="aws_secret"
        )

        # Verify different providers
        # OCI adapter returns the auth type from the wrapped auth object
        assert "oci" in model_auth.get_auth_type()
        assert storage_auth.get_storage_type() == "aws"

    def test_auth_isolation(self):
        """Test that auth objects are isolated between model and storage."""
        # Create model auth with specific credentials
        model_auth = UnifiedAuthFactory.create_model_auth(
            "aws-bedrock",
            access_key_id="model_key",
            secret_access_key="model_secret",
            session_token="model_token",
            region="us-east-1",
        )

        # Create storage auth with different credentials
        storage_auth = UnifiedAuthFactory.create_storage_auth(
            "aws",
            access_key_id="storage_key",
            secret_access_key="storage_secret",
            region="eu-west-1",
        )

        # Verify credentials are isolated
        assert model_auth.access_key_id == "model_key"
        assert model_auth.session_token == "model_token"
        assert model_auth.region == "us-east-1"

        assert storage_auth.access_key_id == "storage_key"
        assert storage_auth.session_token is None
        assert storage_auth.region == "eu-west-1"

    def test_environment_variable_isolation(self):
        """Test that env vars don't leak between model and storage auth."""
        with patch.dict(
            "os.environ",
            {
                "AWS_ACCESS_KEY_ID": "env_key",
                "AWS_SECRET_ACCESS_KEY": "env_secret",
                "AWS_DEFAULT_REGION": "us-west-1",
            },
        ):
            # Create model auth with explicit values
            model_auth = UnifiedAuthFactory.create_model_auth(
                "aws-bedrock", access_key_id="explicit_model_key", region="us-east-1"
            )

            # Create storage auth that should use env vars
            storage_auth = UnifiedAuthFactory.create_storage_auth("aws")

            # Model should use explicit values
            assert model_auth.access_key_id == "explicit_model_key"
            assert model_auth.region == "us-east-1"

            # Storage should use env vars
            assert storage_auth.access_key_id == "env_key"
            assert storage_auth.region == "us-west-1"

    def test_multiple_cloud_combinations(self):
        """Test creating multiple cloud combinations in sequence."""
        combinations = [
            ("aws-bedrock", "azure"),
            ("azure-openai", "gcp"),
            ("gcp-vertex", "github"),
            ("openai", "aws"),
        ]

        created_auths = []

        for model_provider, storage_provider in combinations:
            # Create model auth
            if model_provider == "openai":
                model_auth = UnifiedAuthFactory.create_model_auth(
                    model_provider, api_key=f"{model_provider}_key"
                )
            else:
                model_auth = UnifiedAuthFactory.create_model_auth(model_provider)

            # Create storage auth
            storage_auth = UnifiedAuthFactory.create_storage_auth(storage_provider)

            created_auths.append((model_auth, storage_auth))

        # Verify all combinations were created
        assert len(created_auths) == len(combinations)

        # Verify each combination is unique
        for i, (model_auth, storage_auth) in enumerate(created_auths):
            model_provider, storage_provider = combinations[i]

            # Check model auth type
            if model_provider == "aws-bedrock":
                assert model_auth.get_auth_type() == "aws_bedrock"
            elif model_provider in ["azure-openai", "openai"]:
                assert model_auth.get_auth_type() == "api_key"
            elif model_provider == "gcp-vertex":
                # GCP Vertex defaults to service_account when no API key is provided
                assert model_auth.get_auth_type() in ["api_key", "service_account"]

            # Check storage type
            assert storage_auth.get_storage_type() == storage_provider

    @patch("genai_bench.auth.factory.AuthFactory.create_oci_auth")
    def test_same_provider_different_auth(self, mock_create_oci):
        """Test using same provider with different auth for model and storage."""
        # Mock different OCI auth instances
        model_oci_auth = MagicMock()
        storage_oci_auth = MagicMock()
        mock_create_oci.side_effect = [model_oci_auth, storage_oci_auth]

        # Create OCI model auth with user principal
        model_auth = UnifiedAuthFactory.create_model_auth(
            "oci", auth_type="user_principal", profile="MODEL_PROFILE"
        )

        # Create OCI storage auth with instance principal
        storage_auth = UnifiedAuthFactory.create_storage_auth(
            "oci", auth_type="instance_principal"
        )

        # Verify different auth instances were created
        assert model_auth != storage_auth
        assert mock_create_oci.call_count == 2

        # Check the calls
        calls = mock_create_oci.call_args_list
        assert calls[0][1]["auth_type"] == "user_principal"
        assert calls[0][1]["profile"] == "MODEL_PROFILE"
        assert calls[1][1]["auth_type"] == "instance_principal"

    def test_error_handling_cross_cloud(self):
        """Test error handling in cross-cloud scenarios."""
        # Create valid model auth
        UnifiedAuthFactory.create_model_auth("openai", api_key="test_key")

        # Try to create storage with mismatched provider
        storage_auth = UnifiedAuthFactory.create_storage_auth("aws")

        # This should fail when trying to use AWS storage auth with Azure provider
        with pytest.raises(ValueError) as exc_info:
            StorageFactory.create_storage("azure", storage_auth)

        assert "Storage provider 'azure' does not match auth type 'aws'" in str(
            exc_info.value
        )
