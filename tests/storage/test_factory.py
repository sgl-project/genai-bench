"""Unit tests for the storage factory."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from genai_bench.storage.base import BaseStorage
from genai_bench.storage.factory import StorageFactory


class TestStorageFactory:
    """Test cases for the storage factory."""

    def test_create_oci_storage(self):
        """Test creating OCI storage provider."""
        # Mock the OCI storage class
        mock_oci_storage = MagicMock(spec=BaseStorage)
        mock_oci_class = MagicMock(return_value=mock_oci_storage)

        # Mock the auth provider
        mock_auth = MagicMock()
        mock_auth.get_storage_type.return_value = "oci"

        with patch.dict(
            sys.modules,
            {
                "genai_bench.storage.oci_storage": MagicMock(
                    OCIObjectStorage=mock_oci_class
                )
            },
        ):
            storage = StorageFactory.create_storage(
                "oci", mock_auth, region="us-ashburn-1"
            )

            assert storage == mock_oci_storage
            mock_oci_class.assert_called_once_with(mock_auth, region="us-ashburn-1")
            mock_auth.get_storage_type.assert_called_once()

    def test_create_aws_storage(self):
        """Test creating AWS S3 storage provider."""
        # Mock the AWS storage class
        mock_aws_storage = MagicMock(spec=BaseStorage)
        mock_aws_class = MagicMock(return_value=mock_aws_storage)

        # Mock the auth provider
        mock_auth = MagicMock()
        mock_auth.get_storage_type.return_value = "aws"

        with patch.dict(
            sys.modules,
            {"genai_bench.storage.aws_storage": MagicMock(AWSS3Storage=mock_aws_class)},
        ):
            storage = StorageFactory.create_storage(
                "aws", mock_auth, bucket="test-bucket"
            )

            assert storage == mock_aws_storage
            mock_aws_class.assert_called_once_with(mock_auth, bucket="test-bucket")

    def test_create_azure_storage(self):
        """Test creating Azure Blob storage provider."""
        # Mock the Azure storage class
        mock_azure_storage = MagicMock(spec=BaseStorage)
        mock_azure_class = MagicMock(return_value=mock_azure_storage)

        # Mock the auth provider
        mock_auth = MagicMock()
        mock_auth.get_storage_type.return_value = "azure"

        with patch.dict(
            sys.modules,
            {
                "genai_bench.storage.azure_storage": MagicMock(
                    AzureBlobStorage=mock_azure_class
                )
            },
        ):
            storage = StorageFactory.create_storage(
                "azure", mock_auth, container="test-container"
            )

            assert storage == mock_azure_storage
            mock_azure_class.assert_called_once_with(
                mock_auth, container="test-container"
            )

    def test_create_gcp_storage(self):
        """Test creating GCP Cloud Storage provider."""
        # Mock the GCP storage class
        mock_gcp_storage = MagicMock(spec=BaseStorage)
        mock_gcp_class = MagicMock(return_value=mock_gcp_storage)

        # Mock the auth provider
        mock_auth = MagicMock()
        mock_auth.get_storage_type.return_value = "gcp"

        with patch.dict(
            sys.modules,
            {
                "genai_bench.storage.gcp_storage": MagicMock(
                    GCPCloudStorage=mock_gcp_class
                )
            },
        ):
            storage = StorageFactory.create_storage(
                "gcp", mock_auth, bucket="gcp-bucket"
            )

            assert storage == mock_gcp_storage
            mock_gcp_class.assert_called_once_with(mock_auth, bucket="gcp-bucket")

    def test_create_github_storage(self):
        """Test creating GitHub storage provider."""
        # Mock the GitHub storage class
        mock_github_storage = MagicMock(spec=BaseStorage)
        mock_github_class = MagicMock(return_value=mock_github_storage)

        # Mock the auth provider
        mock_auth = MagicMock()
        mock_auth.get_storage_type.return_value = "github"

        with patch.dict(
            sys.modules,
            {
                "genai_bench.storage.github_storage": MagicMock(
                    GitHubStorage=mock_github_class
                )
            },
        ):
            storage = StorageFactory.create_storage("github", mock_auth, branch="main")

            assert storage == mock_github_storage
            mock_github_class.assert_called_once_with(mock_auth, branch="main")

    def test_create_storage_provider_mismatch(self):
        """Test error when provider doesn't match auth type."""
        mock_auth = MagicMock()
        mock_auth.get_storage_type.return_value = "aws"

        with pytest.raises(ValueError) as exc_info:
            StorageFactory.create_storage("azure", mock_auth)

        assert "Storage provider 'azure' does not match auth type 'aws'" in str(
            exc_info.value
        )

    def test_create_storage_unsupported(self):
        """Test error for unsupported storage provider."""
        mock_auth = MagicMock()
        mock_auth.get_storage_type.return_value = "unsupported"

        with pytest.raises(ValueError) as exc_info:
            StorageFactory.create_storage("unsupported", mock_auth)

        assert "Unsupported storage provider: unsupported" in str(exc_info.value)
        assert "Supported: oci, aws, azure, gcp, github" in str(exc_info.value)

    def test_lazy_import_behavior(self):
        """Test that storage modules are only imported when needed."""
        mock_auth = MagicMock()
        mock_auth.get_storage_type.return_value = "aws"

        # Remove any cached imports
        for module in list(sys.modules.keys()):
            if module.startswith("genai_bench.storage."):
                del sys.modules[module]

        # Mock only the AWS module
        mock_aws_storage = MagicMock(spec=BaseStorage)
        mock_aws_class = MagicMock(return_value=mock_aws_storage)

        with patch.dict(
            sys.modules,
            {"genai_bench.storage.aws_storage": MagicMock(AWSS3Storage=mock_aws_class)},
        ):
            # This should work without importing other storage modules
            storage = StorageFactory.create_storage("aws", mock_auth)
            assert storage == mock_aws_storage

            # Verify other modules weren't imported
            assert "genai_bench.storage.azure_storage" not in sys.modules
            assert "genai_bench.storage.gcp_storage" not in sys.modules

    def test_kwargs_passed_to_storage(self):
        """Test that kwargs are properly passed to storage constructors."""
        mock_auth = MagicMock()
        mock_auth.get_storage_type.return_value = "aws"

        mock_storage = MagicMock(spec=BaseStorage)
        mock_class = MagicMock(return_value=mock_storage)

        with patch.dict(
            sys.modules,
            {"genai_bench.storage.aws_storage": MagicMock(AWSS3Storage=mock_class)},
        ):
            StorageFactory.create_storage(
                "aws",
                mock_auth,
                bucket="test-bucket",
                region="us-east-1",
                custom_option="value",
            )

            mock_class.assert_called_once_with(
                mock_auth,
                bucket="test-bucket",
                region="us-east-1",
                custom_option="value",
            )

    def test_auth_type_validation_for_all_providers(self):
        """Test auth type validation for all supported providers."""
        providers = ["oci", "aws", "azure", "gcp", "github"]

        for provider in providers:
            mock_auth = MagicMock()
            # Set mismatched auth type
            mock_auth.get_storage_type.return_value = "wrong_type"

            with pytest.raises(ValueError) as exc_info:
                StorageFactory.create_storage(provider, mock_auth)

            assert (
                f"Storage provider '{provider}' does not match auth type 'wrong_type'"
                in str(exc_info.value)
            )

    def test_multiple_storage_instances(self):
        """Test creating multiple storage instances."""
        # Create AWS storage
        aws_auth = MagicMock()
        aws_auth.get_storage_type.return_value = "aws"
        aws_storage = MagicMock(spec=BaseStorage)

        # Create Azure storage
        azure_auth = MagicMock()
        azure_auth.get_storage_type.return_value = "azure"
        azure_storage = MagicMock(spec=BaseStorage)

        with patch.dict(
            sys.modules,
            {
                "genai_bench.storage.aws_storage": MagicMock(
                    AWSS3Storage=MagicMock(return_value=aws_storage)
                ),
                "genai_bench.storage.azure_storage": MagicMock(
                    AzureBlobStorage=MagicMock(return_value=azure_storage)
                ),
            },
        ):
            storage1 = StorageFactory.create_storage("aws", aws_auth)
            storage2 = StorageFactory.create_storage("azure", azure_auth)

            assert storage1 == aws_storage
            assert storage2 == azure_storage
            assert storage1 != storage2

    def test_import_error_handling(self):
        """Test handling of import errors for storage modules."""
        mock_auth = MagicMock()
        mock_auth.get_storage_type.return_value = "aws"

        # Create a module that raises ImportError when accessing AWSS3Storage
        mock_module = MagicMock()

        # This will raise ImportError when accessed
        def raise_import_error(*args, **kwargs):
            raise ImportError("boto3 not installed")

        # Set the AWSS3Storage to raise error when accessed
        mock_module.AWSS3Storage = raise_import_error

        with patch.dict(sys.modules, {"genai_bench.storage.aws_storage": mock_module}):
            with pytest.raises(ImportError) as exc_info:
                StorageFactory.create_storage("aws", mock_auth)

            assert "boto3 not installed" in str(exc_info.value)
