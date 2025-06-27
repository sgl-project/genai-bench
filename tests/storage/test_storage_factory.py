"""Tests for storage factory."""

from unittest.mock import MagicMock, patch

import pytest

from genai_bench.storage.factory import StorageFactory


class TestStorageFactory:
    """Test storage factory."""

    def test_create_storage_oci(self):
        """Test creating OCI storage."""
        mock_auth = MagicMock()
        mock_auth.get_storage_type.return_value = "oci"

        with patch("genai_bench.storage.oci_storage.OCIObjectStorage") as mock_storage:
            storage_instance = MagicMock()
            mock_storage.return_value = storage_instance

            result = StorageFactory.create_storage("oci", mock_auth)

            mock_storage.assert_called_once_with(mock_auth)
            assert result == storage_instance

    def test_create_storage_aws(self):
        """Test creating AWS storage."""
        mock_auth = MagicMock()
        mock_auth.get_storage_type.return_value = "aws"

        with patch("genai_bench.storage.aws_storage.AWSS3Storage") as mock_storage:
            storage_instance = MagicMock()
            mock_storage.return_value = storage_instance

            result = StorageFactory.create_storage("aws", mock_auth)

            mock_storage.assert_called_once_with(mock_auth)
            assert result == storage_instance

    def test_create_storage_azure(self):
        """Test creating Azure storage."""
        mock_auth = MagicMock()
        mock_auth.get_storage_type.return_value = "azure"

        with patch(
            "genai_bench.storage.azure_storage.AzureBlobStorage"
        ) as mock_storage:
            storage_instance = MagicMock()
            mock_storage.return_value = storage_instance

            result = StorageFactory.create_storage("azure", mock_auth)

            mock_storage.assert_called_once_with(mock_auth)
            assert result == storage_instance

    def test_create_storage_gcp(self):
        """Test creating GCP storage."""
        mock_auth = MagicMock()
        mock_auth.get_storage_type.return_value = "gcp"

        with patch("genai_bench.storage.gcp_storage.GCPCloudStorage") as mock_storage:
            storage_instance = MagicMock()
            mock_storage.return_value = storage_instance

            result = StorageFactory.create_storage("gcp", mock_auth)

            mock_storage.assert_called_once_with(mock_auth)
            assert result == storage_instance

    def test_create_storage_github(self):
        """Test creating GitHub storage."""
        mock_auth = MagicMock()
        mock_auth.get_storage_type.return_value = "github"

        with patch("genai_bench.storage.github_storage.GitHubStorage") as mock_storage:
            storage_instance = MagicMock()
            mock_storage.return_value = storage_instance

            result = StorageFactory.create_storage("github", mock_auth)

            mock_storage.assert_called_once_with(mock_auth)
            assert result == storage_instance

    def test_create_storage_unsupported(self):
        """Test creating unsupported storage type."""
        mock_auth = MagicMock()
        mock_auth.get_storage_type.return_value = "unsupported"

        with pytest.raises(
            ValueError, match="Unsupported storage provider: unsupported"
        ):
            StorageFactory.create_storage("unsupported", mock_auth)

    def test_create_storage_provider_mismatch(self):
        """Test creating storage with mismatched provider and auth type."""
        mock_auth = MagicMock()
        mock_auth.get_storage_type.return_value = "aws"

        with pytest.raises(
            ValueError, match="Storage provider 'oci' does not match auth type 'aws'"
        ):
            StorageFactory.create_storage("oci", mock_auth)
