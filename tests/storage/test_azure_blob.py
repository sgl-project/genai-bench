"""Unit tests for Azure Blob Storage implementation."""

from unittest.mock import MagicMock, mock_open, patch

import pytest

from genai_bench.auth.storage_auth_provider import StorageAuthProvider


class TestAzureBlobStorage:
    """Test Azure Blob Storage implementation."""

    @pytest.fixture
    def mock_auth(self):
        """Create mock auth provider."""
        auth = MagicMock(spec=StorageAuthProvider)
        auth.get_storage_type.return_value = "azure"
        auth.get_client_config.return_value = {
            "account_name": "testaccount",
            "account_key": "test_key",
        }
        return auth

    def test_init_with_valid_auth(self, mock_auth):
        """Test initialization with valid auth provider."""
        # Mock Azure storage client
        mock_service = MagicMock()
        mock_blob_service_client = MagicMock()
        mock_blob_service_client.return_value = mock_service

        with patch.dict(
            "sys.modules",
            {
                "azure.storage.blob": MagicMock(
                    BlobServiceClient=mock_blob_service_client
                ),
                "azure.core.exceptions": MagicMock(ResourceNotFoundError=Exception),
            },
        ):
            from genai_bench.storage.azure_storage import AzureBlobStorage

            storage = AzureBlobStorage(mock_auth)

            assert storage.auth == mock_auth
            assert storage.config == mock_auth.get_client_config()
            assert storage.client is not None

            # Should create client with account URL and key
            mock_blob_service_client.assert_called_once_with(
                account_url="https://testaccount.blob.core.windows.net",
                credential="test_key",
            )

    def test_init_with_invalid_auth_type(self):
        """Test initialization with wrong auth provider type."""
        auth = MagicMock(spec=StorageAuthProvider)
        auth.get_storage_type.return_value = "aws"

        with patch.dict(
            "sys.modules",
            {"azure.storage.blob": MagicMock(), "azure.core.exceptions": MagicMock()},
        ):
            from genai_bench.storage.azure_storage import AzureBlobStorage

            with pytest.raises(ValueError, match="Auth provider must be for Azure"):
                AzureBlobStorage(auth)

    def test_init_without_azure_storage(self, mock_auth):
        """Test initialization without azure-storage-blob installed."""
        pytest.skip("Cannot test azure-storage-blob import failure when it's installed")

    def test_create_client_with_connection_string(self, mock_auth):
        """Test client creation with connection string."""
        mock_auth.get_client_config.return_value = {
            "connection_string": "DefaultEndpointsProtocol=https;AccountName=test;..."
        }

        mock_service = MagicMock()
        mock_blob_service_client = MagicMock()
        mock_blob_service_client.from_connection_string.return_value = mock_service

        with patch.dict(
            "sys.modules",
            {
                "azure.storage.blob": MagicMock(
                    BlobServiceClient=mock_blob_service_client
                ),
                "azure.core.exceptions": MagicMock(ResourceNotFoundError=Exception),
            },
        ):
            from genai_bench.storage.azure_storage import AzureBlobStorage

            AzureBlobStorage(mock_auth)

            mock_blob_service_client.from_connection_string.assert_called_once_with(
                "DefaultEndpointsProtocol=https;AccountName=test;..."
            )

    def test_create_client_with_sas_token(self, mock_auth):
        """Test client creation with SAS token."""
        mock_auth.get_client_config.return_value = {
            "account_name": "testaccount",
            "sas_token": "?sv=2020-08-04&ss=b&srt=sco...",
        }

        mock_blob_service_client = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "azure.storage.blob": MagicMock(
                    BlobServiceClient=mock_blob_service_client
                ),
                "azure.core.exceptions": MagicMock(ResourceNotFoundError=Exception),
            },
        ):
            from genai_bench.storage.azure_storage import AzureBlobStorage

            AzureBlobStorage(mock_auth)

            mock_blob_service_client.assert_called_once_with(
                account_url="https://testaccount.blob.core.windows.net",
                credential="?sv=2020-08-04&ss=b&srt=sco...",
            )

    def test_upload_file_success(self, mock_auth):
        """Test successful file upload."""
        # Mock Azure clients
        mock_service = MagicMock()
        mock_container = MagicMock()
        mock_blob = MagicMock()

        mock_service.get_container_client.return_value = mock_container
        mock_container.get_blob_client.return_value = mock_blob

        mock_blob_service_client = MagicMock(return_value=mock_service)

        with patch.dict(
            "sys.modules",
            {
                "azure.storage.blob": MagicMock(
                    BlobServiceClient=mock_blob_service_client
                ),
                "azure.core.exceptions": MagicMock(ResourceNotFoundError=Exception),
            },
        ):
            from genai_bench.storage.azure_storage import AzureBlobStorage

            storage = AzureBlobStorage(mock_auth)

            # Mock file operations
            mock_file_data = b"test data"
            with patch("pathlib.Path.exists", return_value=True):
                with patch("builtins.open", mock_open(read_data=mock_file_data)):
                    storage.upload_file(
                        "/local/file.txt", "remote/file.txt", "test-container"
                    )

                    # Verify interactions
                    mock_service.get_container_client.assert_called_once_with(
                        "test-container"
                    )
                    mock_container.get_blob_client.assert_called_once_with(
                        "remote/file.txt"
                    )
                    mock_blob.upload_blob.assert_called_once()

    def test_upload_file_not_found(self, mock_auth):
        """Test upload with non-existent file."""
        mock_blob_service_client = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "azure.storage.blob": MagicMock(
                    BlobServiceClient=mock_blob_service_client
                ),
                "azure.core.exceptions": MagicMock(ResourceNotFoundError=Exception),
            },
        ):
            from genai_bench.storage.azure_storage import AzureBlobStorage

            storage = AzureBlobStorage(mock_auth)

            with patch("pathlib.Path.exists", return_value=False):
                with pytest.raises(FileNotFoundError, match="Local file not found"):
                    storage.upload_file(
                        "/local/missing.txt", "remote/file.txt", "test-container"
                    )

    def test_download_file_success(self, mock_auth):
        """Test successful file download."""
        # Mock Azure clients
        mock_service = MagicMock()
        mock_container = MagicMock()
        mock_blob = MagicMock()

        mock_service.get_container_client.return_value = mock_container
        mock_container.get_blob_client.return_value = mock_blob

        # Mock download stream
        mock_stream = MagicMock()
        mock_stream.readall.return_value = b"downloaded data"
        mock_blob.download_blob.return_value = mock_stream

        mock_blob_service_client = MagicMock(return_value=mock_service)

        with patch.dict(
            "sys.modules",
            {
                "azure.storage.blob": MagicMock(
                    BlobServiceClient=mock_blob_service_client
                ),
                "azure.core.exceptions": MagicMock(ResourceNotFoundError=Exception),
            },
        ):
            from genai_bench.storage.azure_storage import AzureBlobStorage

            storage = AzureBlobStorage(mock_auth)

            with patch("pathlib.Path.mkdir"):
                with patch("builtins.open", mock_open()) as mock_file:
                    storage.download_file(
                        "remote/file.txt", "/local/file.txt", "test-container"
                    )

                    # Verify interactions
                    mock_service.get_container_client.assert_called_once_with(
                        "test-container"
                    )
                    mock_container.get_blob_client.assert_called_once_with(
                        "remote/file.txt"
                    )
                    mock_blob.download_blob.assert_called_once()
                    mock_file().write.assert_called_once_with(b"downloaded data")

    def test_list_objects_success(self, mock_auth):
        """Test successful object listing."""
        # Mock Azure clients
        mock_service = MagicMock()
        mock_container = MagicMock()

        mock_service.get_container_client.return_value = mock_container

        # Mock blob list
        blob1 = MagicMock()
        blob1.name = "file1.txt"
        blob2 = MagicMock()
        blob2.name = "file2.txt"
        blob3 = MagicMock()
        blob3.name = "dir/file3.txt"
        mock_blobs = [blob1, blob2, blob3]
        mock_container.list_blobs.return_value = mock_blobs

        mock_blob_service_client = MagicMock(return_value=mock_service)

        with patch.dict(
            "sys.modules",
            {
                "azure.storage.blob": MagicMock(
                    BlobServiceClient=mock_blob_service_client
                ),
                "azure.core.exceptions": MagicMock(ResourceNotFoundError=Exception),
            },
        ):
            from genai_bench.storage.azure_storage import AzureBlobStorage

            storage = AzureBlobStorage(mock_auth)

            # List objects
            objects = list(storage.list_objects("test-container", prefix="data/"))

            assert objects == ["file1.txt", "file2.txt", "dir/file3.txt"]
            mock_service.get_container_client.assert_called_once_with("test-container")
            mock_container.list_blobs.assert_called_once_with(name_starts_with="data/")

    def test_delete_object_success(self, mock_auth):
        """Test successful object deletion."""
        # Mock Azure clients
        mock_service = MagicMock()
        mock_container = MagicMock()
        mock_blob = MagicMock()

        mock_service.get_container_client.return_value = mock_container
        mock_container.get_blob_client.return_value = mock_blob

        mock_blob_service_client = MagicMock(return_value=mock_service)

        with patch.dict(
            "sys.modules",
            {
                "azure.storage.blob": MagicMock(
                    BlobServiceClient=mock_blob_service_client
                ),
                "azure.core.exceptions": MagicMock(ResourceNotFoundError=Exception),
            },
        ):
            from genai_bench.storage.azure_storage import AzureBlobStorage

            storage = AzureBlobStorage(mock_auth)

            storage.delete_object("remote/file.txt", "test-container")

            mock_service.get_container_client.assert_called_once_with("test-container")
            mock_container.get_blob_client.assert_called_once_with("remote/file.txt")
            mock_blob.delete_blob.assert_called_once()

    def test_get_storage_type(self, mock_auth):
        """Test storage type getter."""
        mock_blob_service_client = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "azure.storage.blob": MagicMock(
                    BlobServiceClient=mock_blob_service_client
                ),
                "azure.core.exceptions": MagicMock(ResourceNotFoundError=Exception),
            },
        ):
            from genai_bench.storage.azure_storage import AzureBlobStorage

            storage = AzureBlobStorage(mock_auth)
            assert storage.get_storage_type() == "azure"
