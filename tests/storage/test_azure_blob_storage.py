"""Tests for Azure Blob storage implementation."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from genai_bench.auth.azure.blob_auth import AzureBlobAuth
from genai_bench.storage.azure_storage import AzureBlobStorage


# Mock the exception
class AzureError(Exception):
    """Mock AzureError."""

    pass


class TestAzureBlobStorage:
    """Test Azure Blob storage implementation."""

    @pytest.fixture
    def mock_auth(self):
        """Create mock auth provider."""
        auth = MagicMock(spec=AzureBlobAuth)
        auth.get_storage_type.return_value = "azure"
        auth.get_client_config.return_value = {
            "account_name": "testaccount",
            "account_key": "testkey",
        }
        return auth

    @pytest.fixture
    def mock_blob_service_client(self):
        """Create mock blob service client."""
        with patch("azure.storage.blob.BlobServiceClient") as mock_client:
            client_instance = MagicMock()
            mock_client.from_connection_string.return_value = client_instance
            mock_client.return_value = client_instance
            yield mock_client, client_instance

    @pytest.fixture
    def storage(self, mock_auth):
        """Create Azure Blob storage instance."""
        with patch("azure.storage.blob.BlobServiceClient") as mock_client:
            client_instance = MagicMock()
            mock_client.return_value = client_instance
            storage = AzureBlobStorage(mock_auth)
            return storage

    def test_init(self, mock_auth):
        """Test initialization."""
        with patch("azure.storage.blob.BlobServiceClient"):
            storage = AzureBlobStorage(mock_auth)
            assert storage.auth == mock_auth
            assert storage.client is not None  # Client is created on init

    def test_authenticate_with_connection_string(self, mock_auth):
        """Test initialization with connection string."""
        mock_auth.get_client_config.return_value = {
            "connection_string": "DefaultEndpointsProtocol=https;AccountName=test"
        }

        with patch("azure.storage.blob.BlobServiceClient") as mock_client_class:
            mock_client_instance = MagicMock()
            mock_client_class.from_connection_string.return_value = mock_client_instance

            storage = AzureBlobStorage(mock_auth)

            mock_client_class.from_connection_string.assert_called_once_with(
                "DefaultEndpointsProtocol=https;AccountName=test"
            )
            assert storage.client == mock_client_instance

    def test_authenticate_with_account_key(self, mock_auth):
        """Test initialization with account key."""
        with patch("azure.storage.blob.BlobServiceClient") as mock_client_class:
            mock_client_instance = MagicMock()
            mock_client_class.return_value = mock_client_instance

            storage = AzureBlobStorage(mock_auth)

            mock_client_class.assert_called_once_with(
                account_url="https://testaccount.blob.core.windows.net",
                credential="testkey",
            )
            assert storage.client == mock_client_instance

    def test_authenticate_with_sas_token(self, mock_auth):
        """Test initialization with SAS token."""
        mock_auth.get_client_config.return_value = {
            "account_name": "testaccount",
            "sas_token": "?sv=2020-08-04",
        }

        with patch("azure.storage.blob.BlobServiceClient") as mock_client_class:
            mock_client_instance = MagicMock()
            mock_client_class.return_value = mock_client_instance

            AzureBlobStorage(mock_auth)

            mock_client_class.assert_called_once_with(
                account_url="https://testaccount.blob.core.windows.net",
                credential="?sv=2020-08-04",
            )

    def test_authenticate_with_azure_ad(self, mock_auth):
        """Test initialization with Azure AD."""
        mock_auth.get_client_config.return_value = {
            "account_name": "testaccount",
            "use_azure_ad": True,
            "tenant_id": "tenant",
            "client_id": "client",
            "client_secret": "secret",
        }

        with patch("azure.identity.ClientSecretCredential") as mock_secret_cred:
            mock_secret_cred_instance = MagicMock()
            mock_secret_cred.return_value = mock_secret_cred_instance

            with patch("azure.storage.blob.BlobServiceClient") as mock_client_class:
                mock_client_instance = MagicMock()
                mock_client_class.return_value = mock_client_instance

                AzureBlobStorage(mock_auth)

                mock_secret_cred.assert_called_once_with(
                    tenant_id="tenant", client_id="client", client_secret="secret"
                )
                mock_client_class.assert_called_once_with(
                    account_url="https://testaccount.blob.core.windows.net",
                    credential=mock_secret_cred_instance,
                )

    def test_authenticate_missing_config(self, mock_auth):
        """Test initialization with missing config."""
        mock_auth.get_client_config.return_value = {}

        with patch("azure.storage.blob.BlobServiceClient"):
            with pytest.raises(
                ValueError, match="Azure account name or connection string required"
            ):
                AzureBlobStorage(mock_auth)

    def test_upload_file(self, mock_auth):
        """Test uploading a file."""
        with patch("azure.storage.blob.BlobServiceClient") as mock_client_class:
            mock_client_instance = MagicMock()
            mock_client_class.return_value = mock_client_instance

            mock_container_client = MagicMock()
            mock_blob_client = MagicMock()
            mock_client_instance.get_container_client.return_value = (
                mock_container_client
            )
            mock_container_client.get_blob_client.return_value = mock_blob_client

            storage = AzureBlobStorage(mock_auth)

            with patch("pathlib.Path.exists", return_value=True):
                with patch("pathlib.Path.is_file", return_value=True):
                    with patch("builtins.open", create=True) as mock_open:
                        mock_file = MagicMock()
                        mock_open.return_value.__enter__.return_value = mock_file

                        storage.upload_file("local.txt", "remote.txt", "container")

                        mock_client_instance.get_container_client.assert_called_once_with(
                            "container"
                        )
                        mock_container_client.get_blob_client.assert_called_once_with(
                            "remote.txt"
                        )
                        mock_blob_client.upload_blob.assert_called_once()

    def test_upload_file_not_found(self, mock_auth):
        """Test uploading non-existent file."""
        with patch("azure.storage.blob.BlobServiceClient"):
            storage = AzureBlobStorage(mock_auth)

            with patch("pathlib.Path.exists", return_value=False):
                with pytest.raises(FileNotFoundError, match="Local file not found"):
                    storage.upload_file("missing.txt", "remote.txt", "container")

    def test_upload_folder(self, mock_auth):
        """Test uploading a folder."""
        with patch("azure.storage.blob.BlobServiceClient") as mock_client_class:
            mock_client_instance = MagicMock()
            mock_client_class.return_value = mock_client_instance

            # Mock the container client and its blob client method
            mock_container_client = MagicMock()
            mock_blob_client = MagicMock()
            mock_client_instance.get_container_client.return_value = (
                mock_container_client
            )
            mock_container_client.get_blob_client.return_value = mock_blob_client

            storage = AzureBlobStorage(mock_auth)

        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.is_dir", return_value=True):
                # Create more complete mock file objects
                mock_file1 = MagicMock()
                mock_file1.is_file.return_value = True
                mock_file1.relative_to.return_value = Path("file1.txt")

                mock_file2 = MagicMock()
                mock_file2.is_file.return_value = True
                mock_file2.relative_to.return_value = Path("file2.txt")

                mock_dir = MagicMock()
                mock_dir.is_file.return_value = False

                mock_files = [mock_file1, mock_file2, mock_dir]

                with patch("pathlib.Path.rglob", return_value=mock_files):
                    with patch("builtins.open", create=True):
                        storage.upload_folder("local_folder", "container", "prefix/")

                        # Should upload only files, not directories
                        assert mock_blob_client.upload_blob.call_count == 2

    def test_download_file(self, mock_auth):
        """Test downloading a file."""
        with patch("azure.storage.blob.BlobServiceClient") as mock_client_class:
            mock_client_instance = MagicMock()
            mock_client_class.return_value = mock_client_instance

            mock_container_client = MagicMock()
            mock_blob_client = MagicMock()
            mock_client_instance.get_container_client.return_value = (
                mock_container_client
            )
            mock_container_client.get_blob_client.return_value = mock_blob_client

            mock_downloader = MagicMock()
            mock_downloader.readall.return_value = b"content"
            mock_blob_client.download_blob.return_value = mock_downloader

            storage = AzureBlobStorage(mock_auth)

        with patch("pathlib.Path.mkdir"):
            with patch("builtins.open", create=True) as mock_open:
                storage.download_file("remote.txt", "local.txt", "container")

                mock_client_instance.get_container_client.assert_called_once_with(
                    "container"
                )
                mock_container_client.get_blob_client.assert_called_once_with(
                    "remote.txt"
                )
                mock_open.assert_called_once_with(Path("local.txt"), "wb")

    def test_list_objects(self, mock_auth):
        """Test listing objects."""
        with patch("azure.storage.blob.BlobServiceClient") as mock_client_class:
            mock_client_instance = MagicMock()
            mock_client_class.return_value = mock_client_instance

            mock_container_client = MagicMock()
            mock_client_instance.get_container_client.return_value = (
                mock_container_client
            )

            # Create proper mock blob objects
            blob1 = MagicMock()
            blob1.name = "file1.txt"
            blob2 = MagicMock()
            blob2.name = "file2.txt"
            blob3 = MagicMock()
            blob3.name = "file3.txt"

            mock_container_client.list_blobs.return_value = [blob1, blob2, blob3]

            storage = AzureBlobStorage(mock_auth)

        objects = list(storage.list_objects("container", "prefix/"))

        assert objects == ["file1.txt", "file2.txt", "file3.txt"]
        mock_container_client.list_blobs.assert_called_once_with(
            name_starts_with="prefix/"
        )

    def test_delete_object(self, mock_auth):
        """Test deleting an object."""
        with patch("azure.storage.blob.BlobServiceClient") as mock_client_class:
            mock_client_instance = MagicMock()
            mock_client_class.return_value = mock_client_instance

            mock_container_client = MagicMock()
            mock_blob_client = MagicMock()
            mock_client_instance.get_container_client.return_value = (
                mock_container_client
            )
            mock_container_client.get_blob_client.return_value = mock_blob_client

            storage = AzureBlobStorage(mock_auth)

        storage.delete_object("file.txt", "container")

        mock_client_instance.get_container_client.assert_called_once_with("container")
        mock_container_client.get_blob_client.assert_called_once_with("file.txt")
        mock_blob_client.delete_blob.assert_called_once()

    def test_get_storage_type(self, storage):
        """Test getting storage type."""
        assert storage.get_storage_type() == "azure"

    def test_upload_file_azure_error(self, mock_auth):
        """Test upload file with Azure error."""
        with patch("azure.storage.blob.BlobServiceClient") as mock_client_class:
            mock_client_instance = MagicMock()
            mock_client_class.return_value = mock_client_instance

            mock_container_client = MagicMock()
            mock_blob_client = MagicMock()
            mock_client_instance.get_container_client.return_value = (
                mock_container_client
            )
            mock_container_client.get_blob_client.return_value = mock_blob_client
            mock_blob_client.upload_blob.side_effect = Exception("Upload failed")

            storage = AzureBlobStorage(mock_auth)

        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.is_file", return_value=True):
                with patch("builtins.open", create=True):
                    with pytest.raises(Exception, match="Upload failed"):
                        storage.upload_file("local.txt", "remote.txt", "container")

    def test_authenticate_azure_ad_default_credential(self, mock_auth):
        """Test initialization with Azure AD using default credential."""
        mock_auth.get_client_config.return_value = {
            "account_name": "testaccount",
            "use_azure_ad": True,
        }

        with patch("azure.identity.DefaultAzureCredential") as mock_cred:
            mock_cred_instance = MagicMock()
            mock_cred.return_value = mock_cred_instance

            with patch("azure.storage.blob.BlobServiceClient") as mock_client_class:
                mock_client_instance = MagicMock()
                mock_client_class.return_value = mock_client_instance

                AzureBlobStorage(mock_auth)

                mock_cred.assert_called_once()
                mock_client_class.assert_called_once_with(
                    account_url="https://testaccount.blob.core.windows.net",
                    credential=mock_cred_instance,
                )

    def test_init_import_error(self, mock_auth):
        """Test initialization with azure-storage-blob import error."""
        with patch.dict("sys.modules", {"azure.storage.blob": None}):
            with pytest.raises(ImportError, match="azure-storage-blob is required"):
                AzureBlobStorage(mock_auth)

    def test_authenticate_azure_ad_import_error(self, mock_auth):
        """Test initialization with azure-identity import error."""
        mock_auth.get_client_config.return_value = {
            "account_name": "testaccount",
            "use_azure_ad": True,
        }

        with patch("azure.storage.blob.BlobServiceClient"):
            with patch.dict("sys.modules", {"azure.identity": None}):
                with pytest.raises(ImportError, match="azure-identity is required"):
                    AzureBlobStorage(mock_auth)

    def test_authenticate_missing_account_name(self, mock_auth):
        """Test initialization with missing account name."""
        mock_auth.get_client_config.return_value = {
            "account_key": "testkey",
        }

        with patch("azure.storage.blob.BlobServiceClient"):
            with pytest.raises(
                ValueError, match="Azure account name or connection string required"
            ):
                AzureBlobStorage(mock_auth)

    def test_upload_folder_not_a_directory(self, mock_auth):
        """Test uploading a file instead of folder."""
        with patch("azure.storage.blob.BlobServiceClient"):
            storage = AzureBlobStorage(mock_auth)

            with patch("pathlib.Path.exists", return_value=True):
                with patch("pathlib.Path.is_dir", return_value=False):
                    with pytest.raises(ValueError, match="not a directory"):
                        storage.upload_folder("file.txt", "container")

    def test_upload_folder_with_no_prefix(self, mock_auth):
        """Test uploading a folder without prefix."""
        with patch("azure.storage.blob.BlobServiceClient") as mock_client_class:
            mock_client_instance = MagicMock()
            mock_client_class.return_value = mock_client_instance

            mock_container_client = MagicMock()
            mock_blob_client = MagicMock()
            mock_client_instance.get_container_client.return_value = (
                mock_container_client
            )
            mock_container_client.get_blob_client.return_value = mock_blob_client

            storage = AzureBlobStorage(mock_auth)

        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.is_dir", return_value=True):
                # Create mock file object with proper methods
                mock_file = MagicMock()
                mock_file.is_file.return_value = True
                mock_file.relative_to.return_value = Path("file.txt")

                mock_files = [mock_file]

                with patch("pathlib.Path.rglob", return_value=mock_files):
                    with patch("builtins.open", create=True):
                        storage.upload_folder("local_folder", "container")

                        # Should upload with no prefix
                        assert mock_blob_client.upload_blob.call_count == 1

    def test_download_file_not_found(self, mock_auth):
        """Test downloading non-existent file."""
        with patch("azure.storage.blob.BlobServiceClient") as mock_client_class:
            mock_client_instance = MagicMock()
            mock_client_class.return_value = mock_client_instance

            mock_container_client = MagicMock()
            mock_blob_client = MagicMock()
            mock_client_instance.get_container_client.return_value = (
                mock_container_client
            )
            mock_container_client.get_blob_client.return_value = mock_blob_client

            # Import the exception class
            from azure.core.exceptions import ResourceNotFoundError

            mock_blob_client.download_blob.side_effect = ResourceNotFoundError()

            storage = AzureBlobStorage(mock_auth)

        local_file = Path("local.txt")
        try:
            with patch("pathlib.Path.mkdir"):
                with pytest.raises(FileNotFoundError, match="Blob not found"):
                    storage.download_file("missing.txt", "local.txt", "container")
        finally:
            # Clean up the created file
            if local_file.exists():
                local_file.unlink()

    def test_download_file_general_error(self, mock_auth):
        """Test download file with general error."""
        with patch("azure.storage.blob.BlobServiceClient") as mock_client_class:
            mock_client_instance = MagicMock()
            mock_client_class.return_value = mock_client_instance

            mock_container_client = MagicMock()
            mock_blob_client = MagicMock()
            mock_client_instance.get_container_client.return_value = (
                mock_container_client
            )
            mock_container_client.get_blob_client.return_value = mock_blob_client
            mock_blob_client.download_blob.side_effect = Exception("Download failed")

            storage = AzureBlobStorage(mock_auth)

        local_file = Path("local.txt")
        try:
            with patch("pathlib.Path.mkdir"):
                with pytest.raises(Exception, match="Download failed"):
                    storage.download_file("file.txt", "local.txt", "container")
        finally:
            # Clean up the created file
            if local_file.exists():
                local_file.unlink()

    def test_list_objects_error(self, mock_auth):
        """Test list objects with error."""
        with patch("azure.storage.blob.BlobServiceClient") as mock_client_class:
            mock_client_instance = MagicMock()
            mock_client_class.return_value = mock_client_instance

            mock_container_client = MagicMock()
            mock_client_instance.get_container_client.return_value = (
                mock_container_client
            )
            mock_container_client.list_blobs.side_effect = Exception("List failed")

            storage = AzureBlobStorage(mock_auth)

        with pytest.raises(Exception, match="List failed"):
            list(storage.list_objects("container"))

    def test_delete_object_error(self, mock_auth):
        """Test delete object with error."""
        with patch("azure.storage.blob.BlobServiceClient") as mock_client_class:
            mock_client_instance = MagicMock()
            mock_client_class.return_value = mock_client_instance

            mock_container_client = MagicMock()
            mock_blob_client = MagicMock()
            mock_client_instance.get_container_client.return_value = (
                mock_container_client
            )
            mock_container_client.get_blob_client.return_value = mock_blob_client
            mock_blob_client.delete_blob.side_effect = Exception("Delete failed")

            storage = AzureBlobStorage(mock_auth)

        with pytest.raises(Exception, match="Delete failed"):
            storage.delete_object("file.txt", "container")
