"""Tests for AWS S3 storage implementation."""

from unittest.mock import MagicMock, patch

import pytest

from genai_bench.auth.aws.s3_auth import AWSS3Auth
from genai_bench.storage.aws_storage import AWSS3Storage


# Mock the exception
class NoCredentialsError(Exception):
    """Mock NoCredentialsError."""

    pass


class TestAWSS3Storage:
    """Test AWS S3 storage implementation."""

    @pytest.fixture
    def mock_auth(self):
        """Create mock auth provider."""
        auth = MagicMock(spec=AWSS3Auth)
        auth.get_storage_type.return_value = "aws"
        auth.get_client_config.return_value = {
            "aws_access_key_id": "test_key",
            "aws_secret_access_key": "test_secret",
            "region_name": "us-east-1",
        }
        return auth

    @pytest.fixture
    def mock_s3_client(self):
        """Create mock S3 client."""
        with patch("boto3.client") as mock_client:
            client_instance = MagicMock()
            mock_client.return_value = client_instance
            yield client_instance

    @pytest.fixture
    def storage(self, mock_auth, mock_s3_client):
        """Create AWS S3 storage instance."""
        storage = AWSS3Storage(mock_auth)
        return storage

    def test_init(self, mock_auth):
        """Test initialization."""
        with patch("boto3.client"):
            storage = AWSS3Storage(mock_auth)
            assert storage.auth == mock_auth
            assert storage.client is not None  # Client is created on init

    @patch("boto3.client")
    def test_authenticate_with_credentials(self, mock_client, mock_auth):
        """Test authentication with credentials."""
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance

        storage = AWSS3Storage(mock_auth)
        # No authenticate method in new implementation

        mock_client.assert_called_once_with(
            "s3",
            aws_access_key_id="test_key",
            aws_secret_access_key="test_secret",
            region_name="us-east-1",
        )
        assert storage.client == mock_client_instance

    @patch("boto3.client")
    def test_authenticate_with_profile(self, mock_client, mock_auth):
        """Test authentication with profile."""
        mock_auth.get_client_config.return_value = {
            "profile_name": "test-profile",
            "region_name": "us-west-2",
        }

        with patch("boto3.Session") as mock_session:
            mock_session_instance = MagicMock()
            mock_session.return_value = mock_session_instance
            mock_session_instance.client.return_value = MagicMock()

            AWSS3Storage(mock_auth)
            # No authenticate method in new implementation

            mock_session.assert_called_once_with(profile_name="test-profile")

    def test_upload_file(self, storage, mock_s3_client):
        """Test uploading a file."""
        # No authenticate method in new implementation

        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.is_file", return_value=True):
                with patch("pathlib.Path.stat") as mock_stat:
                    mock_stat.return_value.st_size = 1000  # 1KB file

                    storage.upload_file("local.txt", "remote.txt", "test-bucket")

                    mock_s3_client.upload_file.assert_called_once_with(
                        "local.txt", "test-bucket", "remote.txt", ExtraArgs={}
                    )

    def test_upload_file_not_found(self, storage):
        """Test uploading non-existent file."""
        # No authenticate method in new implementation

        with patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(FileNotFoundError, match="Local file not found"):
                storage.upload_file("missing.txt", "remote.txt", "bucket")

    def test_upload_large_file(self, storage, mock_s3_client):
        """Test uploading a large file triggers multipart upload."""
        # No authenticate method in new implementation

        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.is_file", return_value=True):
                with patch("pathlib.Path.stat") as mock_stat:
                    mock_stat.return_value.st_size = 200 * 1024 * 1024  # 200MB file

                    storage.upload_file("large.txt", "remote.txt", "test-bucket")

                    # Should use multipart upload
                    assert mock_s3_client.upload_file.call_count == 1
                    call_args = mock_s3_client.upload_file.call_args
                    assert "Config" in call_args[1]

    def test_upload_file_no_credentials(self, storage, mock_s3_client):
        """Test upload file with no credentials."""
        # No authenticate method in new implementation
        # Use the storage's reference to NoCredentialsError
        mock_s3_client.upload_file.side_effect = storage.NoCredentialsError()

        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.is_file", return_value=True):
                with patch("pathlib.Path.stat") as mock_stat:
                    mock_stat.return_value.st_size = 1000

                    with pytest.raises(ValueError, match="AWS credentials not found"):
                        storage.upload_file("local.txt", "remote.txt", "bucket")

    def test_upload_folder(self, storage, mock_s3_client):
        """Test uploading a folder."""
        # No authenticate method in new implementation

        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.is_dir", return_value=True):
                mock_files = [
                    MagicMock(is_file=lambda: True, __str__=lambda x: "file1.txt"),
                    MagicMock(is_file=lambda: True, __str__=lambda x: "file2.txt"),
                    MagicMock(is_file=lambda: False),  # Directory
                ]

                with patch("pathlib.Path.rglob", return_value=mock_files):
                    with patch("pathlib.Path.stat") as mock_stat:
                        mock_stat.return_value.st_size = 1000

                        storage.upload_folder("local_folder", "bucket", "prefix/")

                        # Should upload only files, not directories
                        assert mock_s3_client.upload_file.call_count == 2

    def test_upload_folder_not_found(self, storage):
        """Test uploading non-existent folder."""
        # No authenticate method in new implementation

        with patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(ValueError, match="Local folder not found"):
                storage.upload_folder("missing_folder", "bucket")

    def test_download_file(self, storage, mock_s3_client):
        """Test downloading a file."""
        # No authenticate method in new implementation

        with patch("pathlib.Path.mkdir"):
            storage.download_file("remote.txt", "local.txt", "bucket")

            mock_s3_client.download_file.assert_called_once_with(
                "bucket", "remote.txt", "local.txt"
            )

    def test_download_file_error(self, storage, mock_s3_client):
        """Test download file error handling."""
        # No authenticate method in new implementation
        mock_s3_client.download_file.side_effect = Exception("Download failed")

        with pytest.raises(Exception, match="Download failed"):
            storage.download_file("remote.txt", "local.txt", "bucket")

    def test_list_objects(self, storage, mock_s3_client):
        """Test listing objects."""
        # No authenticate method in new implementation

        # Mock paginator
        mock_paginator = MagicMock()
        mock_s3_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {"Contents": [{"Key": "file1.txt"}, {"Key": "file2.txt"}]},
            {"Contents": [{"Key": "file3.txt"}]},
        ]

        objects = list(storage.list_objects("bucket", "prefix/"))

        assert objects == ["file1.txt", "file2.txt", "file3.txt"]
        mock_paginator.paginate.assert_called_once_with(
            Bucket="bucket", Prefix="prefix/"
        )

    def test_list_objects_empty(self, storage, mock_s3_client):
        """Test listing objects with no results."""
        # No authenticate method in new implementation

        mock_paginator = MagicMock()
        mock_s3_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [{}]  # No Contents key

        objects = list(storage.list_objects("bucket"))
        assert objects == []

    def test_delete_object(self, storage, mock_s3_client):
        """Test deleting an object."""
        # No authenticate method in new implementation

        storage.delete_object("file.txt", "bucket")

        mock_s3_client.delete_object.assert_called_once_with(
            Bucket="bucket", Key="file.txt"
        )

    def test_delete_object_error(self, storage, mock_s3_client):
        """Test delete object error handling."""
        # No authenticate method in new implementation
        mock_s3_client.delete_object.side_effect = Exception("Delete failed")

        with pytest.raises(Exception, match="Delete failed"):
            storage.delete_object("file.txt", "bucket")

    def test_get_storage_type(self, storage):
        """Test getting storage type."""
        assert storage.get_storage_type() == "aws"

    def test_init_import_error(self, mock_auth):
        """Test initialization with boto3 import error."""
        with patch.dict("sys.modules", {"boto3": None}):
            with pytest.raises(ImportError, match="boto3 is required"):
                AWSS3Storage(mock_auth)

    @patch("boto3.client")
    def test_init_with_session_token(self, mock_client, mock_auth):
        """Test initialization with session token."""
        mock_auth.get_client_config.return_value = {
            "aws_access_key_id": "test_key",
            "aws_secret_access_key": "test_secret",
            "aws_session_token": "test_token",
            "region_name": "us-east-1",
        }

        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance

        AWSS3Storage(mock_auth)

        mock_client.assert_called_once_with(
            "s3",
            aws_access_key_id="test_key",
            aws_secret_access_key="test_secret",
            aws_session_token="test_token",
            region_name="us-east-1",
        )

    @patch("boto3.client")
    def test_upload_file_general_exception(self, mock_client, mock_auth):
        """Test upload file with general exception."""
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.upload_file.side_effect = Exception("Upload error")

        storage = AWSS3Storage(mock_auth)

        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.is_file", return_value=True):
                with patch("pathlib.Path.stat") as mock_stat:
                    mock_stat.return_value.st_size = 1000

                    with pytest.raises(Exception, match="Upload error"):
                        storage.upload_file("local.txt", "remote.txt", "bucket")

    @patch("boto3.client")
    def test_download_file_no_credentials(self, mock_client, mock_auth):
        """Test download file with no credentials."""
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance

        storage = AWSS3Storage(mock_auth)
        mock_client_instance.download_file.side_effect = storage.NoCredentialsError()

        with pytest.raises(ValueError, match="AWS credentials not found"):
            storage.download_file("remote.txt", "local.txt", "bucket")

    @patch("boto3.client")
    def test_list_objects_no_credentials(self, mock_client, mock_auth):
        """Test list objects with no credentials."""
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance

        storage = AWSS3Storage(mock_auth)

        mock_paginator = MagicMock()
        mock_client_instance.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.side_effect = storage.NoCredentialsError()

        with pytest.raises(ValueError, match="AWS credentials not found"):
            list(storage.list_objects("bucket"))

    @patch("boto3.client")
    def test_list_objects_general_exception(self, mock_client, mock_auth):
        """Test list objects with general exception."""
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance

        storage = AWSS3Storage(mock_auth)

        mock_paginator = MagicMock()
        mock_client_instance.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.side_effect = Exception("List error")

        with pytest.raises(Exception, match="List error"):
            list(storage.list_objects("bucket"))

    @patch("boto3.client")
    def test_delete_object_no_credentials(self, mock_client, mock_auth):
        """Test delete object with no credentials."""
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance

        storage = AWSS3Storage(mock_auth)
        mock_client_instance.delete_object.side_effect = storage.NoCredentialsError()

        with pytest.raises(ValueError, match="AWS credentials not found"):
            storage.delete_object("file.txt", "bucket")
