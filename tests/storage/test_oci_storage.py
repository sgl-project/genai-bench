"""Tests for OCI Object Storage implementation."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from genai_bench.auth.oci.storage_auth_adapter import OCIStorageAuthAdapter


@pytest.fixture(autouse=True)
def mock_oci_imports():
    """Mock OCI imports to avoid SDK dependencies."""
    # Mock the OCI ObjectStorageClient to prevent config validation
    with patch("genai_bench.storage.oci_object_storage.os_datastore.ObjectStorageClient") as mock_client_class:
        mock_client_instance = MagicMock()
        mock_client_class.return_value = mock_client_instance
        
        with patch("genai_bench.storage.oci_storage.ObjectURI") as mock_uri:
            # Make it return a mock instance when instantiated
            mock_instance = MagicMock()
            mock_uri.return_value = mock_instance
            # Store class reference for inspection
            mock_uri._last_instance = mock_instance

            with patch("genai_bench.storage.oci_storage.OSDataStore") as mock_ds:
                mock_ds_instance = MagicMock()
                mock_ds_instance.get_namespace.return_value = "test-namespace"
                mock_ds.return_value = mock_ds_instance

                yield {
                    "object_uri": mock_uri,
                    "datastore_class": mock_ds,
                    "datastore": mock_ds_instance,
                    "oci_client_class": mock_client_class,
                    "oci_client": mock_client_instance,
                }


class TestOCIObjectStorage:
    """Test OCI Object Storage implementation."""

    @pytest.fixture
    def mock_oci_auth(self):
        """Create mock OCI auth provider."""
        auth = MagicMock()
        auth.get_region.return_value = "us-phoenix-1"
        return auth

    @pytest.fixture
    def mock_auth(self, mock_oci_auth):
        """Create mock auth adapter."""
        auth = MagicMock(spec=OCIStorageAuthAdapter)
        auth.get_storage_type.return_value = "oci"
        auth.get_credentials.return_value = mock_oci_auth
        auth.get_region.return_value = "us-phoenix-1"
        return auth

    @pytest.fixture
    def storage(self, mock_auth, mock_oci_imports):
        """Create OCI storage instance."""
        from genai_bench.storage.oci_storage import OCIObjectStorage

        return OCIObjectStorage(mock_auth)

    def test_init_with_valid_auth(self, mock_auth, mock_oci_imports):
        """Test initialization with valid auth provider."""
        from genai_bench.storage.oci_storage import OCIObjectStorage

        storage = OCIObjectStorage(mock_auth)

        assert storage.auth == mock_auth
        assert storage.datastore == mock_oci_imports["datastore"]
        assert storage.namespace == "test-namespace"

    def test_init_with_invalid_auth_type(self, mock_oci_imports):
        """Test initialization with wrong auth provider type."""
        from genai_bench.storage.oci_storage import OCIObjectStorage

        auth = MagicMock()
        auth.get_storage_type.return_value = "aws"

        with pytest.raises(ValueError, match="Auth provider must be for OCI"):
            OCIObjectStorage(auth)

    def test_init_with_namespace_provided(self, mock_auth, mock_oci_imports):
        """Test initialization with namespace in kwargs."""
        from genai_bench.storage.oci_storage import OCIObjectStorage

        storage = OCIObjectStorage(mock_auth, namespace="custom-namespace")

        assert storage.namespace == "custom-namespace"
        # Should not call get_namespace when namespace is provided
        mock_oci_imports["datastore"].get_namespace.assert_not_called()

    def test_upload_file_success(self, storage, mock_oci_imports):
        """Test successful file upload."""
        with patch("pathlib.Path.exists", return_value=True):
            storage.upload_file("local.txt", "remote.txt", "test-bucket")

            # Verify datastore upload was called
            mock_oci_imports["datastore"].upload.assert_called_once()
            call_args = mock_oci_imports["datastore"].upload.call_args
            assert call_args[0][0] == "local.txt"  # local path

            # Check ObjectURI was created with correct params
            mock_oci_imports["object_uri"].assert_called_with(
                namespace="test-namespace",
                bucket_name="test-bucket",
                object_name="remote.txt",
                region="us-phoenix-1",
                prefix=None,
            )

    def test_upload_file_not_found(self, storage):
        """Test upload with non-existent file."""
        with patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(FileNotFoundError, match="Local file not found"):
                storage.upload_file("missing.txt", "remote.txt", "bucket")

    def test_upload_folder_success(self, storage, mock_oci_imports):
        """Test successful folder upload."""
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.is_dir", return_value=True):
                # Create mock files
                mock_file1 = MagicMock()
                mock_file1.is_file.return_value = True
                mock_file1.relative_to.return_value = Path("file1.txt")

                mock_file2 = MagicMock()
                mock_file2.is_file.return_value = True
                mock_file2.relative_to.return_value = Path("subdir/file2.txt")

                mock_dir = MagicMock()
                mock_dir.is_file.return_value = False

                mock_files = [mock_file1, mock_file2, mock_dir]

                with patch("pathlib.Path.rglob", return_value=mock_files):
                    storage.upload_folder("local_folder", "test-bucket", prefix="data")

                    # Should upload 2 files (not the directory)
                    assert mock_oci_imports["datastore"].upload.call_count == 2

    def test_upload_folder_not_found(self, storage):
        """Test upload folder that doesn't exist."""
        with patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(ValueError, match="Local folder not found"):
                storage.upload_folder("missing_folder", "bucket")

    def test_upload_folder_not_directory(self, storage):
        """Test upload folder with file path."""
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.is_dir", return_value=False):
                with pytest.raises(ValueError, match="not a directory"):
                    storage.upload_folder("file.txt", "bucket")

    def test_upload_folder_no_prefix(self, storage, mock_oci_imports):
        """Test folder upload without prefix."""
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.is_dir", return_value=True):
                mock_file = MagicMock()
                mock_file.is_file.return_value = True
                mock_file.relative_to.return_value = Path("file.txt")

                with patch("pathlib.Path.rglob", return_value=[mock_file]):
                    storage.upload_folder("local_folder", "test-bucket")

                    # Check that ObjectURI was created with no prefix in path
                    mock_oci_imports["object_uri"].assert_called_with(
                        namespace="test-namespace",
                        bucket_name="test-bucket",
                        object_name="file.txt",
                        region="us-phoenix-1",
                        prefix=None,
                    )

    def test_download_file_success(self, storage, mock_oci_imports):
        """Test successful file download."""
        with patch("pathlib.Path.mkdir"):
            storage.download_file("remote.txt", "local.txt", "test-bucket")

            # Verify datastore download was called
            mock_oci_imports["datastore"].download.assert_called_once()

            # Check ObjectURI was created correctly
            mock_oci_imports["object_uri"].assert_called_with(
                namespace="test-namespace",
                bucket_name="test-bucket",
                object_name="remote.txt",
                region="us-phoenix-1",
                prefix=None,
            )

    def test_list_objects_success(self, storage, mock_oci_imports):
        """Test successful object listing."""
        mock_oci_imports["datastore"].list_objects.return_value = [
            "file1.txt",
            "file2.txt",
            "file3.txt",
        ]

        objects = list(storage.list_objects("test-bucket"))

        assert objects == ["file1.txt", "file2.txt", "file3.txt"]

        # Check ObjectURI was created correctly
        mock_oci_imports["object_uri"].assert_called_with(
            namespace="test-namespace",
            bucket_name="test-bucket",
            object_name="",
            region="us-phoenix-1",
            prefix=None,
        )

    def test_list_objects_with_prefix(self, storage, mock_oci_imports):
        """Test object listing with prefix."""
        mock_oci_imports["datastore"].list_objects.return_value = [
            "data/file1.txt",
            "data/file2.txt",
        ]

        objects = list(storage.list_objects("test-bucket", prefix="data"))

        assert objects == ["data/file1.txt", "data/file2.txt"]

        # Check ObjectURI was created with prefix
        mock_oci_imports["object_uri"].assert_called_with(
            namespace="test-namespace",
            bucket_name="test-bucket",
            object_name="data",
            region="us-phoenix-1",
            prefix="data",
        )

    def test_delete_object_success(self, storage, mock_oci_imports):
        """Test successful object deletion."""
        storage.delete_object("file.txt", "test-bucket")

        # Verify datastore delete was called
        mock_oci_imports["datastore"].delete_object.assert_called_once()

        # Check ObjectURI was created correctly
        mock_oci_imports["object_uri"].assert_called_with(
            namespace="test-namespace",
            bucket_name="test-bucket",
            object_name="file.txt",
            region="us-phoenix-1",
            prefix=None,
        )

    def test_get_storage_type(self, storage):
        """Test storage type getter."""
        assert storage.get_storage_type() == "oci"

    def test_no_region(self, mock_auth, mock_oci_imports):
        """Test when auth provider returns None for region."""
        from genai_bench.storage.oci_storage import OCIObjectStorage

        mock_auth.get_region.return_value = None
        storage = OCIObjectStorage(mock_auth)

        with patch("pathlib.Path.exists", return_value=True):
            storage.upload_file("local.txt", "remote.txt", "test-bucket")

            # Check that ObjectURI was created with None region
            mock_oci_imports["object_uri"].assert_called_with(
                namespace="test-namespace",
                bucket_name="test-bucket",
                object_name="remote.txt",
                region=None,
                prefix=None,
            )

    def test_empty_namespace(self, mock_auth, mock_oci_imports):
        """Test when namespace is empty string."""
        from genai_bench.storage.oci_storage import OCIObjectStorage

        # Override the autouse fixture for this test
        with patch("genai_bench.storage.oci_storage.OSDataStore") as mock_ds_class:
            mock_ds_instance = MagicMock()
            mock_ds_instance.get_namespace.return_value = ""
            mock_ds_class.return_value = mock_ds_instance

            storage = OCIObjectStorage(mock_auth)
            assert storage.namespace == ""

    def test_upload_file_with_region(self, mock_auth, mock_oci_imports):
        """Test upload file with region from auth."""
        from genai_bench.storage.oci_storage import OCIObjectStorage

        storage = OCIObjectStorage(mock_auth)

        with patch("pathlib.Path.exists", return_value=True):
            storage.upload_file("local.txt", "remote.txt", "test-bucket")

            # Check ObjectURI was created with region
            mock_oci_imports["object_uri"].assert_called_with(
                namespace="test-namespace",
                bucket_name="test-bucket",
                object_name="remote.txt",
                region="us-phoenix-1",
                prefix=None,
            )
