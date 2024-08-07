"""Test cases for CasperDataStore class."""

from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from oci.object_storage.models import ListObjects, ObjectSummary
from oci.response import Response
from oci.signer import AbstractBaseSigner

from genai_bench.auth.auth_provider import AuthProvider
from genai_bench.oci_object_storage.object_uri import ObjectURI
from genai_bench.oci_object_storage.os_datastore import OSDataStore


class MockSigner(AbstractBaseSigner):
    """Mock signer that bypasses config validation."""

    def __init__(self):
        super().__init__()
        self.api_key = "dummy_key"
        self.private_key = "dummy_private_key"
        self.passphrase = None
        self._basic_signer = self
        self._body_signer = self

    def get_signature_headers(self, *args, **kwargs):
        return {"date": "Thu, 05 Jan 2014 21:31:40 GMT", "authorization": "dummy_auth"}


class MockAuthProvider(AuthProvider):
    """Mock auth provider for testing."""

    def get_config(self):
        return {
            "region": "test-region",
            "key_file": "/path/to/key.pem",
            "user": "ocid1.user.oc1..dummy",
            "tenancy": "ocid1.tenancy.oc1..dummy",
            "fingerprint": "20:3b:97:13:55:1c:5b:0d:d3:37:d8:50:4e:c5:3a:34",
            "pass_phrase": None,
            "key_content": "dummy_key_content",
        }

    def get_auth_credentials(self):
        return MockSigner()


@pytest.fixture
def mock_client():
    """Create a mock OCI client."""
    mock = MagicMock()
    mock.base_client = MagicMock()
    mock.get_namespace = MagicMock()
    mock.get_object = MagicMock()
    mock.put_object = MagicMock()
    mock.list_objects = MagicMock()
    return mock


@pytest.fixture
@patch("oci.config.validate_config")
@patch("genai_bench.oci_object_storage.os_datastore.ObjectStorageClient")
def test_store(mock_client_class, mock_validate_config, mock_client):
    """Create a CasperDataStore with mock client."""
    mock_client_class.return_value = mock_client
    auth = MockAuthProvider()
    store = OSDataStore(auth)
    return store


@patch("oci.config.validate_config")
@patch("genai_bench.oci_object_storage.os_datastore.ObjectStorageClient")
def test_initialization(mock_client_class, mock_validate_config, mock_client):
    """Test CasperDataStore initialization."""
    # Setup test data
    mock_client_class.return_value = mock_client
    auth = MockAuthProvider()
    config = auth.get_config()  # Get actual config from auth provider

    # Initialize store
    store = OSDataStore(auth)

    # Verify store attributes
    assert store.auth == auth
    assert store.config == config
    assert store.client == mock_client

    # Verify client initialization
    mock_client_class.assert_called_once()
    mock_validate_config.assert_not_called()  # No config validation
    client_args = mock_client_class.call_args[1]
    assert client_args["config"] == config
    assert isinstance(client_args["signer"], MockSigner)


def test_set_region(test_store):
    """Test setting region."""
    old_region = test_store.config["region"]
    test_store.set_region("new-region")

    # Verify region is updated
    assert test_store.config["region"] == "new-region"
    assert test_store.config["region"] != old_region

    # Verify client region is updated
    test_store.client.base_client.set_region.assert_called_once_with("new-region")


def test_get_namespace(test_store):
    """Test getting namespace."""
    # Setup mock response
    mock_response = MagicMock()
    mock_response.data = "test-namespace"
    test_store.client.get_namespace.return_value = mock_response

    # Get namespace
    namespace = test_store.get_namespace()

    # Verify response
    assert namespace == "test-namespace"
    assert isinstance(namespace, str)
    test_store.client.get_namespace.assert_called_once()
    test_store.client.get_namespace.assert_called_with()  # No args


def test_get_object(test_store):
    """Test getting object."""
    # Setup test data
    uri = ObjectURI(namespace="test", bucket_name="bucket", object_name="test.txt")
    mock_response = MagicMock()
    mock_response.status = 200  # Simulate successful response
    test_store.client.get_object.return_value = mock_response

    # Get object
    response = test_store.get_object(uri)

    # Verify response
    assert response == mock_response
    assert response.status == 200
    test_store.client.get_object.assert_called_once_with(
        namespace_name="test",
        bucket_name="bucket",
        object_name="test.txt",
    )


def test_put_object(test_store):
    """Test putting object."""
    # Setup test data
    uri = ObjectURI(namespace="test", bucket_name="bucket", object_name="test.txt")
    data = BytesIO(b"test data")
    test_store.client.put_object.return_value = MagicMock(status=200)

    # Put object
    test_store.put_object(uri, data)

    # Verify put_object call
    test_store.client.put_object.assert_called_once()
    args = test_store.client.put_object.call_args[1]
    assert args["namespace_name"] == "test"
    assert args["bucket_name"] == "bucket"
    assert args["object_name"] == "test.txt"
    assert args["put_object_body"] == data
    assert args["put_object_body"].getvalue() == b"test data"


def test_list_objects(test_store):
    """Test listing objects."""
    # Setup test data
    uri = ObjectURI(namespace="test", bucket_name="bucket", prefix="test/")
    mock_objects = [
        ObjectSummary(name="test/file1.txt"),
        ObjectSummary(name="test/file2.txt"),
    ]
    mock_response = MagicMock(spec=Response)
    mock_response.data = ListObjects(objects=mock_objects, next_start_with=None)
    test_store.client.list_objects.return_value = mock_response

    # List objects
    objects = list(test_store.list_objects(uri))

    # Verify response
    assert len(objects) == 2
    assert objects == ["test/file1.txt", "test/file2.txt"]
    assert all(isinstance(obj, str) for obj in objects)
    test_store.client.list_objects.assert_called_once_with(
        namespace_name="test",
        bucket_name="bucket",
        prefix="test/",
    )


def test_list_objects_auto_namespace(test_store):
    """Test listing objects with auto namespace."""
    # Setup test data
    uri = ObjectURI(namespace="", bucket_name="bucket", prefix="test/")
    namespace_response = MagicMock()
    namespace_response.data = "auto-namespace"
    test_store.client.get_namespace.return_value = namespace_response

    mock_objects = [ObjectSummary(name="test/file1.txt")]
    mock_response = MagicMock(spec=Response)
    mock_response.data = ListObjects(objects=mock_objects, next_start_with=None)
    test_store.client.list_objects.return_value = mock_response

    # List objects
    objects = list(test_store.list_objects(uri))

    # Verify response
    assert len(objects) == 1
    assert objects == ["test/file1.txt"]
    assert all(isinstance(obj, str) for obj in objects)

    # Verify namespace handling
    test_store.client.get_namespace.assert_called_once()
    test_store.client.get_namespace.assert_called_with()
    assert uri.namespace == "auto-namespace"  # URI should be updated

    # Verify list_objects call
    test_store.client.list_objects.assert_called_once_with(
        namespace_name="auto-namespace",
        bucket_name="bucket",
        prefix="test/",
    )


def test_download(test_store, tmp_path):
    """Test downloading object."""
    # Setup test data
    target = tmp_path / "test.txt"
    uri = ObjectURI(namespace="test", bucket_name="bucket", object_name="test.txt")
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.data.raw.stream.return_value = [b"test data"]
    test_store.client.get_object.return_value = mock_response

    # Download object
    test_store.download(uri, str(target))

    # Verify file contents
    assert target.exists()
    assert target.read_bytes() == b"test data"
    assert target.stat().st_size == len(b"test data")

    # Verify get_object call
    test_store.client.get_object.assert_called_once_with(
        namespace_name="test",
        bucket_name="bucket",
        object_name="test.txt",
    )


def test_download_auto_namespace(test_store, tmp_path):
    """Test downloading object with auto namespace."""
    # Setup test data
    target = tmp_path / "test.txt"
    uri = ObjectURI(namespace="", bucket_name="bucket", object_name="test.txt")

    namespace_response = MagicMock()
    namespace_response.data = "auto-namespace"
    test_store.client.get_namespace.return_value = namespace_response

    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.data.raw.stream.return_value = [b"test data"]
    test_store.client.get_object.return_value = mock_response

    # Download object
    test_store.download(uri, str(target))

    # Verify file contents
    assert target.exists()
    assert target.read_bytes() == b"test data"
    assert target.stat().st_size == len(b"test data")

    # Verify namespace handling
    test_store.client.get_namespace.assert_called_once()
    test_store.client.get_namespace.assert_called_with()
    assert uri.namespace == "auto-namespace"  # URI should be updated

    # Verify get_object call
    test_store.client.get_object.assert_called_once_with(
        namespace_name="auto-namespace",
        bucket_name="bucket",
        object_name="test.txt",
    )


def test_upload_small_file(test_store, tmp_path):
    """Test uploading small file (single-part)."""
    # Setup test data
    source = tmp_path / "test.txt"
    test_data = b"test data"
    source.write_bytes(test_data)
    uri = ObjectURI(namespace="test", bucket_name="bucket", object_name="test.txt")

    # Mock successful response
    test_store.client.put_object.return_value = MagicMock(status=200)

    # Upload file
    test_store.upload(str(source), uri)

    # Verify put_object call
    test_store.client.put_object.assert_called_once()
    args = test_store.client.put_object.call_args[1]
    assert args["namespace_name"] == "test"
    assert args["bucket_name"] == "bucket"
    assert args["object_name"] == "test.txt"
    # File is opened in binary mode
    assert hasattr(args["put_object_body"], "read")
    assert hasattr(args["put_object_body"], "seek")


def test_upload_auto_namespace(test_store, tmp_path):
    """Test uploading file with auto namespace."""
    # Setup test data
    source = tmp_path / "test.txt"
    test_data = b"test data"
    source.write_bytes(test_data)
    uri = ObjectURI(namespace="", bucket_name="bucket", object_name="test.txt")

    namespace_response = MagicMock()
    namespace_response.data = "auto-namespace"
    test_store.client.get_namespace.return_value = namespace_response
    test_store.client.put_object.return_value = MagicMock(status=200)

    # Upload file
    test_store.upload(str(source), uri)

    # Verify namespace handling
    test_store.client.get_namespace.assert_called_once()
    test_store.client.get_namespace.assert_called_with()
    assert uri.namespace == "auto-namespace"  # URI should be updated

    # Verify put_object call
    test_store.client.put_object.assert_called_once()
    args = test_store.client.put_object.call_args[1]
    assert args["namespace_name"] == "auto-namespace"
    assert args["bucket_name"] == "bucket"
    assert args["object_name"] == "test.txt"
    # File is opened in binary mode
    assert hasattr(args["put_object_body"], "read")
    assert hasattr(args["put_object_body"], "seek")


@pytest.fixture
def mock_upload_manager():
    """Create a mock UploadManager."""
    mock = MagicMock()
    mock.upload_file = MagicMock()
    mock.upload_file.return_value = MagicMock(status=200)
    return mock


@patch("genai_bench.oci_object_storage.os_datastore.UploadManager")
def test_upload_large_file(
    mock_upload_manager_class, mock_upload_manager, test_store, tmp_path
):
    """Test uploading large file (multipart)."""
    # Setup test data
    source = tmp_path / "test.txt"
    file_size = 128 * 1024 * 1024 + 1  # Just over 128MB
    source.write_bytes(b"x" * file_size)
    uri = ObjectURI(namespace="test", bucket_name="bucket", object_name="test.txt")

    # Set up mock
    mock_upload_manager_class.return_value = mock_upload_manager
    mock_upload_manager.upload_file.return_value = MagicMock(status=200)

    # Upload file
    test_store.upload(str(source), uri)

    # Verify UploadManager setup
    mock_upload_manager_class.assert_called_once_with(
        test_store.client, allow_multipart_uploads=True
    )

    # Verify upload_file call
    mock_upload_manager.upload_file.assert_called_once_with(
        namespace_name="test",
        bucket_name="bucket",
        object_name="test.txt",
        file_path=str(source),
        part_size=128 * 1024 * 1024,  # 128MB parts
        parallel_process_count=3,  # Default workers
    )


@patch("genai_bench.oci_object_storage.os_datastore.UploadManager")
def test_upload_large_file_error(
    mock_upload_manager_class, mock_upload_manager, test_store, tmp_path
):
    """Test uploading large file with error."""
    # Setup test data
    source = tmp_path / "test.txt"
    file_size = 128 * 1024 * 1024 + 1  # Just over 128MB
    source.write_bytes(b"x" * file_size)
    uri = ObjectURI(namespace="test", bucket_name="bucket", object_name="test.txt")

    # Set up mock with error response
    mock_upload_manager_class.return_value = mock_upload_manager
    error_response = MagicMock()
    error_response.status = 500
    error_response.data.error_message = "Upload failed"
    mock_upload_manager.upload_file.return_value = error_response

    # Verify error handling
    with pytest.raises(Exception) as exc:
        test_store.upload(str(source), uri)
    assert str(exc.value) == "Multipart upload failed: Upload failed"

    # Verify UploadManager setup
    mock_upload_manager_class.assert_called_once_with(
        test_store.client, allow_multipart_uploads=True
    )

    # Verify upload_file call
    mock_upload_manager.upload_file.assert_called_once_with(
        namespace_name="test",
        bucket_name="bucket",
        object_name="test.txt",
        file_path=str(source),
        part_size=128 * 1024 * 1024,  # 128MB parts
        parallel_process_count=3,  # Default workers
    )


def test_upload_folder(test_store, tmp_path):
    """Test uploading a folder."""
    # Create test directory structure
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()

    # Create some test files
    (test_dir / "file1.txt").write_text("content1")
    (test_dir / "file2.txt").write_text("content2")

    # Create nested directory with files
    nested_dir = test_dir / "nested"
    nested_dir.mkdir()
    (nested_dir / "file3.txt").write_text("content3")

    # Mock the upload method
    test_store.upload = MagicMock()

    # Test uploading folder without prefix
    test_store.upload_folder(
        folder_path=test_dir, bucket="test-bucket", namespace="test-namespace"
    )

    # Verify upload calls
    assert test_store.upload.call_count == 3

    # Convert calls to a set for order-independent comparison
    actual_calls = {
        (
            call_args[0][0],
            str(call_args[0][1]),
        )  # Convert ObjectURI to string for comparison
        for call_args in test_store.upload.call_args_list
    }

    expected_calls = {
        (
            str(test_dir / "file1.txt"),
            str(
                ObjectURI(
                    namespace="test-namespace",
                    bucket_name="test-bucket",
                    object_name="file1.txt",
                    prefix="",
                    region=test_store.config["region"],
                )
            ),
        ),
        (
            str(test_dir / "file2.txt"),
            str(
                ObjectURI(
                    namespace="test-namespace",
                    bucket_name="test-bucket",
                    object_name="file2.txt",
                    prefix="",
                    region=test_store.config["region"],
                )
            ),
        ),
        (
            str(test_dir / "nested/file3.txt"),
            str(
                ObjectURI(
                    namespace="test-namespace",
                    bucket_name="test-bucket",
                    object_name="nested/file3.txt",
                    prefix="",
                    region=test_store.config["region"],
                )
            ),
        ),
    }

    assert actual_calls == expected_calls

    # Reset mock
    test_store.upload.reset_mock()

    # Test uploading folder with prefix
    test_store.upload_folder(
        folder_path=test_dir,
        bucket="test-bucket",
        namespace="test-namespace",
        prefix="prefix",
    )

    # Verify upload calls with prefix
    assert test_store.upload.call_count == 3

    # Convert calls to a set for order-independent comparison
    actual_calls_with_prefix = {
        (
            call_args[0][0],
            str(call_args[0][1]),
        )  # Convert ObjectURI to string for comparison
        for call_args in test_store.upload.call_args_list
    }

    expected_calls_with_prefix = {
        (
            str(test_dir / "file1.txt"),
            str(
                ObjectURI(
                    namespace="test-namespace",
                    bucket_name="test-bucket",
                    object_name="prefix/file1.txt",
                    prefix="prefix",
                    region=test_store.config["region"],
                )
            ),
        ),
        (
            str(test_dir / "file2.txt"),
            str(
                ObjectURI(
                    namespace="test-namespace",
                    bucket_name="test-bucket",
                    object_name="prefix/file2.txt",
                    prefix="prefix",
                    region=test_store.config["region"],
                )
            ),
        ),
        (
            str(test_dir / "nested/file3.txt"),
            str(
                ObjectURI(
                    namespace="test-namespace",
                    bucket_name="test-bucket",
                    object_name="prefix/nested/file3.txt",
                    prefix="prefix",
                    region=test_store.config["region"],
                )
            ),
        ),
    }

    assert actual_calls_with_prefix == expected_calls_with_prefix


def test_upload_folder_not_exists(test_store, tmp_path):
    """Test uploading a non-existent folder."""
    non_existent_dir = tmp_path / "non_existent"

    with pytest.raises(ValueError) as exc_info:
        test_store.upload_folder(
            folder_path=non_existent_dir,
            bucket="test-bucket",
            namespace="test-namespace",
        )

    assert str(exc_info.value) == f"Path {non_existent_dir} is not a directory"
