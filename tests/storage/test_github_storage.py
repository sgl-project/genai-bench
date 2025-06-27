"""Unit tests for GitHub storage implementation."""

from unittest.mock import MagicMock, mock_open, patch

import pytest

from genai_bench.auth.storage_auth_provider import StorageAuthProvider


class TestGitHubStorage:
    """Test GitHub storage implementation."""

    @pytest.fixture
    def mock_auth(self):
        """Create mock auth provider."""
        auth = MagicMock(spec=StorageAuthProvider)
        auth.get_storage_type.return_value = "github"
        auth.get_client_config.return_value = {
            "token": "test_token",
            "owner": "test_owner",
            "repo": "test_repo",
        }
        return auth

    def test_init_with_valid_auth(self, mock_auth):
        """Test initialization with valid auth provider."""
        from genai_bench.storage.github_storage import GitHubStorage

        storage = GitHubStorage(mock_auth)

        assert storage.auth == mock_auth
        assert storage.config == mock_auth.get_client_config()
        assert storage.token == "test_token"
        assert storage.owner == "test_owner"
        assert storage.repo == "test_repo"
        assert storage.api_base == "https://api.github.com"
        assert storage.upload_base == "https://uploads.github.com"
        assert storage.headers["Authorization"] == "token test_token"

    def test_init_with_invalid_auth_type(self):
        """Test initialization with wrong auth provider type."""
        from genai_bench.storage.github_storage import GitHubStorage

        auth = MagicMock(spec=StorageAuthProvider)
        auth.get_storage_type.return_value = "aws"

        with pytest.raises(ValueError, match="Auth provider must be for GitHub"):
            GitHubStorage(auth)

    def test_init_with_missing_config(self, mock_auth):
        """Test initialization with missing configuration."""
        from genai_bench.storage.github_storage import GitHubStorage

        mock_auth.get_client_config.return_value = {"token": "test_token"}

        with pytest.raises(
            ValueError, match="GitHub token, owner, and repo are required"
        ):
            GitHubStorage(mock_auth)

    def test_init_with_owner_repo_combined(self, mock_auth):
        """Test initialization with owner/repo as single string."""
        from genai_bench.storage.github_storage import GitHubStorage

        mock_auth.get_client_config.return_value = {
            "token": "test_token",
            "owner": "test_owner/test_repo",
            "repo": "",  # Empty string instead of None
        }

        storage = GitHubStorage(mock_auth)

        assert storage.owner == "test_owner"
        assert storage.repo == "test_repo"

    def test_make_request_success(self, mock_auth):
        """Test successful API request."""
        from genai_bench.storage.github_storage import GitHubStorage

        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None

        with patch("requests.request", return_value=mock_response) as mock_request:
            storage = GitHubStorage(mock_auth)
            response = storage._make_request("GET", "https://api.github.com/test")

            mock_request.assert_called_once_with(
                "GET", "https://api.github.com/test", headers=storage.headers
            )
            response.raise_for_status.assert_called_once()

    def test_get_or_create_release_existing(self, mock_auth):
        """Test getting existing release."""
        from genai_bench.storage.github_storage import GitHubStorage

        release_data = {"id": 123, "tag_name": "v1.0.0"}
        mock_response = MagicMock()
        mock_response.json.return_value = release_data
        mock_response.raise_for_status.return_value = None

        with patch("requests.request", return_value=mock_response) as mock_request:
            storage = GitHubStorage(mock_auth)
            result = storage._get_or_create_release("v1.0.0")

            assert result == release_data
            mock_request.assert_called_once_with(
                "GET",
                "https://api.github.com/repos/test_owner/test_repo/releases/tags/v1.0.0",
                headers=storage.headers,
            )

    def test_get_or_create_release_new(self, mock_auth):
        """Test creating new release when not found."""
        from genai_bench.storage.github_storage import GitHubStorage

        # First request fails (release not found)
        mock_get_response = MagicMock()
        mock_get_response.raise_for_status.side_effect = Exception("Not found")

        # Second request succeeds (create release)
        mock_create_response = MagicMock()
        mock_create_response.json.return_value = {"id": 456, "tag_name": "v2.0.0"}
        mock_create_response.raise_for_status.return_value = None

        with patch(
            "requests.request", side_effect=[mock_get_response, mock_create_response]
        ) as mock_request:
            storage = GitHubStorage(mock_auth)
            result = storage._get_or_create_release("v2.0.0")

            assert result["id"] == 456
            assert mock_request.call_count == 2

    def test_upload_file_success(self, mock_auth):
        """Test successful file upload."""
        from genai_bench.storage.github_storage import GitHubStorage

        # Mock release data
        release_data = {
            "id": 123,
            "upload_url": "https://uploads.github.com/repos/test/releases/123/assets{?name,label}",
            "assets": [],
        }

        mock_get_response = MagicMock()
        mock_get_response.json.return_value = release_data
        mock_get_response.raise_for_status.return_value = None

        mock_upload_response = MagicMock()
        mock_upload_response.raise_for_status.return_value = None

        with patch(
            "requests.request", side_effect=[mock_get_response, mock_upload_response]
        ) as mock_request:
            storage = GitHubStorage(mock_auth)

            # Mock file
            with patch("pathlib.Path.exists", return_value=True):
                with patch("builtins.open", mock_open(read_data=b"file content")):
                    storage.upload_file("/local/file.txt", "remote_file.txt", "v1.0.0")

                    # Should get/create release and upload
                    assert mock_request.call_count == 2

    def test_upload_file_not_found(self, mock_auth):
        """Test upload with non-existent file."""
        from genai_bench.storage.github_storage import GitHubStorage

        storage = GitHubStorage(mock_auth)

        with patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(FileNotFoundError, match="Local file not found"):
                storage.upload_file("/local/missing.txt", "remote_file.txt", "v1.0.0")

    def test_upload_file_replace_existing(self, mock_auth):
        """Test upload replacing existing asset."""
        from genai_bench.storage.github_storage import GitHubStorage

        # Mock release with existing asset
        release_data = {
            "id": 123,
            "upload_url": "https://uploads.github.com/repos/test/releases/123/assets{?name,label}",
            "assets": [{"id": 456, "name": "remote_file.txt"}],
        }

        mock_responses = [
            MagicMock(
                json=MagicMock(return_value=release_data), raise_for_status=MagicMock()
            ),
            MagicMock(raise_for_status=MagicMock()),  # Delete response
            MagicMock(raise_for_status=MagicMock()),  # Upload response
        ]

        with patch("requests.request", side_effect=mock_responses) as mock_request:
            storage = GitHubStorage(mock_auth)

            with patch("pathlib.Path.exists", return_value=True):
                with patch("builtins.open", mock_open(read_data=b"new content")):
                    storage.upload_file("/local/file.txt", "remote_file.txt", "v1.0.0")

                    # Should get release, delete existing, and upload new
                    assert mock_request.call_count == 3

    def test_download_file_success(self, mock_auth):
        """Test successful file download."""
        from genai_bench.storage.github_storage import GitHubStorage

        # Mock release data
        release_data = {
            "assets": [
                {
                    "name": "remote_file.txt",
                    "browser_download_url": "https://github.com/test/releases/download/v1.0.0/remote_file.txt",
                }
            ]
        }

        mock_get_release = MagicMock()
        mock_get_release.json.return_value = release_data
        mock_get_release.raise_for_status.return_value = None

        mock_download = MagicMock()
        mock_download.iter_content.return_value = [b"chunk1", b"chunk2"]
        mock_download.raise_for_status.return_value = None

        with patch("requests.request", side_effect=[mock_get_release, mock_download]):
            storage = GitHubStorage(mock_auth)

            with patch("pathlib.Path.mkdir"):
                with patch("builtins.open", mock_open()) as mock_file:
                    storage.download_file(
                        "remote_file.txt", "/local/file.txt", "v1.0.0"
                    )

                    # Verify file write
                    mock_file().write.assert_any_call(b"chunk1")
                    mock_file().write.assert_any_call(b"chunk2")

    def test_download_file_not_found(self, mock_auth):
        """Test download with non-existent asset."""
        from genai_bench.storage.github_storage import GitHubStorage

        release_data = {"assets": []}
        mock_response = MagicMock()
        mock_response.json.return_value = release_data
        mock_response.raise_for_status.return_value = None

        with patch("requests.request", return_value=mock_response):
            storage = GitHubStorage(mock_auth)

            with patch("pathlib.Path.mkdir"):
                with pytest.raises(FileNotFoundError, match="Asset not found"):
                    storage.download_file("missing.txt", "/local/file.txt", "v1.0.0")

    def test_list_objects_success(self, mock_auth):
        """Test successful object listing."""
        from genai_bench.storage.github_storage import GitHubStorage

        release_data = {
            "assets": [
                {"name": "file1.txt"},
                {"name": "data-file2.txt"},
                {"name": "other-file3.txt"},
            ]
        }
        mock_response = MagicMock()
        mock_response.json.return_value = release_data
        mock_response.raise_for_status.return_value = None

        with patch("requests.request", return_value=mock_response):
            storage = GitHubStorage(mock_auth)

            # List all objects
            objects = list(storage.list_objects("v1.0.0"))
            assert objects == ["file1.txt", "data-file2.txt", "other-file3.txt"]

            # List with prefix
            objects = list(storage.list_objects("v1.0.0", prefix="data"))
            assert objects == ["data-file2.txt"]

    def test_delete_object_success(self, mock_auth):
        """Test successful object deletion."""
        from genai_bench.storage.github_storage import GitHubStorage

        release_data = {"assets": [{"id": 456, "name": "remote_file.txt"}]}

        mock_get_response = MagicMock()
        mock_get_response.json.return_value = release_data
        mock_get_response.raise_for_status.return_value = None

        mock_delete_response = MagicMock()
        mock_delete_response.raise_for_status.return_value = None

        with patch(
            "requests.request", side_effect=[mock_get_response, mock_delete_response]
        ) as mock_request:
            storage = GitHubStorage(mock_auth)
            storage.delete_object("remote_file.txt", "v1.0.0")

            # Should get release and delete asset
            assert mock_request.call_count == 2

    def test_delete_object_not_found(self, mock_auth):
        """Test deletion of non-existent asset."""
        from genai_bench.storage.github_storage import GitHubStorage

        release_data = {"assets": []}
        mock_response = MagicMock()
        mock_response.json.return_value = release_data
        mock_response.raise_for_status.return_value = None

        with patch("requests.request", return_value=mock_response) as mock_request:
            storage = GitHubStorage(mock_auth)

            # Should not raise, just log warning
            storage.delete_object("missing.txt", "v1.0.0")

            # Should only make one call (get release)
            assert mock_request.call_count == 1

    def test_get_storage_type(self, mock_auth):
        """Test storage type getter."""
        from genai_bench.storage.github_storage import GitHubStorage

        storage = GitHubStorage(mock_auth)
        assert storage.get_storage_type() == "github"

    def test_upload_file_error(self, mock_auth):
        """Test upload file with error."""
        from genai_bench.storage.github_storage import GitHubStorage

        release_data = {
            "id": 123,
            "upload_url": "https://uploads.github.com/repos/test/releases/123/assets{?name,label}",
            "assets": [],
        }
        mock_get_response = MagicMock()
        mock_get_response.json.return_value = release_data
        mock_get_response.raise_for_status.return_value = None

        mock_upload_response = MagicMock()
        mock_upload_response.raise_for_status.side_effect = Exception("Upload failed")

        with patch(
            "requests.request", side_effect=[mock_get_response, mock_upload_response]
        ):
            storage = GitHubStorage(mock_auth)

            with patch("pathlib.Path.exists", return_value=True):
                with patch("builtins.open", mock_open(read_data=b"file content")):
                    with pytest.raises(Exception, match="Upload failed"):
                        storage.upload_file(
                            "/local/file.txt", "remote_file.txt", "v1.0.0"
                        )

    def test_upload_folder_success(self, mock_auth):
        """Test successful folder upload."""
        from genai_bench.storage.github_storage import GitHubStorage

        release_data = {
            "id": 123,
            "upload_url": "https://uploads.github.com/repos/test/releases/123/assets{?name,label}",
            "assets": [],
        }

        # Mock responses for get release and multiple uploads
        mock_responses = []
        # First call for get_or_create_release for each file
        for _ in range(2):  # Two files
            mock_get = MagicMock()
            mock_get.json.return_value = release_data
            mock_get.raise_for_status.return_value = None
            mock_responses.append(mock_get)

            mock_upload = MagicMock()
            mock_upload.raise_for_status.return_value = None
            mock_responses.append(mock_upload)

        with patch("requests.request", side_effect=mock_responses):
            storage = GitHubStorage(mock_auth)

            with patch("pathlib.Path.exists", return_value=True):
                with patch("pathlib.Path.is_dir", return_value=True):
                    # Create mock files
                    from pathlib import Path

                    mock_file1 = MagicMock()
                    mock_file1.is_file.return_value = True
                    mock_file1.relative_to.return_value = Path("subdir/file1.txt")

                    mock_file2 = MagicMock()
                    mock_file2.is_file.return_value = True
                    mock_file2.relative_to.return_value = Path("file2.txt")

                    mock_dir = MagicMock()
                    mock_dir.is_file.return_value = False

                    mock_files = [mock_file1, mock_file2, mock_dir]

                    with patch("pathlib.Path.rglob", return_value=mock_files):
                        with patch("builtins.open", mock_open(read_data=b"content")):
                            storage.upload_folder(
                                "/local/folder", "v1.0.0", prefix="test"
                            )

                            # Should upload 2 files (not the directory)
                            # 2 files * 2 requests each = 4 total requests
                            assert len(mock_responses) == 4

    def test_upload_folder_not_found(self, mock_auth):
        """Test upload folder that doesn't exist."""
        from genai_bench.storage.github_storage import GitHubStorage

        storage = GitHubStorage(mock_auth)

        with patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(ValueError, match="Local folder not found"):
                storage.upload_folder("/missing/folder", "v1.0.0")

    def test_upload_folder_not_directory(self, mock_auth):
        """Test upload folder with file path."""
        from genai_bench.storage.github_storage import GitHubStorage

        storage = GitHubStorage(mock_auth)

        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.is_dir", return_value=False):
                with pytest.raises(ValueError, match="not a directory"):
                    storage.upload_folder("/local/file.txt", "v1.0.0")

    def test_upload_folder_no_prefix(self, mock_auth):
        """Test folder upload without prefix."""
        from genai_bench.storage.github_storage import GitHubStorage

        release_data = {
            "id": 123,
            "upload_url": "https://uploads.github.com/repos/test/releases/123/assets{?name,label}",
            "assets": [],
        }

        mock_get = MagicMock()
        mock_get.json.return_value = release_data
        mock_get.raise_for_status.return_value = None

        mock_upload = MagicMock()
        mock_upload.raise_for_status.return_value = None

        with patch("requests.request", side_effect=[mock_get, mock_upload]):
            storage = GitHubStorage(mock_auth)

            with patch("pathlib.Path.exists", return_value=True):
                with patch("pathlib.Path.is_dir", return_value=True):
                    from pathlib import Path

                    mock_file = MagicMock()
                    mock_file.is_file.return_value = True
                    mock_file.relative_to.return_value = Path("file.txt")

                    with patch("pathlib.Path.rglob", return_value=[mock_file]):
                        with patch("builtins.open", mock_open(read_data=b"content")):
                            storage.upload_folder("/local/folder", "v1.0.0")

    def test_list_objects_error(self, mock_auth):
        """Test list objects with error."""
        from genai_bench.storage.github_storage import GitHubStorage

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("List failed")

        with patch("requests.request", return_value=mock_response):
            storage = GitHubStorage(mock_auth)

            with pytest.raises(Exception, match="List failed"):
                list(storage.list_objects("v1.0.0"))

    def test_delete_object_error(self, mock_auth):
        """Test delete object with error."""
        from genai_bench.storage.github_storage import GitHubStorage

        release_data = {"assets": [{"id": 456, "name": "remote_file.txt"}]}

        mock_get_response = MagicMock()
        mock_get_response.json.return_value = release_data
        mock_get_response.raise_for_status.return_value = None

        mock_delete_response = MagicMock()
        mock_delete_response.raise_for_status.side_effect = Exception("Delete failed")

        with patch(
            "requests.request", side_effect=[mock_get_response, mock_delete_response]
        ):
            storage = GitHubStorage(mock_auth)

            with pytest.raises(Exception, match="Delete failed"):
                storage.delete_object("remote_file.txt", "v1.0.0")
