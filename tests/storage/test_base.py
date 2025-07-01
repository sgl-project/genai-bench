"""Tests for base storage interface."""

import pytest

from genai_bench.storage.base import BaseStorage


class TestBaseStorage:
    """Test base storage interface."""

    def test_abstract_interface(self):
        """Test that BaseStorage cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseStorage()

    def test_abstract_methods(self):
        """Test that abstract methods are defined."""

        # Create a concrete implementation for testing
        class ConcreteStorage(BaseStorage):
            def upload_file(self, local_path, remote_path, bucket, **kwargs):
                return f"uploaded {local_path} to {bucket}/{remote_path}"

            def upload_folder(self, local_folder, bucket, prefix="", **kwargs):
                return f"uploaded {local_folder} to {bucket}/{prefix}"

            def download_file(self, remote_path, local_path, bucket, **kwargs):
                return f"downloaded {bucket}/{remote_path} to {local_path}"

            def list_objects(self, bucket, prefix=None, **kwargs):
                yield "file1.txt"
                yield "file2.txt"

            def delete_object(self, remote_path, bucket, **kwargs):
                return f"deleted {bucket}/{remote_path}"

            def get_storage_type(self):
                return "test"

        # Test the concrete implementation
        storage = ConcreteStorage()

        # Test all methods
        assert (
            storage.upload_file("local.txt", "remote.txt", "bucket")
            == "uploaded local.txt to bucket/remote.txt"
        )
        assert storage.upload_folder("local", "bucket") == "uploaded local to bucket/"
        assert (
            storage.download_file("remote.txt", "local.txt", "bucket")
            == "downloaded bucket/remote.txt to local.txt"
        )
        assert list(storage.list_objects("bucket")) == ["file1.txt", "file2.txt"]
        assert (
            storage.delete_object("remote.txt", "bucket") == "deleted bucket/remote.txt"
        )
        assert storage.get_storage_type() == "test"

    def test_missing_method_implementation(self):
        """Test that missing method implementation raises error."""

        # Create incomplete implementation
        class IncompleteStorage(BaseStorage):
            def upload_file(self, local_path, remote_path, bucket, **kwargs):
                return ""

            def upload_folder(self, local_folder, bucket, prefix="", **kwargs):
                return ""

            def download_file(self, remote_path, local_path, bucket, **kwargs):
                return ""

            def list_objects(self, bucket, prefix=None, **kwargs):
                return []

            def delete_object(self, remote_path, bucket, **kwargs):
                return ""

            # Missing get_storage_type

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteStorage()
