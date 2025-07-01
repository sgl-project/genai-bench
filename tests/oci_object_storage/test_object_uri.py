"""Test cases for ObjectURI class."""

import pytest
from pydantic import ValidationError

from genai_bench.storage.oci_object_storage.object_uri import ObjectURI


def test_object_uri_creation():
    """Test creating ObjectURI with valid parameters."""
    uri = ObjectURI(
        namespace="test-namespace",
        bucket_name="test-bucket",
        object_name="test/object.txt",
        prefix="test/",
        region="us-ashburn-1",
    )
    assert uri.namespace == "test-namespace"
    assert uri.bucket_name == "test-bucket"
    assert uri.object_name == "test/object.txt"
    assert uri.region == "us-ashburn-1"
    assert uri.prefix == "test/"


def test_object_uri_validation():
    """Test validation of required fields."""
    # Missing required fields
    with pytest.raises(ValidationError):
        ObjectURI()

    with pytest.raises(ValidationError):
        ObjectURI(namespace="test")

    with pytest.raises(ValidationError):
        ObjectURI(bucket_name="test")

    # Valid minimal URI
    uri = ObjectURI(namespace="test", bucket_name="bucket")
    assert uri.namespace == "test"
    assert uri.bucket_name == "bucket"
    assert uri.object_name is None
    assert uri.region is None
    assert uri.prefix is None


def test_from_uri_parsing():
    """Test parsing URIs into ObjectURI objects."""
    # Test bucket-only URI
    uri = ObjectURI.from_uri("oci://n/test-namespace/b/test-bucket/")
    assert uri.namespace == "test-namespace"
    assert uri.bucket_name == "test-bucket"
    assert uri.object_name is None
    assert uri.prefix is None

    # Test URI with object in subdirectory
    uri = ObjectURI.from_uri("oci://n/test-namespace/b/test-bucket/o/test/object.txt")
    assert uri.namespace == "test-namespace"
    assert uri.bucket_name == "test-bucket"
    assert uri.object_name == "test/object.txt"
    assert uri.prefix == "test/"

    # Test URI with object in root
    uri = ObjectURI.from_uri("oci://n/test-namespace/b/test-bucket/o/object.txt")
    assert uri.namespace == "test-namespace"
    assert uri.bucket_name == "test-bucket"
    assert uri.object_name == "object.txt"
    assert uri.prefix == ""


def test_from_uri_validation():
    """Test validation of URI format."""
    invalid_uris = [
        # Invalid scheme
        "s3://n/test/b/bucket",
        # Missing namespace prefix
        "oci://test/b/bucket",
        # Missing bucket prefix
        "oci://n/test/bucket",
        # Incomplete URI
        "oci://n/test/b",
        # Empty URI
        "",
    ]

    for uri in invalid_uris:
        with pytest.raises(ValueError, match=f"Invalid OCI storage URI format: {uri}"):
            ObjectURI.from_uri(uri)


def test_string_representation():
    """Test string representation of ObjectURI."""
    # Bucket only
    uri = ObjectURI(namespace="test", bucket_name="bucket")
    assert str(uri) == "oci://n/test/b/bucket"

    # With object
    uri = ObjectURI(
        namespace="test-namespace",
        bucket_name="test-bucket",
        object_name="test/object.txt",
    )
    assert str(uri) == "oci://n/test-namespace/b/test-bucket/o/test/object.txt"

    # With region (not included in string representation)
    uri = ObjectURI(
        namespace="test",
        bucket_name="bucket",
        region="us-ashburn-1",
    )
    assert str(uri) == "oci://n/test/b/bucket"
