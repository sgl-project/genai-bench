"""OCI Object URI data class."""

import os
from typing import Optional

from pydantic import BaseModel, Field


class ObjectURI(BaseModel):
    """Represents an OCI Storage URI.

    Format: oci://n/{namespace}/b/{bucket}/o/{object_path}
    """

    namespace: str = Field(description="OCI namespace")
    bucket_name: str = Field(description="Bucket name")
    object_name: Optional[str] = Field(None, description="Object name/path")
    region: Optional[str] = Field(None, description="OCI region")
    prefix: Optional[str] = Field(None, description="Object prefix for listing")

    @classmethod
    def from_uri(cls, uri: str) -> "ObjectURI":
        """Create ObjectURI from string URI.

        Args:
            uri: OCI storage URI string

        Returns:
            ObjectURI object

        Raises:
            ValueError: If URI format is invalid
        """
        if not uri.startswith("oci://"):
            raise ValueError(f"Invalid OCI storage URI format: {uri}")

        # Remove oci:// prefix
        path = uri[6:]

        try:
            # Parse namespace
            if not path.startswith("n/"):
                raise ValueError("Missing namespace (n/) in URI")
            path = path[2:]
            namespace, path = path.split("/", 1)

            # Parse bucket
            if not path.startswith("b/"):
                raise ValueError("Missing bucket (b/) in URI")
            path = path[2:]
            bucket, path = path.split("/", 1)

            # Parse object path (optional)
            object_name = None
            prefix = None
            if path.startswith("o/"):
                path = path[2:]
                if path:
                    object_name = path
                    prefix = os.path.dirname(object_name)
                    if prefix:
                        prefix = prefix + "/"

            return cls(
                namespace=namespace,
                bucket_name=bucket,
                object_name=object_name,
                region=None,  # Region is not part of the URI format
                prefix=prefix,
            )

        except ValueError as e:
            raise ValueError(f"Invalid OCI storage URI format: {uri}") from e

    def __str__(self) -> str:
        """Get string representation of URI.

        Returns:
            URI string in format oci://n/{namespace}/b/{bucket}/o/{object_name}
        """
        base = f"oci://n/{self.namespace}/b/{self.bucket_name}"
        if self.object_name:
            return f"{base}/o/{self.object_name}"
        return base
