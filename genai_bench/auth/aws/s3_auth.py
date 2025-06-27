"""AWS S3 storage authentication provider."""

import os
from typing import Any, Dict, Optional

from genai_bench.auth.storage_auth_provider import StorageAuthProvider


class AWSS3Auth(StorageAuthProvider):
    """AWS S3 authentication provider."""

    def __init__(
        self,
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
        session_token: Optional[str] = None,
        region: Optional[str] = None,
        profile: Optional[str] = None,
    ):
        """Initialize AWS S3 authentication.

        Args:
            access_key_id: AWS access key ID
            secret_access_key: AWS secret access key
            session_token: Optional AWS session token for temporary credentials
            region: AWS region
            profile: AWS profile name to use from credentials file
        """
        self.access_key_id = access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
        self.secret_access_key = secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY")
        self.session_token = session_token or os.getenv("AWS_SESSION_TOKEN")
        # Store original region to track if it was explicitly set
        self._explicit_region = region
        self.region = region or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
        self.profile = profile or os.getenv("AWS_PROFILE")

    def get_client_config(self) -> Dict[str, Any]:
        """Get S3 client configuration.

        Returns:
            Dict[str, Any]: Configuration for S3 client
        """
        config = {}

        # Only include region if it was explicitly set or from environment
        if self._explicit_region or os.getenv("AWS_DEFAULT_REGION"):
            config["region_name"] = self.region

        if self.profile:
            config["profile_name"] = self.profile
        else:
            if self.access_key_id:
                config["aws_access_key_id"] = self.access_key_id
            if self.secret_access_key:
                config["aws_secret_access_key"] = self.secret_access_key
            if self.session_token:
                config["aws_session_token"] = self.session_token

        return config

    def get_credentials(self) -> Dict[str, Optional[str]]:
        """Get AWS credentials for storage operations.

        Returns:
            Dict with AWS credentials
        """
        return {
            "aws_access_key_id": self.access_key_id,
            "aws_secret_access_key": self.secret_access_key,
            "aws_session_token": self.session_token,
            "region_name": self.region,
            "profile_name": self.profile,
        }

    def get_storage_type(self) -> str:
        """Get the storage provider type.

        Returns:
            'aws'
        """
        return "aws"

    def get_region(self) -> Optional[str]:
        """Get the AWS region.

        Returns:
            AWS region or None
        """
        return self.region
