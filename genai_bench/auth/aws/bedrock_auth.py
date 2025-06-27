"""AWS Bedrock authentication provider."""

import os
from typing import Any, Dict, Optional

from genai_bench.auth.model_auth_provider import ModelAuthProvider


class AWSBedrockAuth(ModelAuthProvider):
    """AWS Bedrock authentication provider using AWS credentials."""

    def __init__(
        self,
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
        session_token: Optional[str] = None,
        region: Optional[str] = None,
        profile: Optional[str] = None,
    ):
        """Initialize AWS Bedrock authentication.

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
        self.region = region or os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        self.profile = profile or os.getenv("AWS_PROFILE")

    def get_headers(self) -> Dict[str, str]:
        """Get authentication headers for Bedrock API requests.

        Note: AWS Bedrock uses SigV4 signing which is handled by the SDK,
        not through headers.

        Returns:
            Empty dict as auth is handled by AWS SDK
        """
        return {}

    def get_config(self) -> Dict[str, Any]:
        """Get AWS Bedrock configuration.

        Returns:
            Dict containing AWS configuration
        """
        config = {
            "region": self.region,
            "auth_type": self.get_auth_type(),
        }

        if self.profile:
            config["profile"] = self.profile
        else:
            config["access_key_id"] = self.access_key_id
            if self.session_token:
                config["session_token"] = self.session_token

        return config

    def get_auth_type(self) -> str:
        """Get the authentication type identifier.

        Returns:
            'aws_bedrock'
        """
        return "aws_bedrock"

    def get_credentials(self) -> Dict[str, Optional[str]]:
        """Get AWS credentials for SDK initialization.

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
