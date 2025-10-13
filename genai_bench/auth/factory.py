from typing import Optional

import oci.config

from genai_bench.auth.auth_provider import AuthProvider
from genai_bench.auth.oci.instance_principal import OCIInstancePrincipalAuth
from genai_bench.auth.oci.obo_token import OCIOBOTokenAuth
from genai_bench.auth.oci.session import OCISessionAuth
from genai_bench.auth.oci.user_principal import OCIUserPrincipalAuth
from genai_bench.auth.openai.auth import OpenAIAuth
from genai_bench.auth.together.auth import TogetherAuth


class AuthFactory:
    """Factory for creating authentication providers."""

    @staticmethod
    def create_openai_auth(api_key: str) -> OpenAIAuth:
        """Create OpenAI authentication provider.

        Args:
            api_key (str): OpenAI API key

        Returns:
            OpenAIAuth: OpenAI auth provider
        """
        return OpenAIAuth(api_key=api_key)

    @staticmethod
    def create_together_auth(api_key: str) -> TogetherAuth:
        """Create Together authentication provider.

        Args:
            api_key (str): Together API key

        Returns:
            TogetherAuth: OpenAI auth provider
        """
        return TogetherAuth(api_key=api_key)

    @staticmethod
    def create_oci_auth(
        auth_type: str,
        config_path: Optional[str] = oci.config.DEFAULT_LOCATION,
        profile: Optional[str] = oci.config.DEFAULT_PROFILE,
        token: Optional[str] = None,
        region: Optional[str] = None,
    ) -> AuthProvider:
        """Create OCI authentication provider.

        Args:
            auth_type (str): Type of OCI auth. One of: 'user_principal',
            'instance_principal', 'security_token', or 'instance_obo_user'

            config_path (Optional[str]): Path to OCI config file. Required for
            user_principal and session auth.

            profile (Optional[str]): Profile name in config.
            Only used with user_principal and session auth. Defaults to "DEFAULT".

            token (Optional[str]): token string. Required for instance_obo_user auth.
            region (Optional[str]): OCI region. Required for instance_obo_user auth.

        Returns:
            AuthProvider: OCI auth provider

        Raises:
            ValueError: If auth_type is invalid or required args are missing
        """
        # Validate auth_type first
        valid_types = [
            "user_principal",
            "instance_principal",
            "security_token",
            "instance_obo_user",
        ]
        if auth_type not in valid_types:
            raise ValueError(
                f"Invalid auth_type: {auth_type}. Must be one of: "
                f"{', '.join(repr(t) for t in valid_types)}"
            )

        if auth_type == "instance_principal":
            return OCIInstancePrincipalAuth(security_token=token, region=region)

        if auth_type == "instance_obo_user":
            if token is None or region is None:
                raise ValueError(
                    "token and region are required for obo_token authentication"
                )
            return OCIOBOTokenAuth(token=token, region=region)

        if auth_type == "user_principal":
            return OCIUserPrincipalAuth(config_path=config_path, profile=profile)

        if auth_type == "security_token":
            return OCISessionAuth(config_path=config_path, profile=profile)

        # This code is unreachable since we validate auth_type at the start
        raise ValueError(f"Unsupported auth_type: {auth_type}")  # pragma: no cover
