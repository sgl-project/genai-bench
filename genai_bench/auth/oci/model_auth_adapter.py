"""OCI model authentication adapter for backward compatibility."""

from typing import Any, Dict

from genai_bench.auth.auth_provider import AuthProvider
from genai_bench.auth.model_auth_provider import ModelAuthProvider


class OCIModelAuthAdapter(ModelAuthProvider):
    """Adapter to use existing OCI auth providers as model auth providers."""

    def __init__(self, oci_auth: AuthProvider):
        """Initialize OCI model auth adapter.

        Args:
            oci_auth: Existing OCI auth provider instance
        """
        self.oci_auth = oci_auth

    def get_headers(self) -> Dict[str, str]:
        """Get authentication headers for OCI model API requests.

        Returns:
            Empty dict as OCI uses signers, not headers
        """
        return {}

    def get_config(self) -> Dict[str, Any]:
        """Get OCI model service configuration.

        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        # Return a serializable configuration
        config = {
            "auth_type": self.get_auth_type(),
        }

        # Add any serializable config from the underlying auth if available
        if hasattr(self.oci_auth, "get_config"):
            config.update(self.oci_auth.get_config())

        return config

    def get_auth_type(self) -> str:
        """Get the authentication type identifier.

        Returns:
            str: 'oci_' + the underlying auth type
        """
        # Get the class name and convert to auth type
        class_name = self.oci_auth.__class__.__name__
        if "InstancePrincipal" in class_name:
            return "oci_instance_principal"
        elif "UserPrincipal" in class_name:
            return "oci_user_principal"
        elif "OBOToken" in class_name:
            return "oci_obo_token"
        elif "Session" in class_name:
            return "oci_security_token"
        else:
            return "oci_unknown"

    def get_credentials(self) -> Any:
        """Get the credentials of the underlying OCI auth provider.

        This is used to authenticate with the OCI model service.

        Returns:
            The credentials of the underlying OCI auth provider
        """
        return self.oci_auth.get_credentials()
