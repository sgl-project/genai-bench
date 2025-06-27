"""OCI storage authentication adapter for backward compatibility."""

from typing import Any, Dict, Optional

from genai_bench.auth.auth_provider import AuthProvider
from genai_bench.auth.storage_auth_provider import StorageAuthProvider


class OCIStorageAuthAdapter(StorageAuthProvider):
    """Adapter to use existing OCI auth providers as storage auth providers."""

    def __init__(self, oci_auth: AuthProvider):
        """Initialize OCI storage auth adapter.

        Args:
            oci_auth: Existing OCI auth provider instance
        """
        self.oci_auth = oci_auth

    def get_client_config(self) -> Dict[str, Any]:
        """Get OCI storage client configuration.

        Returns:
            Dict[str, Any]: Configuration for storage client
        """
        return {
            "auth_provider": self.oci_auth,
        }

    def get_credentials(self) -> Any:
        """Get OCI authentication credentials.

        Returns:
            The OCI auth provider instance
        """
        return self.oci_auth

    def get_storage_type(self) -> str:
        """Get the storage provider type.

        Returns:
            'oci'
        """
        return "oci"

    def get_region(self) -> Optional[str]:
        """Get the OCI region.

        Returns:
            OCI region from the auth provider
        """
        # Try to get region from the auth provider if available
        if self.oci_auth and hasattr(self.oci_auth, "region"):
            return self.oci_auth.region
        return None
