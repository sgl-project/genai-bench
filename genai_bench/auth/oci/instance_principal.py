from typing import Any, Dict, Optional

import oci

from genai_bench.auth.auth_provider import AuthProvider


class OCIInstancePrincipalAuth(AuthProvider):
    """OCI Authentication Provider using instance principal."""

    def __init__(
        self, security_token: Optional[str] = None, region: Optional[str] = None
    ):
        """Initialize OCI Instance Principal Auth Provider.

        Args:
            security_token (Optional[str]): Security token for authentication
            region (Optional[str]): OCI region
        """
        self._signer = None
        self.region = region

    def get_config(self) -> Dict[str, Any]:
        """Get OCI configuration.

        Returns:
            Dict[str, Any]: OCI configuration dictionary with region and tenancy

        Raises:
            Exception: If instance principal auth is not configured
        """
        signer = self.get_auth_credentials()
        config = {"region": signer.region, "tenancy": signer.tenancy_id}
        if self.region:
            config["region"] = self.region
        return config

    def get_auth_credentials(self) -> oci.signer.AbstractBaseSigner:
        """Get OCI instance principal signer.

        Returns:
            oci.signer.AbstractBaseSigner: OCI instance principal signer

        Raises:
            Exception: If instance principal auth is not configured
        """
        if self._signer is None:
            self._signer = oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
        return self._signer
