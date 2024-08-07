from typing import Any, Dict

import oci

from genai_bench.auth.auth_provider import AuthProvider


class OCIOBOTokenAuth(AuthProvider):
    """OCI Authentication Provider using OBO (On-Behalf-Of) token."""

    def __init__(self, token: str, region: str):
        """Initialize OCI OBO Token Auth Provider.

        Args:
            token (str): The security token for OBO authentication
            region (str): The OCI region (e.g., 'us-ashburn-1')
        """
        self.token = token
        self.region = region
        self._signer = None

    def get_config(self) -> Dict[str, Any]:
        """Get OCI configuration.

        Returns:
            Dict[str, Any]: OCI configuration dictionary with region
        """
        return {
            "region": self.region,
        }

    def get_auth_credentials(self) -> oci.signer.AbstractBaseSigner:
        """Get OCI security token signer.

        Returns:
            oci.signer.AbstractBaseSigner: OCI security token signer

        Raises:
            Exception: If token is invalid
        """
        if self._signer is None:
            self._signer = oci.auth.signers.SecurityTokenSigner(
                self.token, region=self.region
            )
        return self._signer
