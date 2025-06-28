"""Factory for creating storage providers."""

from genai_bench.auth.storage_auth_provider import StorageAuthProvider
from genai_bench.storage.base import BaseStorage


class StorageFactory:
    """Factory for creating storage provider instances."""

    @staticmethod
    def create_storage(
        provider: str, auth: StorageAuthProvider, **kwargs
    ) -> BaseStorage:
        """Create a storage provider instance.

        Args:
            provider: Storage provider type ('oci', 'aws', 'azure', 'gcp', 'github')
            auth: Storage authentication provider
            **kwargs: Additional provider-specific configuration

        Returns:
            BaseStorage instance

        Raises:
            ValueError: If provider is not supported
        """
        storage_type = auth.get_storage_type()

        # Validate provider matches auth type
        if provider != storage_type:
            raise ValueError(
                f"Storage provider '{provider}' does not match auth type "
                f"'{storage_type}'"
            )

        if provider == "oci":
            # Lazy import to avoid requiring OCI SDK if not used
            from genai_bench.storage.oci_storage import OCIObjectStorage

            return OCIObjectStorage(auth, **kwargs)
        elif provider == "aws":
            # Lazy import to avoid requiring AWS SDK if not used
            from genai_bench.storage.aws_storage import AWSS3Storage

            return AWSS3Storage(auth, **kwargs)
        elif provider == "azure":
            # Lazy import to avoid requiring Azure SDK if not used
            from genai_bench.storage.azure_storage import AzureBlobStorage

            return AzureBlobStorage(auth, **kwargs)
        elif provider == "gcp":
            # Lazy import to avoid requiring GCP SDK if not used
            from genai_bench.storage.gcp_storage import GCPCloudStorage

            return GCPCloudStorage(auth, **kwargs)
        elif provider == "github":
            # Lazy import to avoid requiring GitHub SDK if not used
            from genai_bench.storage.github_storage import GitHubStorage

            return GitHubStorage(auth, **kwargs)
        else:
            raise ValueError(
                f"Unsupported storage provider: {provider}. "
                f"Supported: oci, aws, azure, gcp, github"
            )
