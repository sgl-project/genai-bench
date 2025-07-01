"""Tests for OCI authentication providers."""

from unittest.mock import patch

import oci
import pytest

from genai_bench.auth.oci.instance_principal import OCIInstancePrincipalAuth
from genai_bench.auth.oci.obo_token import OCIOBOTokenAuth
from genai_bench.auth.oci.user_principal import OCIUserPrincipalAuth

# Mock OCI config for testing
MOCK_CONFIG = {
    "user": "ocid1.user.oc1..aaaaaaaa123456789genaibenchrocks",
    "fingerprint": "20:3b:97:13:55:1c:5b:0d:d3:37:d8:50:4e:c5:3a:34",
    "key_file": "~/.oci/test_key.pem",
    "tenancy": "ocid1.tenancy.oc1..aaaaaaaa123456789genaibenchrocks",
    "region": "us-ashburn-1",
}

MOCK_REGION = "us-ashburn-1"
MOCK_TOKEN = "test.security.token"


class MockOCISigner(oci.signer.AbstractBaseSigner):
    def __init__(self):
        self.region = "us-ashburn-1"
        self.tenancy_id = "ocid1.tenancy.oc1..aaaaaaaa123456789genaibenchrocks"

    def sign(self, request):
        pass


@pytest.fixture
def mock_config_file(tmp_path):
    """Create a temporary OCI config file with multiple profiles."""
    config_path = tmp_path / "config"
    with open(config_path, "w") as f:
        # Default profile
        f.write("[DEFAULT]\n")
        for key, value in MOCK_CONFIG.items():
            f.write(f"{key} = {value}\n")

        # Additional profile
        f.write("[CUSTOM_PROFILE]\n")
        for key, value in MOCK_CONFIG.items():
            f.write(f"{key} = custom_{value}\n")
    return str(config_path)


class TestOCIUserPrincipalAuth:
    def test_init_default_profile(self, mock_config_file):
        """Test initialization with default profile."""
        auth = OCIUserPrincipalAuth(config_path=mock_config_file)
        assert auth.config_path == mock_config_file
        assert auth.profile == "DEFAULT"
        assert auth._config is None
        assert auth._signer is None

    def test_init_custom_profile(self, mock_config_file):
        """Test initialization with custom profile."""
        auth = OCIUserPrincipalAuth(
            config_path=mock_config_file, profile="CUSTOM_PROFILE"
        )
        assert auth.config_path == mock_config_file
        assert auth.profile == "CUSTOM_PROFILE"
        assert auth._config is None
        assert auth._signer is None

    @patch("oci.config.validate_config")
    @patch("oci.config.from_file")
    def test_get_config_default_profile(
        self, mock_from_file, mock_validate, mock_config_file
    ):
        """Test getting OCI config with default profile."""
        mock_from_file.return_value = MOCK_CONFIG.copy()

        auth = OCIUserPrincipalAuth(config_path=mock_config_file)
        config = auth.get_config()

        assert config == MOCK_CONFIG
        mock_from_file.assert_called_once_with(mock_config_file, "DEFAULT")
        mock_validate.assert_called_once_with(MOCK_CONFIG)

        # Second call should use cached config
        config = auth.get_config()
        assert config == MOCK_CONFIG
        mock_from_file.assert_called_once()
        mock_validate.assert_called_once()

    @patch("oci.config.validate_config")
    @patch("oci.config.from_file")
    def test_get_config_custom_profile(
        self, mock_from_file, mock_validate, mock_config_file
    ):
        """Test getting OCI config with custom profile."""
        custom_config = {k: f"custom_{v}" for k, v in MOCK_CONFIG.items()}
        mock_from_file.return_value = custom_config

        auth = OCIUserPrincipalAuth(
            config_path=mock_config_file, profile="CUSTOM_PROFILE"
        )
        config = auth.get_config()

        assert config == custom_config
        mock_from_file.assert_called_once_with(mock_config_file, "CUSTOM_PROFILE")
        mock_validate.assert_called_once_with(custom_config)

    @patch("oci.signer.Signer.from_config")
    @patch("oci.config.from_file")
    def test_get_credentials(self, mock_from_file, mock_from_config, mock_config_file):
        """Test getting OCI signer."""
        mock_from_file.return_value = MOCK_CONFIG.copy()
        mock_signer = MockOCISigner()
        mock_from_config.return_value = mock_signer

        auth = OCIUserPrincipalAuth(config_path=mock_config_file)
        signer = auth.get_credentials()

        assert signer == mock_signer
        mock_from_file.assert_called_once_with(mock_config_file, "DEFAULT")
        mock_from_config.assert_called_once_with(MOCK_CONFIG)

        # Second call should use cached signer
        signer = auth.get_credentials()
        assert signer == mock_signer
        mock_from_file.assert_called_once()
        mock_from_config.assert_called_once()

    @patch("oci.config.validate_config")
    @patch("oci.config.from_file")
    def test_config_validation(self, mock_from_file, mock_validate, mock_config_file):
        """Test config validation raises exception for invalid config."""
        mock_from_file.return_value = MOCK_CONFIG.copy()
        mock_validate.side_effect = oci.exceptions.InvalidConfig(
            {"fingerprint": "malformed"}
        )

        auth = OCIUserPrincipalAuth(config_path=mock_config_file)

        with pytest.raises(oci.exceptions.InvalidConfig):
            auth.get_config()


class TestOCIInstancePrincipalAuth:
    def test_init(self):
        """Test initialization."""
        auth = OCIInstancePrincipalAuth()
        assert auth._signer is None

    @patch("oci.auth.signers.InstancePrincipalsSecurityTokenSigner")
    def test_get_credentials(self, mock_signer_class):
        """Test getting instance principal signer."""
        mock_signer = MockOCISigner()
        mock_signer_class.return_value = mock_signer

        auth = OCIInstancePrincipalAuth()
        signer = auth.get_credentials()

        assert signer == mock_signer
        mock_signer_class.assert_called_once()

        # Second call should use cached signer
        signer = auth.get_credentials()
        assert signer == mock_signer
        mock_signer_class.assert_called_once()

    @patch("oci.auth.signers.InstancePrincipalsSecurityTokenSigner")
    def test_get_config(self, mock_signer_class):
        """Test getting config with instance principal."""
        mock_signer = MockOCISigner()
        mock_signer_class.return_value = mock_signer

        auth = OCIInstancePrincipalAuth()
        config = auth.get_config()

        assert isinstance(config, dict)
        assert "region" in config
        assert "tenancy" in config
        assert config["region"] == mock_signer.region
        assert config["tenancy"] == mock_signer.tenancy_id

    def test_init_with_token(self):
        """Test initialization with security token."""
        auth = OCIInstancePrincipalAuth(security_token=MOCK_TOKEN)
        assert auth._signer is None

    def test_init_with_token_and_region(self):
        """Test initialization with security token and region."""
        auth = OCIInstancePrincipalAuth(security_token=MOCK_TOKEN, region=MOCK_REGION)
        assert auth._signer is None
        assert auth.region == MOCK_REGION


class TestOCIOBOTokenAuth:
    def test_init(self):
        """Test initialization."""
        auth = OCIOBOTokenAuth(token=MOCK_TOKEN, region=MOCK_REGION)
        assert auth.token == MOCK_TOKEN
        assert auth.region == MOCK_REGION
        assert auth._signer is None

    @patch("oci.auth.signers.SecurityTokenSigner")
    def test_get_credentials(self, mock_signer_class):
        """Test getting security token signer."""
        mock_signer = MockOCISigner()
        mock_signer_class.return_value = mock_signer

        auth = OCIOBOTokenAuth(token=MOCK_TOKEN, region=MOCK_REGION)
        signer = auth.get_credentials()

        assert signer == mock_signer
        mock_signer_class.assert_called_once_with(MOCK_TOKEN, region=MOCK_REGION)

        # Second call should use cached signer
        signer = auth.get_credentials()
        assert signer == mock_signer
        mock_signer_class.assert_called_once()

    def test_get_config(self):
        """Test getting config with OBO token."""
        auth = OCIOBOTokenAuth(token=MOCK_TOKEN, region=MOCK_REGION)
        config = auth.get_config()

        assert config == {"region": MOCK_REGION}
