from unittest.mock import mock_open, patch

import oci
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from genai_bench.auth.oci.session import OCISessionAuth

# Mock OCI config for testing
MOCK_CONFIG = {
    "fingerprint": "20:3b:97:13:55:1c:5b:0d:d3:37:d8:50:4e:c5:3a:34",
    "key_file": "~/.oci/sessions/TestProfile/oci_api_key.pem",
    "tenancy": "ocid1.tenancy.oc1..aaaaaaaa123456789genaibenchrocks",
    "region": "us-ashburn-1",
    "security_token_file": "~/.oci/sessions/TestProfile/token",
}

MOCK_SECURITY_TOKEN = "mock.security.token.contents"


class MockOCISigner(oci.signer.AbstractBaseSigner):
    def __init__(self):
        self.region = "us-ashburn-1"
        self.tenancy_id = "ocid1.tenancy.oc1..aaaaaaaa123456789genaibenchrocks"

    def sign(self, request):
        pass


@pytest.fixture
def mock_config_file(tmp_path):
    """Create a temporary OCI config file with session profile."""
    config_path = tmp_path / "config"
    with open(config_path, "w") as f:
        f.write("[DEFAULT]\\n")
        for key, value in MOCK_CONFIG.items():
            f.write(f"{key} = {value}\\n")
    return str(config_path)


class TestOCISessionAuth:
    def test_init_default_profile(self, mock_config_file):
        """Test initialization with default profile."""
        auth = OCISessionAuth(config_path=mock_config_file)
        assert auth.config_path == mock_config_file
        assert auth.profile == "DEFAULT"
        assert auth._config is None
        assert auth._signer is None

    def test_init_custom_profile(self, mock_config_file):
        """Test initialization with custom profile."""
        auth = OCISessionAuth(config_path=mock_config_file, profile="CUSTOM")
        assert auth.config_path == mock_config_file
        assert auth.profile == "CUSTOM"
        assert auth._config is None
        assert auth._signer is None

    @patch("oci.config.from_file")
    def test_get_config(self, mock_from_file, mock_config_file):
        """Test getting OCI config with security token."""
        mock_from_file.return_value = MOCK_CONFIG

        auth = OCISessionAuth(config_path=mock_config_file)
        config = auth.get_config()

        assert config == MOCK_CONFIG
        mock_from_file.assert_called_once_with(
            file_location=mock_config_file, profile_name="DEFAULT"
        )

    @patch("oci.config.from_file")
    def test_get_config_missing_token_file(self, mock_from_file, mock_config_file):
        """Test getting config without security token file."""
        config_without_token = {
            k: v for k, v in MOCK_CONFIG.items() if k != "security_token_file"
        }
        mock_from_file.return_value = config_without_token

        auth = OCISessionAuth(config_path=mock_config_file)
        with pytest.raises(oci.exceptions.InvalidConfig) as exc:
            auth.get_config()
        assert "security_token_file" in str(exc.value)

    @patch("oci.config.from_file")
    @patch("oci.signer.load_private_key_from_file")
    @patch("builtins.open", new_callable=mock_open, read_data=MOCK_SECURITY_TOKEN)
    def test_get_credentials(
        self, mock_file, mock_load_key, mock_from_file, mock_config_file
    ):
        """Test getting OCI signer with security token."""
        mock_from_file.return_value = MOCK_CONFIG

        # Create a real RSA key for testing
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        mock_load_key.return_value = private_key

        auth = OCISessionAuth(config_path=mock_config_file)
        signer = auth.get_credentials()

        assert isinstance(signer, oci.auth.signers.SecurityTokenSigner)
        mock_load_key.assert_called_once_with(MOCK_CONFIG["key_file"], None)
        mock_file.assert_called_once_with(MOCK_CONFIG["security_token_file"])

        # Test caching
        signer2 = auth.get_credentials()
        assert signer2 is signer  # Should return cached signer
