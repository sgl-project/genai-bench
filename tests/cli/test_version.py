from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

import genai_bench.cli.cli
import genai_bench.version


@pytest.fixture
def cli_runner():
    return CliRunner()


def test_version_override(cli_runner):
    mock_version = MagicMock(return_value="1.2.3")

    # Patch at the source - importlib.metadata.version
    with patch("importlib.metadata.version", mock_version):
        # Force reload of both modules to pick up the mock
        import importlib

        importlib.reload(genai_bench.version)
        importlib.reload(genai_bench.cli.cli)

        result = cli_runner.invoke(genai_bench.cli.cli.cli, ["--version"])

        assert result.exit_code == 0, f"Command failed with output: {result.output}"
        assert result.output == "genai-bench version 1.2.3\n"
        mock_version.assert_called_with("genai-bench")
