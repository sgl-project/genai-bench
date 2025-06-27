"""Unit tests for CLI validation with multi-cloud providers."""

from unittest.mock import MagicMock, patch

import click
import pytest

from genai_bench.cli.validation import (
    API_BACKEND_USER_MAP,
    validate_api_backend,
    validate_api_key,
    validate_task,
)
from genai_bench.user.aws_bedrock_user import AWSBedrockUser
from genai_bench.user.azure_openai_user import AzureOpenAIUser
from genai_bench.user.gcp_vertex_user import GCPVertexUser
from genai_bench.user.oci_cohere_user import OCICohereUser
from genai_bench.user.openai_user import OpenAIUser


class TestMultiCloudValidation:
    """Test cases for multi-cloud CLI validation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.ctx = MagicMock()
        self.ctx.params = {}
        self.ctx.obj = {}
        self.param = MagicMock()

    # API Backend Validation Tests

    def test_validate_openai_backend(self):
        """Test OpenAI backend validation."""
        result = validate_api_backend(self.ctx, self.param, "openai")

        assert result == "openai"
        assert self.ctx.obj["user_class"] == OpenAIUser

    def test_validate_aws_bedrock_backend(self):
        """Test AWS Bedrock backend validation."""
        result = validate_api_backend(self.ctx, self.param, "aws-bedrock")

        assert result == "aws-bedrock"
        assert self.ctx.obj["user_class"] == AWSBedrockUser

    def test_validate_azure_openai_backend(self):
        """Test Azure OpenAI backend validation."""
        result = validate_api_backend(self.ctx, self.param, "azure-openai")

        assert result == "azure-openai"
        assert self.ctx.obj["user_class"] == AzureOpenAIUser

    def test_validate_gcp_vertex_backend(self):
        """Test GCP Vertex AI backend validation."""
        result = validate_api_backend(self.ctx, self.param, "gcp-vertex")

        assert result == "gcp-vertex"
        assert self.ctx.obj["user_class"] == GCPVertexUser

    def test_validate_oci_cohere_backend(self):
        """Test OCI Cohere backend validation."""
        result = validate_api_backend(self.ctx, self.param, "oci-cohere")

        assert result == "oci-cohere"
        assert self.ctx.obj["user_class"] == OCICohereUser

    def test_validate_vllm_backend(self):
        """Test vLLM backend validation (uses OpenAI user)."""
        result = validate_api_backend(self.ctx, self.param, "vllm")

        assert result == "vllm"
        assert self.ctx.obj["user_class"] == OpenAIUser

    def test_validate_sglang_backend(self):
        """Test SGLang backend validation (uses OpenAI user)."""
        result = validate_api_backend(self.ctx, self.param, "sglang")

        assert result == "sglang"
        assert self.ctx.obj["user_class"] == OpenAIUser

    def test_validate_unsupported_backend(self):
        """Test unsupported backend validation."""
        with pytest.raises(click.BadParameter) as exc_info:
            validate_api_backend(self.ctx, self.param, "unsupported-backend")

        assert "unsupported-backend is not a supported API backend" in str(
            exc_info.value
        )

    def test_validate_backend_case_insensitive(self):
        """Test backend validation is case insensitive."""
        result = validate_api_backend(self.ctx, self.param, "AWS-BEDROCK")

        assert result == "aws-bedrock"
        assert self.ctx.obj["user_class"] == AWSBedrockUser

    # API Key Validation Tests

    def test_validate_api_key_openai_required(self):
        """Test API key is required for OpenAI."""
        self.ctx.params["api_backend"] = "openai"

        # No API key provided
        with pytest.raises(click.BadParameter) as exc_info:
            validate_api_key(self.ctx, self.param, None)

        assert "API key is required for openai backend" in str(exc_info.value)

    def test_validate_api_key_openai_provided(self):
        """Test API key validation for OpenAI with key provided."""
        self.ctx.params["api_backend"] = "openai"

        result = validate_api_key(self.ctx, self.param, "sk-test-key")
        assert result == "sk-test-key"

    def test_validate_api_key_vllm_required(self):
        """Test API key is required for vLLM."""
        self.ctx.params["api_backend"] = "vllm"

        with pytest.raises(click.BadParameter) as exc_info:
            validate_api_key(self.ctx, self.param, None)

        assert "API key is required for vllm backend" in str(exc_info.value)

    def test_validate_api_key_sglang_required(self):
        """Test API key is required for SGLang."""
        self.ctx.params["api_backend"] = "sglang"

        with pytest.raises(click.BadParameter) as exc_info:
            validate_api_key(self.ctx, self.param, None)

        assert "API key is required for sglang backend" in str(exc_info.value)

    def test_validate_api_key_azure_optional(self):
        """Test API key is optional for Azure OpenAI."""
        self.ctx.params["api_backend"] = "azure-openai"

        # No API key is fine for Azure (can use Azure AD)
        result = validate_api_key(self.ctx, self.param, None)
        assert result is None

        # API key can also be provided
        result = validate_api_key(self.ctx, self.param, "azure-key")
        assert result == "azure-key"

    @patch("click.echo")
    def test_validate_api_key_aws_bedrock_warns(self, mock_echo):
        """Test API key warning for AWS Bedrock."""
        self.ctx.params["api_backend"] = "aws-bedrock"

        result = validate_api_key(self.ctx, self.param, "some-key")

        assert result is None
        mock_echo.assert_called_once()
        assert "API key is not used for aws-bedrock backend" in str(mock_echo.call_args)

    @patch("click.echo")
    def test_validate_api_key_gcp_vertex_warns(self, mock_echo):
        """Test API key warning for GCP Vertex."""
        self.ctx.params["api_backend"] = "gcp-vertex"

        result = validate_api_key(self.ctx, self.param, "some-key")

        assert result is None
        mock_echo.assert_called_once()
        assert "API key is not used for gcp-vertex backend" in str(mock_echo.call_args)

    @patch("click.echo")
    def test_validate_api_key_oci_cohere_warns(self, mock_echo):
        """Test API key warning for OCI Cohere."""
        self.ctx.params["api_backend"] = "oci-cohere"

        result = validate_api_key(self.ctx, self.param, "some-key")

        assert result is None
        mock_echo.assert_called_once()
        assert "API key is not used for oci-cohere backend" in str(mock_echo.call_args)

    def test_validate_api_key_no_backend(self):
        """Test API key validation without backend specified."""
        # No api_backend in params
        with pytest.raises(click.BadParameter) as exc_info:
            validate_api_key(self.ctx, self.param, "test-key")

        assert "api_backend must be specified before api_key" in str(exc_info.value)

    # Task Validation Tests

    def test_validate_task_supported(self):
        """Test task validation for supported tasks."""
        # Set up user class
        self.ctx.obj["user_class"] = OpenAIUser

        result = validate_task(self.ctx, self.param, "text-to-text")

        assert result == "text-to-text"
        assert "user_task" in self.ctx.obj

    def test_validate_task_unsupported(self):
        """Test task validation for unsupported tasks."""
        # Mock a user class with limited task support
        mock_user_class = MagicMock()
        mock_user_class.is_task_supported.return_value = False
        mock_user_class.supported_tasks = {"text-to-text": "generate_text"}

        self.ctx.obj["user_class"] = mock_user_class

        with pytest.raises(click.BadParameter) as exc_info:
            validate_task(self.ctx, self.param, "unsupported-task")

        assert "Task 'unsupported-task' is not supported" in str(exc_info.value)
        assert "Supported tasks are: text-to-text" in str(exc_info.value)

    def test_validate_task_no_user_class(self):
        """Test task validation without user class set."""
        self.ctx.obj = {}  # No user_class

        with pytest.raises(click.BadParameter) as exc_info:
            validate_task(self.ctx, self.param, "text-to-text")

        assert "API backend is not set" in str(exc_info.value)

    def test_validate_task_case_insensitive(self):
        """Test task validation is case insensitive."""
        self.ctx.obj["user_class"] = OpenAIUser

        result = validate_task(self.ctx, self.param, "TEXT-TO-TEXT")

        assert result == "text-to-text"

    # Backend-Specific Task Support Tests

    def test_aws_bedrock_task_support(self):
        """Test AWS Bedrock supports expected tasks."""
        self.ctx.obj["user_class"] = AWSBedrockUser

        # Should support text-to-text
        result = validate_task(self.ctx, self.param, "text-to-text")
        assert result == "text-to-text"

        # Check other tasks based on what AWSBedrockUser supports
        # This depends on the actual implementation

    def test_azure_openai_task_support(self):
        """Test Azure OpenAI supports expected tasks."""
        self.ctx.obj["user_class"] = AzureOpenAIUser

        # Should support text-to-text
        result = validate_task(self.ctx, self.param, "text-to-text")
        assert result == "text-to-text"

    def test_gcp_vertex_task_support(self):
        """Test GCP Vertex AI supports expected tasks."""
        self.ctx.obj["user_class"] = GCPVertexUser

        # Should support text-to-text
        result = validate_task(self.ctx, self.param, "text-to-text")
        assert result == "text-to-text"

    # Integration Tests

    def test_full_validation_flow_openai(self):
        """Test full validation flow for OpenAI."""
        ctx = MagicMock()
        ctx.params = {}
        ctx.obj = None
        param = MagicMock()

        # Validate backend
        validate_api_backend(ctx, param, "openai")

        # Add backend to params for api_key validation
        ctx.params["api_backend"] = "openai"

        # Validate API key
        api_key = validate_api_key(ctx, param, "sk-test-key")
        assert api_key == "sk-test-key"

        # Validate task
        task = validate_task(ctx, param, "text-to-text")
        assert task == "text-to-text"

    def test_full_validation_flow_aws_bedrock(self):
        """Test full validation flow for AWS Bedrock."""
        ctx = MagicMock()
        ctx.params = {}
        ctx.obj = None
        param = MagicMock()

        # Validate backend
        validate_api_backend(ctx, param, "aws-bedrock")

        # Add backend to params
        ctx.params["api_backend"] = "aws-bedrock"

        # Validate API key (should return None)
        api_key = validate_api_key(ctx, param, None)
        assert api_key is None

        # Validate task
        task = validate_task(ctx, param, "text-to-text")
        assert task == "text-to-text"

    def test_full_validation_flow_azure_openai(self):
        """Test full validation flow for Azure OpenAI."""
        ctx = MagicMock()
        ctx.params = {}
        ctx.obj = None
        param = MagicMock()

        # Validate backend
        validate_api_backend(ctx, param, "azure-openai")

        # Add backend to params
        ctx.params["api_backend"] = "azure-openai"

        # Validate without API key (using Azure AD)
        api_key = validate_api_key(ctx, param, None)
        assert api_key is None

        # Validate task
        task = validate_task(ctx, param, "text-to-text")
        assert task == "text-to-text"

    def test_backend_user_map_completeness(self):
        """Test that all expected backends are in the user map."""
        expected_backends = [
            "openai",
            "oci-cohere",
            "cohere",
            "aws-bedrock",
            "azure-openai",
            "gcp-vertex",
            "vllm",
            "sglang",
        ]

        for backend in expected_backends:
            assert backend in API_BACKEND_USER_MAP
            assert API_BACKEND_USER_MAP[backend] is not None

    def test_backend_aliases(self):
        """Test backend aliases work correctly."""
        # vLLM and SGLang should use OpenAIUser
        assert API_BACKEND_USER_MAP["vllm"] == OpenAIUser
        assert API_BACKEND_USER_MAP["sglang"] == OpenAIUser
