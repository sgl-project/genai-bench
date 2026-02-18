import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import click
import pytest
from transformers import BertTokenizerFast

from genai_bench.cli.validation import (
    DEFAULT_BATCH_SIZES,
    DEFAULT_NUM_CONCURRENCIES,
    DEFAULT_SCENARIOS_BY_TASK,
    set_model_from_tokenizer,
    validate_additional_request_params,
    validate_api_backend,
    validate_api_key,
    validate_dataset_path_callback,
    validate_filter_criteria,
    validate_iteration_params,
    validate_object_storage_options,
    validate_scenario_callback,
    validate_task,
    validate_tokenizer,
    validate_traffic_scenario_callback,
    validate_warmup_cooldown_ratio_options,
)


def test_validate_scenario_callback():
    value = "D(100,100)"
    result = validate_scenario_callback(value)
    assert result == value

    with pytest.raises(click.BadParameter):
        validate_scenario_callback("invalid_scenario")


def test_validate_traffic_scenario_callback():
    ctx = click.Context(click.Command("test"))
    ctx.params = {"task": "text-to-text"}
    param = None

    # Test with user-provided scenarios
    user_input = ["D(100,100)", "D(50,50)"]
    result = validate_traffic_scenario_callback(ctx, param, user_input)
    assert result == user_input

    # Test with default scenarios
    result = validate_traffic_scenario_callback(ctx, param, None)
    assert result == DEFAULT_SCENARIOS_BY_TASK["text-to-text"]

    # Test with missing task
    ctx.params = {}
    with pytest.raises(click.BadParameter) as exc:
        validate_traffic_scenario_callback(ctx, param, None)
    assert "The '--task' option is required" in str(exc.value)

    # Test with invalid task
    ctx.params = {"task": "unknown_task"}
    with pytest.raises(click.BadParameter) as exc:
        validate_traffic_scenario_callback(ctx, param, None)
    assert "No default traffic scenarios defined for task 'unknown_task'" in str(
        exc.value
    )


def test_validate_api_backend():
    ctx = click.Context(click.Command("test"))
    ctx.obj = {}
    param = None

    result = validate_api_backend(ctx, param, "openai")
    assert result == "openai"
    assert "user_class" in ctx.obj

    with pytest.raises(click.BadParameter):
        validate_api_backend(ctx, param, "invalid_backend")


def test_validate_task():
    # Mocked user class with supported tasks
    class MockUserClass:
        supported_tasks = {
            "text-to-text": "chat",
            "text-to-embeddings": "embeddings",
        }

        @staticmethod
        def is_task_supported(task):
            return task in MockUserClass.supported_tasks

        def chat(self):
            pass

        def embeddings(self):
            pass

    # Case 1: Valid task with user class supporting the task
    ctx = click.Context(click.Command("test"))
    ctx.obj = {"user_class": MockUserClass}
    param = None

    result = validate_task(ctx, param, "text-to-text")
    assert result == "text-to-text"
    assert "user_task" in ctx.obj
    assert ctx.obj["user_task"] == MockUserClass.chat

    # Case 2: Invalid task not supported by user class
    with pytest.raises(click.BadParameter) as excinfo:
        validate_task(ctx, param, "invalid_task")
    assert "is not supported by the selected API backend" in str(excinfo.value)

    # Case 3: Missing user_class in context
    ctx = click.Context(click.Command("test"))
    ctx.obj = {}
    with pytest.raises(click.BadParameter) as excinfo:
        validate_task(ctx, param, "text-to-text")
    assert "API backend is not set" in str(excinfo.value)

    # Case 4: Missing context object entirely
    ctx = click.Context(click.Command("test"))
    ctx.obj = None
    with pytest.raises(click.BadParameter) as excinfo:
        validate_task(ctx, param, "text-to-text")
    assert "API backend is not set" in str(excinfo.value)


def test_validate_tokenizer_with_local_path(mock_tokenizer_path):
    tokenizer = validate_tokenizer(mock_tokenizer_path)

    assert isinstance(tokenizer, BertTokenizerFast)


def test_validate_tokenizer_with_hf_api(monkeypatch):
    hf_token = "mock_api_key"
    mock_tokenizer = MagicMock()

    monkeypatch.setenv("HF_TOKEN", hf_token)

    with patch(
        "transformers.AutoTokenizer.from_pretrained",
        return_value=mock_tokenizer,
    ) as mock_from_pretrained:
        model_name = "bert-base-uncased"
        monkeypatch.setattr(Path, "exists", lambda self: False)

        tokenizer = validate_tokenizer(model_name)

        mock_from_pretrained.assert_called_once_with(model_name, token=hf_token)
        assert tokenizer == mock_tokenizer


def test_validate_tokenizer_no_hf_token(monkeypatch):
    """Test tokenizer validation with public model succeeds without token."""
    monkeypatch.setattr(Path, "exists", lambda self: False)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    mock_tokenizer = MagicMock()
    with patch(
        "transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer
    ) as mock_from_pretrained:
        model_name = "bert-base-uncased"
        tokenizer = validate_tokenizer(model_name)
        mock_from_pretrained.assert_called_once_with(model_name, token=None)
        assert tokenizer == mock_tokenizer


def test_validate_tokenizer_access_errors(monkeypatch):
    """Test tokenizer validation catches access errors for restricted repos."""
    monkeypatch.setattr(Path, "exists", lambda self: False)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    # Test gated repo error
    gated_error = OSError(
        "You are trying to access a gated repo.\n"
        "Make sure to have access to it at https://huggingface.co/meta-llama/Meta-Llama-3-8B."
    )
    with patch("transformers.AutoTokenizer.from_pretrained", side_effect=gated_error):
        with pytest.raises(click.BadParameter) as exc_info:
            validate_tokenizer("meta-llama/Meta-Llama-3-8B")
        assert "Hugging Face requires authentication" in str(exc_info.value)

    # Test private repo error
    private_error = OSError(
        "private/model is not a local folder and is not a valid model identifier\n"
        "If this is a private repository, make sure to pass a token"
    )
    with patch("transformers.AutoTokenizer.from_pretrained", side_effect=private_error):
        with pytest.raises(click.BadParameter) as exc_info:
            validate_tokenizer("private/model")
        assert "Hugging Face requires authentication" in str(exc_info.value)


def test_set_model_from_tokenizer():
    ctx = click.Context(click.Command("test"))
    ctx.params = {"model_tokenizer": "openai/gpt2"}
    param = None

    result = set_model_from_tokenizer(ctx, param, None)
    assert result == "gpt2"

    result = set_model_from_tokenizer(ctx, param, "custom_model")
    assert result == "custom_model"


def test_validate_additional_request_params(caplog):
    ctx = click.Context(click.Command("test"))
    param = None

    result = validate_additional_request_params(ctx, param, '{"temperature": 0.7}')
    assert result == {"temperature": 0.7}

    with pytest.raises(click.BadParameter):
        validate_additional_request_params(ctx, param, "invalid_json")

    # Test ignore_eos warning
    with (
        caplog.at_level(logging.WARNING),
        patch("click.confirm", return_value=False),
    ):
        result = validate_additional_request_params(ctx, param, '{"ignore_eos": false}')
        assert result == {"ignore_eos": False}
    assert (
        "You have set ignore_eos to False. This will cause inaccurate "
        "num_output_tokens because the generation can stop before reaching "
        "the num_output_tokens you passed. Unless you are doing correctness"
        " test, it is recommended to set ignore_eos to True."
    ) in caplog.text


def test_validate_filter_criteria():
    ctx = click.Context(click.Command("test"))
    param = None
    value = None

    result = validate_filter_criteria(ctx, param, value)
    assert result == {}

    value = '{"server_gpu_type": 4}'
    result = validate_filter_criteria(ctx, param, value)
    assert result == {"server_gpu_type": 4}


def test_validate_dataset_path_callback(caplog):
    ctx = click.Context(click.Command("test"))
    param = None

    # Test with image task and dataset path
    ctx.params = {"task": "image-text-to-text"}
    result = validate_dataset_path_callback(ctx, param, "/path/to/dataset")
    assert result == "/path/to/dataset"

    # Test with image task and missing dataset path (no dataset config)
    ctx.params = {"task": "image-text-to-text", "dataset_config": None}
    with pytest.raises(click.BadParameter) as exc:
        validate_dataset_path_callback(ctx, param, None)
    assert (
        '--dataset-path is required when --task includes "image" input modality '
        "and --dataset-config is not provided." in str(exc.value)
    )

    # Test with image task, missing dataset path, but dataset config provided
    with caplog.at_level(logging.WARNING):
        ctx.params = {
            "task": "image-text-to-text",
            "dataset_config": "/path/to/config.json",
        }
        result = validate_dataset_path_callback(ctx, param, None)
        assert result is None
    assert "Using dataset configuration file for image task" in caplog.text

    # Test with text task (dataset path optional)
    ctx.params = {"task": "text-to-text"}
    result = validate_dataset_path_callback(ctx, param, None)
    assert result is None

    # Test with missing task
    ctx.params = {}
    with pytest.raises(click.BadParameter) as exc:
        validate_dataset_path_callback(ctx, param, None)
    assert "The '--task' option is required" in str(exc.value)


def test_validate_traffic_scenarios():
    ctx = click.Context(click.Command("test"))
    ctx.params = {"task": "text-to-text"}
    param = None

    # Test with user-provided scenarios
    user_input = ["D(100,100)", "D(50,50)"]
    result = validate_traffic_scenario_callback(ctx, param, user_input)
    assert result == user_input

    # Test with default scenarios
    result = validate_traffic_scenario_callback(ctx, param, None)
    assert result == DEFAULT_SCENARIOS_BY_TASK["text-to-text"]

    # Test with missing task
    ctx.params = {}
    with pytest.raises(click.BadParameter) as exc:
        validate_traffic_scenario_callback(ctx, param, None)
    assert "The '--task' option is required" in str(exc.value)

    # Test with invalid task
    ctx.params = {"task": "unknown_task"}
    with pytest.raises(click.BadParameter) as exc:
        validate_traffic_scenario_callback(ctx, param, None)
    assert "No default traffic scenarios defined for task 'unknown_task'" in str(
        exc.value
    )

    # Test with dataset-mode
    ctx.params = {"task": "text-to-text", "dataset_path": "path/to/dataset"}
    result = validate_traffic_scenario_callback(ctx, param, None)
    assert result == ["dataset"]


def test_validate_iteration_params():
    """Test validation of iteration parameters for different tasks."""
    ctx = click.Context(click.Command("test"))
    param = None

    # Test embedding task (should use batch_size)
    ctx.params = {
        "task": "text-to-embeddings",
        "num_concurrency": [1, 2, 4],
        "batch_size": [8, 16, 32],
    }
    result = validate_iteration_params(ctx, param, "num_concurrency")
    assert result == "batch_size"
    assert ctx.params["batch_size"] == [8, 16, 32]
    assert ctx.params["num_concurrency"] == [1]

    # Test chat task (should use num_concurrency)
    ctx.params = {
        "task": "text-to-text",
        "num_concurrency": [1, 2, 4],
        "batch_size": [8, 16, 32],
    }
    result = validate_iteration_params(ctx, param, "batch_size")
    assert result == "num_concurrency"
    assert ctx.params["batch_size"] == [1]
    assert ctx.params["num_concurrency"] == [1, 2, 4]


def test_validate_iteration_params_with_defaults():
    """Test validate_iteration_params with default values."""
    ctx = click.Context(click.Command("test"))
    param = None

    # Test with default values for text-to-text
    with patch("click.echo") as mock_echo:
        ctx.params = {
            "task": "text-to-text",
            "num_concurrency": [],
            "batch_size": [],
        }
        result = validate_iteration_params(ctx, param, "batch_size")
        mock_echo.assert_called_once_with(
            "Note: Using num_concurrency iteration for text-to-text task"
        )

        assert result == "num_concurrency"
        assert ctx.params["batch_size"] == [1]
        assert ctx.params["num_concurrency"] == DEFAULT_NUM_CONCURRENCIES

    # Test with default values for text-to-embeddings
    with patch("click.echo") as mock_echo:
        ctx.params = {
            "task": "text-to-embeddings",
            "num_concurrency": [],
            "batch_size": [],
        }
        result = validate_iteration_params(ctx, param, "num_concurrency")
        mock_echo.assert_called_once_with(
            "Note: Using batch_size iteration for text-to-embeddings task"
        )

        assert result == "batch_size"
        assert ctx.params["batch_size"] == DEFAULT_BATCH_SIZES
        assert ctx.params["num_concurrency"] == [1]


def test_validate_iteration_params_with_request_rate():
    """Test validate_iteration_params sets iteration_type to request_rate."""
    ctx = click.Context(click.Command("test"))
    param = None

    # Test with request_rate provided (without explicit num_concurrency)
    with patch("click.echo") as mock_echo:
        ctx.params = {
            "task": "text-to-text",
            # Defaults, not explicitly provided
            "num_concurrency": DEFAULT_NUM_CONCURRENCIES,
            "batch_size": [8, 16],
            "request_rate": [5, 10, 20],
        }
        result = validate_iteration_params(ctx, param, "num_concurrency")
        mock_echo.assert_called_once_with(
            "Note: Using request_rate iteration since --request-rate was provided"
        )

        assert result == "request_rate"
        assert ctx.params["request_rate"] == [5, 10, 20]
        assert ctx.params["num_concurrency"] == [1]
        assert ctx.params["batch_size"] == [1]

    # Test with request_rate and explicit iteration_type
    # (without explicit num_concurrency)
    ctx.params = {
        "task": "text-to-text",
        "num_concurrency": DEFAULT_NUM_CONCURRENCIES,  # Defaults
        "request_rate": [10],
    }
    result = validate_iteration_params(ctx, param, "request_rate")
    assert result == "request_rate"


def test_validate_iteration_params_request_rate_validates_positive():
    """Test validate_iteration_params validates positive request_rate values."""
    ctx = click.Context(click.Command("test"))
    param = None

    # Test with zero value
    ctx.params = {
        "task": "text-to-text",
        "request_rate": [0, 5],
    }
    with pytest.raises(
        click.BadParameter, match="All request_rate values must be positive"
    ):
        validate_iteration_params(ctx, param, "num_concurrency")

    # Test with negative value
    ctx.params = {
        "task": "text-to-text",
        "request_rate": [-1, 5],
    }
    with pytest.raises(
        click.BadParameter, match="All request_rate values must be positive"
    ):
        validate_iteration_params(ctx, param, "num_concurrency")

    # Test with all positive values (should pass)
    ctx.params = {
        "task": "text-to-text",
        "request_rate": [1, 5, 10],
    }
    result = validate_iteration_params(ctx, param, "num_concurrency")
    assert result == "request_rate"


def test_validate_iteration_params_mutually_exclusive():
    """Test that request_rate and num_concurrency are mutually exclusive."""
    ctx = click.Context(click.Command("test"))
    param = None

    # Test that providing both explicitly raises an error
    ctx.params = {
        "task": "text-to-text",
        "num_concurrency": [1, 5, 10],  # Explicitly provided, not defaults
        "request_rate": [2, 5],
    }
    with pytest.raises(
        click.BadParameter,
        match="--num-concurrency and --request-rate are mutually exclusive",
    ):
        validate_iteration_params(ctx, param, "num_concurrency")

    # Test with different explicit num_concurrency values
    ctx.params = {
        "task": "text-to-text",
        "num_concurrency": [2, 4, 8],  # Explicitly provided
        "request_rate": [1],
    }
    with pytest.raises(
        click.BadParameter,
        match="--num-concurrency and --request-rate are mutually exclusive",
    ):
        validate_iteration_params(ctx, param, "num_concurrency")

    # Test that defaults are converted to [1] when request_rate is provided
    # (treating defaults as "not provided")
    ctx.params = {
        "task": "text-to-text",
        "num_concurrency": DEFAULT_NUM_CONCURRENCIES,  # Defaults
        "request_rate": [2, 5],
    }
    result = validate_iteration_params(ctx, param, "num_concurrency")
    assert result == "request_rate"
    # Defaults should be converted to [1] when request_rate is provided
    assert ctx.params["num_concurrency"] == [1]


def test_validate_iteration_params_request_rate_with_batch_size_task():
    """Test that batch_size task still uses batch_size even with request_rate."""
    ctx = click.Context(click.Command("test"))
    param = None

    # For embeddings/rerank tasks, batch_size should take precedence
    ctx.params = {
        "task": "text-to-embeddings",
        "num_concurrency": [1, 2],
        "batch_size": [8, 16],
        "request_rate": [5, 10],
    }
    result = validate_iteration_params(ctx, param, "request_rate")
    # Should still use batch_size for embeddings
    assert result == "batch_size"
    assert ctx.params["batch_size"] == [8, 16]
    assert ctx.params["num_concurrency"] == [1]

    # Same for rerank
    ctx.params = {
        "task": "text-to-rerank",
        "batch_size": [4, 8],
        "request_rate": [5],
    }
    result = validate_iteration_params(ctx, param, "request_rate")
    assert result == "batch_size"


def test_validate_iteration_params_max_concurrency_requires_request_rate():
    """Test that max_concurrency can only be used with request_rate."""
    ctx = click.Context(click.Command("test"))
    param = None

    # Test that providing max_concurrency without request_rate raises an error
    ctx.params = {
        "task": "text-to-text",
        "num_concurrency": [1, 2, 4],
        "max_concurrency": 1000,
        "request_rate": None,
    }
    with pytest.raises(
        click.BadParameter,
        match="--max-concurrency can only be used with --request-rate",
    ):
        validate_iteration_params(ctx, param, "num_concurrency")

    # Test that providing max_concurrency with request_rate is allowed
    ctx.params = {
        "task": "text-to-text",
        "num_concurrency": DEFAULT_NUM_CONCURRENCIES,
        "max_concurrency": 1000,
        "request_rate": [5, 10],
    }
    result = validate_iteration_params(ctx, param, "num_concurrency")
    assert result == "request_rate"

    # Test that not providing max_concurrency (None) with num_concurrency is allowed
    ctx.params = {
        "task": "text-to-text",
        "num_concurrency": [1, 2, 4],
        "max_concurrency": None,
        "request_rate": None,
    }
    result = validate_iteration_params(ctx, param, "num_concurrency")
    assert result == "num_concurrency"


def test_validate_object_storage_options():
    """Test validation of object storage options."""
    # Mock Click context and param
    ctx = MagicMock()
    param = MagicMock()

    # Test case 1: When upload_results is False, any value should pass
    param.name = "upload_results"
    ctx.params = {}
    assert validate_object_storage_options(ctx, param, False) is False

    # Test case 2: When upload_results is True and storage_bucket is in params
    param.name = "upload_results"
    ctx.params = {"storage_bucket": "test-bucket"}
    assert validate_object_storage_options(ctx, param, True) is True

    # Test case 3: When upload_results is True but storage_bucket is not provided
    param.name = "upload_results"
    ctx.params = {}
    with pytest.raises(click.UsageError) as exc_info:
        validate_object_storage_options(ctx, param, True)
    assert "You must provide a storage bucket name (--storage-bucket)" in str(
        exc_info.value
    )

    # Test case 4: When validating storage_bucket parameter
    param.name = "storage_bucket"
    ctx.params = {"upload_results": True}

    # Valid storage bucket name
    assert validate_object_storage_options(ctx, param, "test-bucket") == "test-bucket"


@patch("click.confirm")
def test_validate_api_key_cohere_warning(mock_confirm):
    """Test API key validation for Cohere backend with warning."""
    ctx = MagicMock()
    param = MagicMock()
    ctx.params = {"api_backend": "oci-cohere"}

    # Test warning when API key is provided for Cohere
    result = validate_api_key(ctx, param, "test-key")
    assert result is None  # API key should be ignored for Cohere

    # Test when api_backend is not set
    ctx.params = {}
    with pytest.raises(click.BadParameter) as exc_info:
        validate_api_key(ctx, param, "test-key")
    assert "api_backend must be specified before api_key" in str(exc_info.value)


@patch("click.confirm")
@patch("click.prompt")
def test_validate_additional_request_params_warnings(mock_prompt, mock_confirm):
    """Test warnings in additional request params validation."""
    ctx = MagicMock()
    param = MagicMock()

    # Test ignore_eos warning with user choosing to change
    mock_confirm.return_value = True
    result = validate_additional_request_params(ctx, param, '{"ignore_eos": false}')
    assert result["ignore_eos"] is True


def test_validate_filter_criteria_invalid_json():
    """Test filter criteria validation with invalid JSON."""
    ctx = MagicMock()
    param = MagicMock()

    # Test invalid JSON
    with pytest.raises(click.BadParameter) as exc_info:
        validate_filter_criteria(ctx, param, "{invalid json}")
    assert "Invalid JSON string in `--filter-criteria`" in str(exc_info.value)


def test_validate_warmup_cooldown_ratio_options():
    """Test validation of warmup and cooldown ratio options."""
    ctx = MagicMock()
    param = MagicMock()

    # Test with valid warmup_ratio value
    ctx.params = {
        "warmup_ratio": 0.1,
    }
    validate_warmup_cooldown_ratio_options(ctx, param, None)
    assert ctx.params["warmup_ratio"] == 0.1

    # Test with valid cooldown_ratio value
    ctx.params = {
        "cooldown_ratio": 0.1,
    }
    validate_warmup_cooldown_ratio_options(ctx, param, 0.1)
    assert ctx.params["cooldown_ratio"] == 0.1

    # Test with valid warmup_ratio and cooldown_ratio value
    ctx.params = {
        "warmup_ratio": 0.1,
        "cooldown_ratio": 0.1,
    }
    validate_warmup_cooldown_ratio_options(ctx, param, 0.1)
    assert ctx.params["warmup_ratio"] == 0.1
    assert ctx.params["cooldown_ratio"] == 0.1

    # Test with warmup_ratio + cooldown_ratio >= 1.0
    with pytest.raises(click.BadParameter):
        ctx.params = {
            "warmup_ratio": 0.5,
            "cooldown_ratio": 0.5,
        }
        validate_warmup_cooldown_ratio_options(ctx, param, 0.5)


def test_validate_iteration_params_request_rate_required():
    """Test that request_rate is required when iteration_type is 'request_rate'."""
    ctx = click.Context(click.Command("test"))
    param = None

    # Test that setting iteration_type to request_rate without request_rate raises error
    ctx.params = {
        "task": "text-to-text",
        "request_rate": None,
    }
    with pytest.raises(click.BadParameter) as exc_info:
        validate_iteration_params(ctx, param, "request_rate")
    assert "--request-rate is required" in str(exc_info.value)
