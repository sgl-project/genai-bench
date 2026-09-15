import copy
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import click
import pytest
from click.core import ParameterSource
from click.testing import CliRunner
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
from genai_bench.user.oci_cohere_user import OCICohereUser
from genai_bench.user.oci_cohere_v2_user import OCICohereV2User


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

    # Case 5: text-to-rerank with allowed backend (vllm)
    class MockReRankUserClass:
        supported_tasks = {
            "text-to-rerank": "rerank",
        }

        @staticmethod
        def is_task_supported(task):
            return task in MockReRankUserClass.supported_tasks

        def rerank(self):
            pass

    ctx = click.Context(click.Command("test"))
    ctx.params = {"api_backend": "vllm"}
    ctx.obj = {"user_class": MockReRankUserClass}
    result = validate_task(ctx, param, "text-to-rerank")
    assert result == "text-to-rerank"

    # Case 6: text-to-rerank with disallowed backend (openai)
    ctx = click.Context(click.Command("test"))
    ctx.params = {"api_backend": "openai"}
    ctx.obj = {"user_class": MockReRankUserClass}
    with pytest.raises(click.BadParameter) as excinfo:
        validate_task(ctx, param, "text-to-rerank")
    assert "not supported by the 'openai' backend" in str(excinfo.value)


def test_validate_task_oci_cohere_v2_invalid_task():
    ctx = click.Context(click.Command("test"))
    ctx.params = {"api_backend": "oci-cohere-v2"}
    ctx.obj = {"user_class": OCICohereV2User}

    with pytest.raises(click.BadParameter) as excinfo:
        validate_task(ctx, None, "text-to-embeddings")
    assert "not supported by the selected API backend" in str(excinfo.value)


def test_validate_task_oci_cohere_v2_allows_chat():
    ctx = click.Context(click.Command("test"))
    ctx.params = {"api_backend": "oci-cohere-v2"}
    ctx.obj = {"user_class": OCICohereV2User}

    result = validate_task(ctx, None, "text-to-text")
    assert result == "text-to-text"


def test_validate_task_oci_cohere_v1_rejects_image_chat():
    ctx = click.Context(click.Command("test"))
    ctx.params = {"api_backend": "oci-cohere"}
    ctx.obj = {"user_class": OCICohereUser}

    with pytest.raises(click.BadParameter) as excinfo:
        validate_task(ctx, None, "image-text-to-text")
    assert "not supported by the selected API backend" in str(excinfo.value)


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


def _iteration_ctx(task, num_concurrency, batch_size, explicit=()):
    """Build a Context for validate_iteration_params with known param sources."""
    ctx = click.Context(click.Command("test"))
    ctx.params = {
        "task": task,
        "num_concurrency": num_concurrency,
        "batch_size": batch_size,
    }
    for name in ("task", "num_concurrency", "batch_size", "iteration_type"):
        ctx.set_parameter_source(
            name,
            ParameterSource.COMMANDLINE
            if name in explicit
            else ParameterSource.DEFAULT,
        )
    return ctx


def test_validate_iteration_params():
    """Test validation of iteration parameters for different tasks."""
    param = None

    # Pooling tasks keep iterating over batch_size when --iteration-type is
    # left at its default
    for task in ("text-to-embeddings", "text-to-rerank"):
        ctx = _iteration_ctx(task, [1, 2, 4], [8, 16, 32])
        result = validate_iteration_params(ctx, param, "num_concurrency")
        assert result == "batch_size"
        assert ctx.params["batch_size"] == [8, 16, 32]
        assert ctx.params["num_concurrency"] == [1]

    # Test chat task (should use num_concurrency)
    ctx = _iteration_ctx("text-to-text", [1, 2, 4], [8, 16, 32])
    result = validate_iteration_params(ctx, param, "batch_size")
    assert result == "num_concurrency"
    assert ctx.params["batch_size"] == [1]
    assert ctx.params["num_concurrency"] == [1, 2, 4]


def test_validate_iteration_params_honors_explicit_concurrency_for_pooling():
    """An explicit --iteration-type wins for embeddings and rerank."""
    param = None

    for task in ("text-to-embeddings", "text-to-rerank"):
        ctx = _iteration_ctx(
            task,
            [1, 2, 4],
            [8, 16, 32],
            explicit=("iteration_type", "num_concurrency"),
        )
        result = validate_iteration_params(ctx, param, "num_concurrency")
        assert result == "num_concurrency"
        assert ctx.params["num_concurrency"] == [1, 2, 4]
        assert ctx.params["batch_size"] == [1]


def test_validate_iteration_params_explicit_concurrency_uses_defaults():
    """--iteration-type num_concurrency without --num-concurrency defaults."""
    ctx = _iteration_ctx("text-to-embeddings", [], [], explicit=("iteration_type",))
    result = validate_iteration_params(ctx, None, "num_concurrency")
    assert result == "num_concurrency"
    assert ctx.params["num_concurrency"] == DEFAULT_NUM_CONCURRENCIES
    assert ctx.params["batch_size"] == [1]


def test_validate_iteration_params_honors_explicit_batch_size_for_pooling():
    """--iteration-type batch_size stays on the batch_size axis."""
    ctx = _iteration_ctx(
        "text-to-embeddings", [1], [8, 16], explicit=("iteration_type", "batch_size")
    )
    result = validate_iteration_params(ctx, None, "batch_size")
    assert result == "batch_size"
    assert ctx.params["batch_size"] == [8, 16]
    assert ctx.params["num_concurrency"] == [1]


def test_validate_iteration_params_warns_on_dropped_concurrency():
    """--num-concurrency is reported as ignored instead of dropped silently."""
    ctx = _iteration_ctx(
        "text-to-embeddings", [4, 8], [8, 16], explicit=("num_concurrency",)
    )
    with patch("genai_bench.cli.validation.logger") as mock_logger:
        result = validate_iteration_params(ctx, None, "num_concurrency")

    assert result == "batch_size"
    assert ctx.params["num_concurrency"] == [1]
    warning = " ".join(call.args[0] for call in mock_logger.warning.call_args_list)
    assert "--num-concurrency" in warning
    assert "--iteration-type num_concurrency" in warning


def test_validate_iteration_params_warns_on_dropped_batch_size():
    """--batch-size is reported as ignored when iterating over concurrency."""
    ctx = _iteration_ctx("text-to-text", [4, 8], [8, 16], explicit=("batch_size",))
    with patch("genai_bench.cli.validation.logger") as mock_logger:
        result = validate_iteration_params(ctx, None, "num_concurrency")

    assert result == "num_concurrency"
    assert ctx.params["batch_size"] == [1]
    warning = " ".join(call.args[0] for call in mock_logger.warning.call_args_list)
    assert "--batch-size" in warning


def test_validate_iteration_params_rejects_batch_size_for_generative_task():
    """batch_size iteration is only implemented for pooling tasks."""
    ctx = _iteration_ctx("text-to-text", [1, 2], [8, 16], explicit=("iteration_type",))
    with patch("genai_bench.cli.validation.logger") as mock_logger:
        result = validate_iteration_params(ctx, None, "batch_size")

    assert result == "num_concurrency"
    assert ctx.params["num_concurrency"] == [1, 2]
    assert ctx.params["batch_size"] == [1]
    warning = " ".join(call.args[0] for call in mock_logger.warning.call_args_list)
    assert "not supported for text-to-text" in warning


def test_validate_iteration_params_with_defaults():
    """Test validate_iteration_params with default values."""
    param = None

    # Test with default values for text-to-text
    ctx = _iteration_ctx("text-to-text", [], [])
    with patch("genai_bench.cli.validation.logger") as mock_logger:
        result = validate_iteration_params(ctx, param, "batch_size")
        mock_logger.warning.assert_called_once()

    assert result == "num_concurrency"
    assert ctx.params["batch_size"] == [1]
    assert ctx.params["num_concurrency"] == DEFAULT_NUM_CONCURRENCIES

    # Test with default values for text-to-embeddings
    ctx = _iteration_ctx("text-to-embeddings", [], [])
    with patch("genai_bench.cli.validation.logger") as mock_logger:
        result = validate_iteration_params(ctx, param, "num_concurrency")
        mock_logger.info.assert_called_once()

    assert result == "batch_size"
    assert ctx.params["batch_size"] == DEFAULT_BATCH_SIZES
    assert ctx.params["num_concurrency"] == [1]


def test_validate_iteration_params_without_parameter_source():
    """A bare context without param sources falls back to legacy behavior."""
    ctx = click.Context(click.Command("test"))
    ctx.params = {
        "task": "text-to-embeddings",
        "num_concurrency": [1, 2],
        "batch_size": [8],
    }
    with patch.object(ctx, "get_parameter_source", return_value=None):
        result = validate_iteration_params(ctx, None, "num_concurrency")
    assert result == "batch_size"
    assert ctx.params["batch_size"] == [8]
    assert ctx.params["num_concurrency"] == [1]


def _iteration_probe_command():
    """
    Build a command from the real benchmark options involved in axis selection.

    Copies the Parameter objects so the shared benchmark command is untouched,
    and drops prompting so CliRunner can drive it non-interactively. Using the
    real options means this exercises Click's own parameter-source tracking,
    which is what distinguishes an explicit --iteration-type from the default.
    """
    from genai_bench.cli.cli import benchmark

    needed = {"api_backend", "task", "num_concurrency", "batch_size", "iteration_type"}
    params = []
    for param in benchmark.params:
        if param.name not in needed:
            continue
        param = copy.copy(param)
        param.prompt = None
        param.required = False
        params.append(param)

    @click.command()
    def probe(**kwargs):
        ctx = click.get_current_context()
        click.echo(
            f"{ctx.params['iteration_type']}|"
            f"{list(ctx.params['num_concurrency'])}|"
            f"{list(ctx.params['batch_size'])}"
        )

    probe.params = params
    return probe


@pytest.mark.parametrize(
    "args,expected",
    [
        # Pooling tasks keep the historical batch_size axis by default
        (
            [],
            f"batch_size|[1]|{DEFAULT_BATCH_SIZES}",
        ),
        # An explicit --iteration-type switches the axis and picks up the
        # concurrency defaults
        (
            ["--iteration-type", "num_concurrency"],
            f"num_concurrency|{DEFAULT_NUM_CONCURRENCIES}|[1]",
        ),
        # ... and honors explicitly requested concurrency levels
        (
            [
                "--iteration-type",
                "num_concurrency",
                "--num-concurrency",
                "4",
                "--num-concurrency",
                "16",
            ],
            "num_concurrency|[4, 16]|[1]",
        ),
        # An explicit batch_size axis still works
        (
            ["--iteration-type", "batch_size", "--batch-size", "8"],
            "batch_size|[1]|[8]",
        ),
    ],
)
def test_validate_iteration_params_end_to_end(args, expected):
    """Axis selection resolves correctly through a real Click invocation."""
    runner = CliRunner()
    result = runner.invoke(
        _iteration_probe_command(),
        ["--api-backend", "openai", "--task", "text-to-embeddings", *args],
    )
    assert result.exit_code == 0, result.output
    assert expected in result.output


def test_validate_iteration_params_end_to_end_warns_on_dropped_concurrency(caplog):
    """--num-concurrency without an explicit axis switch warns and is dropped."""
    runner = CliRunner()
    with caplog.at_level(logging.WARNING, logger="genai_bench.cli.validation"):
        result = runner.invoke(
            _iteration_probe_command(),
            [
                "--api-backend",
                "openai",
                "--task",
                "text-to-embeddings",
                "--num-concurrency",
                "4",
            ],
        )
    assert result.exit_code == 0, result.output
    assert "Ignoring --num-concurrency" in caplog.text
    assert f"batch_size|[1]|{DEFAULT_BATCH_SIZES}" in result.output


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
def test_validate_api_key_cohere_v2_warning(mock_confirm):
    """Test API key validation for Cohere V2 backend with warning."""
    ctx = MagicMock()
    param = MagicMock()
    ctx.params = {"api_backend": "oci-cohere-v2"}

    result = validate_api_key(ctx, param, "test-key")
    assert result is None


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


def test_validate_prefix_len_with_dataset_scenario():
    """Test that prefix_len and prefix_ratio require all scenarios to be
    deterministic."""
    from genai_bench.cli.validation import validate_prefix_options

    # Test mutual exclusivity
    with pytest.raises(click.UsageError) as exc:
        validate_prefix_options(
            prefix_len=50,
            prefix_ratio=0.5,
            task="text-to-text",
            dataset_path="path/to/dataset",
            dataset_config=None,
            traffic_scenario=["D(100,50)"],
        )
    assert "mutually exclusive" in str(exc.value)

    # Test with dataset in traffic_scenario list
    with pytest.raises(click.UsageError) as exc:
        validate_prefix_options(
            prefix_len=50,
            prefix_ratio=None,
            task="text-to-text",
            dataset_path="path/to/dataset",
            dataset_config=None,
            traffic_scenario=["D(100,50)", "dataset"],
        )
    assert "not supported with dataset mode" in str(exc.value)

    # Test with only dataset scenario
    with pytest.raises(click.UsageError) as exc:
        validate_prefix_options(
            prefix_len=50,
            prefix_ratio=None,
            task="text-to-text",
            dataset_path="path/to/dataset",
            dataset_config=None,
            traffic_scenario=["dataset"],
        )
    assert "not supported with dataset mode" in str(exc.value)

    # Test --prefix-ratio with mixed deterministic and non-deterministic scenarios
    with pytest.raises(click.UsageError) as exc:
        validate_prefix_options(
            prefix_len=None,
            prefix_ratio=0.5,
            task="text-to-text",
            dataset_path="path/to/dataset",
            dataset_config=None,
            traffic_scenario=["D(100,50)", "N(480,240)/(300,150)"],
        )
    assert "all traffic scenarios to be deterministic" in str(exc.value)
    assert "N(480,240)/(300,150)" in str(exc.value)

    # Test --prefix-ratio with only non-deterministic scenarios
    with pytest.raises(click.UsageError) as exc:
        validate_prefix_options(
            prefix_len=None,
            prefix_ratio=0.5,
            task="text-to-text",
            dataset_path="path/to/dataset",
            dataset_config=None,
            traffic_scenario=["N(480,240)/(300,150)", "U(50,100)/(200,250)"],
        )
    assert "all traffic scenarios to be deterministic" in str(exc.value)

    # Test --prefix-len with non-deterministic scenarios (should pass)
    # prefix_len works with any scenario type as long as min_input_tokens >= prefix_len
    # N(500,100): min = max(1, 500 - 3*100) = max(1, 200) = 200 >= 50
    validate_prefix_options(
        prefix_len=50,
        prefix_ratio=None,
        task="text-to-text",
        dataset_path="path/to/dataset",
        dataset_config=None,
        traffic_scenario=["N(500,100)/(300,50)"],
    )

    # Test --prefix-len with uniform distribution (should pass)
    validate_prefix_options(
        prefix_len=50,
        prefix_ratio=None,
        task="text-to-text",
        dataset_path="path/to/dataset",
        dataset_config=None,
        traffic_scenario=["U(100,200)/(50,100)"],  # min_input = 100 >= 50
    )

    # Test with scenario where min_input_tokens < prefix_len (deterministic)
    with pytest.raises(click.UsageError) as exc:
        validate_prefix_options(
            prefix_len=150,
            prefix_ratio=None,
            task="text-to-text",
            dataset_path="path/to/dataset",
            dataset_config=None,
            traffic_scenario=["D(100,50)", "D(200,100)"],
        )
    assert "must be <= minimum input tokens" in str(exc.value)
    assert "D(100,50)" in str(exc.value)

    # Test with scenario where min_input_tokens < prefix_len (uniform)
    with pytest.raises(click.UsageError) as exc:
        validate_prefix_options(
            prefix_len=150,
            prefix_ratio=None,
            task="text-to-text",
            dataset_path="path/to/dataset",
            dataset_config=None,
            traffic_scenario=["U(100,200)/(50,100)"],  # min_input = 100 < 150
        )
    assert "must be <= minimum input tokens" in str(exc.value)
    assert "min_input_tokens=100" in str(exc.value)

    # Test with valid deterministic scenarios only (should pass)
    validate_prefix_options(
        prefix_len=50,
        prefix_ratio=None,
        task="text-to-text",
        dataset_path="path/to/dataset",
        dataset_config=None,
        traffic_scenario=["D(100,50)", "D(200,100)"],
    )
