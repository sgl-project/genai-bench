"""Iteration options accept a list without repeating the option name."""

import click
import pytest

from genai_bench.cli.cli import benchmark

BASE = [
    "--api-backend",
    "vllm",
    "--api-base",
    "http://localhost:8080",
    "--api-key",
    "test",
    "--api-model-name",
    "test",
    "--model-tokenizer",
    "test",
    "--max-time-per-run",
    "1",
    "--max-requests-per-run",
    "5",
]


@pytest.mark.parametrize(
    "option, task, field",
    [
        ("--num-concurrency", "text-to-text", "num_concurrency"),
        ("--batch-size", "text-to-embeddings", "batch_size"),
        ("--batch-size", "text-to-rerank", "batch_size"),
    ],
)
@pytest.mark.parametrize(
    "values",
    [
        ["1", "2", "4"],
        ["1", "--repeat", "2", "4"],
    ],
)
def test_benchmark_accepts_spaced_iteration_values(option, task, field, values):
    args = [option if value == "--repeat" else value for value in values]
    with benchmark.make_context(
        "benchmark", BASE + ["--task", task, option] + args
    ) as ctx:
        assert list(ctx.params[field]) == [1, 2, 4]


@pytest.mark.parametrize(
    "option, task, field",
    [
        ("--num-concurrency", "text-to-text", "num_concurrency"),
        ("--batch-size", "text-to-embeddings", "batch_size"),
    ],
)
def test_benchmark_accepts_inline_first_value(option, task, field):
    with benchmark.make_context(
        "benchmark", BASE + ["--task", task, option + "=1", "2", "4"]
    ) as ctx:
        assert list(ctx.params[field]) == [1, 2, 4]


def test_other_option_value_is_not_expanded():
    args = BASE + [
        "--task",
        "text-to-text",
        "--api-model-name",
        "--num-concurrency",
        "--num-concurrency",
        "1",
        "2",
        "--batch-size",
        "8",
        "16",
    ]
    with benchmark.make_context("benchmark", args) as ctx:
        assert ctx.params["api_model_name"] == "--num-concurrency"
        assert list(ctx.params["num_concurrency"]) == [1, 2]


@pytest.mark.parametrize(
    "suffix",
    [
        ["--num-concurrency", "1", "nope"],
        ["--batch-size", "1", "nope"],
    ],
)
def test_invalid_integer_keeps_click_validation(suffix):
    with pytest.raises(click.BadParameter, match="integer"):
        benchmark.make_context("benchmark", BASE + ["--task", "text-to-text"] + suffix)


def test_missing_iteration_value_keeps_click_validation():
    with pytest.raises(click.BadOptionUsage, match="requires an argument"):
        benchmark.make_context(
            "benchmark", BASE + ["--task", "text-to-text", "--num-concurrency"]
        )


def test_repeated_flags_remain_compatible():
    args = BASE + [
        "--task",
        "text-to-text",
        "--num-concurrency",
        "1",
        "--num-concurrency",
        "2",
    ]
    with benchmark.make_context("benchmark", args) as ctx:
        assert list(ctx.params["num_concurrency"]) == [1, 2]


def test_signed_integer_is_an_iteration_value():
    args = BASE + ["--task", "text-to-text", "--num-concurrency", "1", "-2"]
    with benchmark.make_context("benchmark", args) as ctx:
        assert list(ctx.params["num_concurrency"]) == [1, -2]


def test_invalid_signed_integer_keeps_click_validation():
    with pytest.raises(click.BadParameter, match="integer"):
        benchmark.make_context(
            "benchmark",
            BASE + ["--task", "text-to-text", "--num-concurrency", "1", "-2.5"],
        )


def test_end_of_options_is_preserved():
    with pytest.raises(click.UsageError, match="unexpected extra argument"):
        benchmark.make_context(
            "benchmark",
            BASE + ["--task", "text-to-text", "--num-concurrency", "1", "--", "2"],
        )


def test_unknown_option_is_not_absorbed_as_a_value():
    with pytest.raises(click.NoSuchOption, match="--unknown-option"):
        benchmark.make_context(
            "benchmark",
            BASE
            + ["--task", "text-to-text", "--num-concurrency", "1", "--unknown-option"],
        )


def test_inline_other_option_value_is_preserved():
    args = BASE + [
        "--task",
        "text-to-text",
        "--api-model-name=--batch-size",
        "--num-concurrency",
        "1",
        "2",
    ]
    with benchmark.make_context("benchmark", args) as ctx:
        assert ctx.params["api_model_name"] == "--batch-size"
        assert list(ctx.params["num_concurrency"]) == [1, 2]
