"""Async runner execution engine for load testing (supports both open-loop and closed-loop modes)."""

from genai_bench.async_runner.base import BaseAsyncRunner
from genai_bench.async_runner.closed_loop import ClosedLoopRunner
from genai_bench.async_runner.factory import create_runner
from genai_bench.async_runner.open_loop import OpenLoopRunner

# For backward compatibility, export OpenLoopRunner as the default
# but recommend using create_runner() factory function
__all__ = [
    "BaseAsyncRunner",
    "OpenLoopRunner",
    "ClosedLoopRunner",
    "create_runner",
]
