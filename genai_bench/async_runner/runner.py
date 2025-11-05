"""
Compatibility shim for OpenLoopRunner.

This module provides backward compatibility for code that imports OpenLoopRunner
from genai_bench.openloop.runner. The actual implementation has been refactored
into separate modules:
- genai_bench.async_runner.base: BaseAsyncRunner (shared functionality)
- genai_bench.async_runner.open_loop: OpenLoopRunner (QPS-based execution)
- genai_bench.async_runner.closed_loop: ClosedLoopRunner (concurrency-based execution)
- genai_bench.async_runner.factory: create_runner() (factory function)

New code should use the factory function or import directly from the new modules.
"""

# Import from the new refactored modules for backward compatibility
from genai_bench.async_runner.open_loop import OpenLoopRunner

# Export for backward compatibility
__all__ = ["OpenLoopRunner"]
