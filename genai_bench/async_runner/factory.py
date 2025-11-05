"""Factory function for creating appropriate async runner based on parameters."""

from typing import Optional

from genai_bench.async_runner.closed_loop import ClosedLoopRunner
from genai_bench.async_runner.open_loop import OpenLoopRunner
from genai_bench.logging import init_logger

logger = init_logger(__name__)


def create_runner(
    *,
    sampler,
    api_backend: str,
    api_base: str,
    api_model_name: str,
    auth_provider,
    aggregated_metrics_collector,
    dashboard=None,
    qps_level: Optional[float] = None,
    target_concurrency: Optional[int] = None,
):
    """
    Factory function to create the appropriate runner based on parameters.

    Args:
        sampler: Sampler instance for generating requests
        api_backend: API backend name (e.g., "openai", "baseten")
        api_base: Base URL for API
        api_model_name: Model name
        auth_provider: Authentication provider
        aggregated_metrics_collector: Metrics collector
        dashboard: Optional dashboard for live updates
        qps_level: If provided, creates OpenLoopRunner (open-loop mode)
        target_concurrency: If provided, creates ClosedLoopRunner (closed-loop mode)

    Returns:
        Either OpenLoopRunner or ClosedLoopRunner instance

    Raises:
        ValueError: If neither or both qps_level and target_concurrency are provided
    """
    # Validate that exactly one mode is specified
    if qps_level is None and target_concurrency is None:
        raise ValueError(
            "Must specify either qps_level (open-loop mode) or target_concurrency (closed-loop mode)"
        )

    if qps_level is not None and target_concurrency is not None:
        raise ValueError(
            "Cannot specify both qps_level and target_concurrency. "
            "Choose either open-loop (qps_level) or closed-loop (target_concurrency) mode."
        )

    # Validate QPS value if provided
    if qps_level is not None:
        if qps_level <= 0:
            raise ValueError(
                f"qps_level must be positive, got {qps_level}. "
                "QPS (queries per second) must be greater than 0."
            )
        if qps_level > 10000:
            raise ValueError(
                f"qps_level is too high: {qps_level}. "
                "Maximum allowed QPS is 10000. If you need higher throughput, "
                "consider using multiple runner instances or distributed execution."
            )

    # Create appropriate runner
    if qps_level is not None:
        logger.info("📊 Creating async runner for open-loop (QPS-based) execution")
        return OpenLoopRunner(
            sampler=sampler,
            api_backend=api_backend,
            api_base=api_base,
            api_model_name=api_model_name,
            auth_provider=auth_provider,
            aggregated_metrics_collector=aggregated_metrics_collector,
            dashboard=dashboard,
        )
    else:
        logger.info(
            "🔄 Creating ClosedLoopRunner for closed-loop (concurrency-based) execution"
        )
        return ClosedLoopRunner(
            sampler=sampler,
            api_backend=api_backend,
            api_base=api_base,
            api_model_name=api_model_name,
            auth_provider=auth_provider,
            aggregated_metrics_collector=aggregated_metrics_collector,
            dashboard=dashboard,
        )
