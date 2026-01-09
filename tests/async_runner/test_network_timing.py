"""Tests for network timing functionality in the async runner."""

import pytest
import time

from genai_bench.async_runner.base import NetworkTimingContext, create_trace_config


class TestNetworkTimingContext:
    """Tests for NetworkTimingContext dataclass."""

    def test_dns_time_calculation(self):
        """Test DNS time calculation with valid start and end times."""
        ctx = NetworkTimingContext()
        ctx.dns_start = 1000.0
        ctx.dns_end = 1000.05  # 50ms

        assert ctx.dns_time == pytest.approx(0.05)

    def test_dns_time_none_when_incomplete(self):
        """Test DNS time returns None when times are incomplete."""
        ctx = NetworkTimingContext()
        ctx.dns_start = 1000.0
        # dns_end not set

        assert ctx.dns_time is None

    def test_connect_time_calculation(self):
        """Test connection time calculation with valid start and end times."""
        ctx = NetworkTimingContext()
        ctx.connect_start = 1000.0
        ctx.connect_end = 1000.044  # 44ms (TCP + TLS)

        assert ctx.connect_time == pytest.approx(0.044)

    def test_connect_time_none_when_incomplete(self):
        """Test connection time returns None when times are incomplete."""
        ctx = NetworkTimingContext()
        # Neither connect_start nor connect_end set

        assert ctx.connect_time is None

    def test_tls_time_calculation(self):
        """Test TLS time calculation with valid start and end times."""
        ctx = NetworkTimingContext()
        ctx.tls_start = 1000.015  # After TCP connect
        ctx.tls_end = 1000.044  # 29ms for TLS

        assert ctx.tls_time == pytest.approx(0.029)

    def test_tls_time_none_when_incomplete(self):
        """Test TLS time returns None when times are incomplete."""
        ctx = NetworkTimingContext()
        ctx.tls_start = 1000.0
        # tls_end not set

        assert ctx.tls_time is None

    def test_all_times_together(self):
        """Test all timing properties work together."""
        ctx = NetworkTimingContext()
        base = time.monotonic()

        ctx.dns_start = base
        ctx.dns_end = base + 0.015  # 15ms DNS
        ctx.connect_start = base + 0.015
        ctx.connect_end = base + 0.044  # 29ms connection (TCP + TLS)

        assert ctx.dns_time == pytest.approx(0.015)
        assert ctx.connect_time == pytest.approx(0.029)
        assert ctx.tls_time is None  # TLS times not set separately


class TestTraceConfig:
    """Tests for trace config creation."""

    def test_create_trace_config_returns_trace_config(self):
        """Test that create_trace_config returns a valid TraceConfig."""
        import aiohttp

        trace_config = create_trace_config()

        assert isinstance(trace_config, aiohttp.TraceConfig)

    def test_trace_config_has_event_handlers(self):
        """Test that trace config has the expected event handlers registered."""
        trace_config = create_trace_config()

        # Check that handlers are registered
        assert len(trace_config.on_dns_resolvehost_start) > 0
        assert len(trace_config.on_dns_resolvehost_end) > 0
        assert len(trace_config.on_connection_create_start) > 0
        assert len(trace_config.on_connection_create_end) > 0
