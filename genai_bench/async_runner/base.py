"""Base class for async runners with shared functionality."""

import asyncio
import time
from dataclasses import dataclass
from typing import Optional

import aiohttp

try:
    import orjson as json_lib  # type: ignore[no-redef]
except ImportError:
    import json as json_lib  # type: ignore[no-redef]

from genai_bench.logging import init_logger
from genai_bench.metrics.request_metrics_collector import RequestMetricsCollector
from genai_bench.protocol import (
    UserChatRequest,
    UserChatResponse,
    UserEmbeddingRequest,
    UserImageChatRequest,
    UserResponse,
)
from genai_bench.scenarios.base import Scenario

logger = init_logger(__name__)


@dataclass
class NetworkTimingContext:
    """Context to track network timing for a single request."""

    dns_start: Optional[float] = None
    dns_end: Optional[float] = None
    connect_start: Optional[float] = None
    connect_end: Optional[float] = None
    tls_start: Optional[float] = None
    tls_end: Optional[float] = None

    @property
    def dns_time(self) -> Optional[float]:
        """DNS resolution time in seconds."""
        if self.dns_start is not None and self.dns_end is not None:
            return self.dns_end - self.dns_start
        return None

    @property
    def connect_time(self) -> Optional[float]:
        """Total connection time (TCP + TLS) in seconds."""
        if self.connect_start is not None and self.connect_end is not None:
            return self.connect_end - self.connect_start
        return None

    @property
    def tls_time(self) -> Optional[float]:
        """TLS handshake time in seconds."""
        if self.tls_start is not None and self.tls_end is not None:
            return self.tls_end - self.tls_start
        return None


def create_trace_config() -> aiohttp.TraceConfig:
    """
    Create an aiohttp TraceConfig to capture network timing metrics.

    The trace config hooks into various connection events to measure:
    - DNS resolution time
    - TCP + TLS connection time
    - TLS handshake time (if separate from TCP)

    Note: With connection pooling, DNS and connection times may be 0
    for requests that reuse an existing connection.
    """

    async def on_dns_resolvehost_start(session, ctx, params):
        ctx.trace_request_ctx.dns_start = time.monotonic()

    async def on_dns_resolvehost_end(session, ctx, params):
        ctx.trace_request_ctx.dns_end = time.monotonic()

    async def on_connection_create_start(session, ctx, params):
        ctx.trace_request_ctx.connect_start = time.monotonic()

    async def on_connection_create_end(session, ctx, params):
        ctx.trace_request_ctx.connect_end = time.monotonic()

    # Note: aiohttp doesn't have separate TLS events, but we can approximate
    # by using the connection reuseconn events when available

    trace_config = aiohttp.TraceConfig()
    trace_config.on_dns_resolvehost_start.append(on_dns_resolvehost_start)
    trace_config.on_dns_resolvehost_end.append(on_dns_resolvehost_end)
    trace_config.on_connection_create_start.append(on_connection_create_start)
    trace_config.on_connection_create_end.append(on_connection_create_end)

    return trace_config


class BaseAsyncRunner:
    """
    Base class for async runners with shared functionality:
    - Request preparation and sending
    - Authentication and headers
    - Latency probing
    - Common utilities
    """

    def __init__(
        self,
        *,
        sampler,
        api_backend: str,
        api_base: str,
        api_model_name: str,
        auth_provider,
        aggregated_metrics_collector,
        dashboard=None,
        track_network_timing: bool = False,
    ) -> None:
        self.sampler = sampler
        self.api_backend = api_backend
        self.api_base = api_base
        self.api_model_name = api_model_name
        self.auth_provider = auth_provider
        self.aggregated = aggregated_metrics_collector
        self.dashboard = dashboard
        self._track_network_timing = track_network_timing

        self.headers = None
        if auth_provider and hasattr(auth_provider, "get_headers"):
            headers = auth_provider.get_headers()
            if headers:
                self.headers = headers.copy()
                self.headers["Content-Type"] = "application/json"
        elif auth_provider and hasattr(auth_provider, "get_credentials"):
            token = auth_provider.get_credentials()
            if token:
                self.headers = {
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                }

        # AIOHTTP settings aligned with tore-speed
        self._aio_timeout = aiohttp.ClientTimeout(total=6 * 60 * 60)
        self._aio_read_bufsize = 256 * 1024

        # Trace config for network timing metrics (only if tracking enabled)
        self._trace_config = create_trace_config() if track_network_timing else None

        # Reuse session per runner instance to avoid creating new session per request
        self._session: Optional[aiohttp.ClientSession] = None

    def _prepare_request(self, scenario_input):
        """Prepare a request from a scenario string or Scenario object."""
        # Accept either a prebuilt Scenario or a scenario string, for parity with Locust path
        if isinstance(scenario_input, str):
            scenario_obj = Scenario.from_string(scenario_input)
        else:
            scenario_obj = scenario_input
        req = self.sampler.sample(scenario_obj)

        # Validate request is properly formed
        if req is None:
            raise ValueError(
                "Sampler returned None request. Check your sampler configuration."
            )

        # Validate request type matches API backend expectations
        if isinstance(req, (UserChatRequest, UserImageChatRequest)):
            if not hasattr(req, "model") or not req.model:
                raise ValueError("Chat request missing required 'model' field")
            if not hasattr(req, "prompt") and not isinstance(req, UserImageChatRequest):
                raise ValueError("Chat request missing required 'prompt' field")
        elif isinstance(req, UserEmbeddingRequest):
            if not hasattr(req, "model") or not req.model:
                raise ValueError("Embedding request missing required 'model' field")
            if not hasattr(req, "documents") or not req.documents:
                raise ValueError("Embedding request missing required 'documents' field")
        else:
            raise ValueError(
                f"Unsupported request type: {type(req)}. "
                f"Expected UserChatRequest, UserImageChatRequest, or UserEmbeddingRequest."
            )

        return req

    async def _probe_latency(
        self, scenario: str, num_probe_requests: int = 10
    ) -> float:
        """
        Run a latency probe to measure average request latency.

        Args:
            scenario: Scenario string to use for probe requests
            num_probe_requests: Number of probe requests to send (default: 10)

        Returns:
            Average latency in seconds
        """
        logger.info(
            f"🔍 Running latency probe with {num_probe_requests} requests to determine QPS..."
        )

        latencies = []
        for i in range(num_probe_requests):
            req = self._prepare_request(scenario)
            response = await self._send_request(req)
            if (
                response.status_code == 200
                and response.start_time is not None
                and response.end_time is not None
            ):
                latency = response.end_time - response.start_time
                latencies.append(latency)
                logger.info(
                    f"  Probe request {i + 1}/{num_probe_requests}: "
                    f"latency = {latency:.3f}s"
                )
            else:
                error_msg = response.error_message or "Unknown error"
                logger.warning(
                    f"  Probe request {i + 1}/{num_probe_requests} failed: "
                    f"status={response.status_code}, error={error_msg[:200]}"
                )

        if not latencies:
            raise ValueError(
                "Latency probe failed: no successful requests. "
                "Check your API endpoint and authentication."
            )

        avg_latency = sum(latencies) / len(latencies)
        logger.info(
            f"✅ Latency probe complete: average latency = {avg_latency:.3f}s "
            f"(from {len(latencies)} successful requests)"
        )
        return avg_latency

    async def _send_request(self, req) -> UserResponse:
        """Send a request and return the response. Handles streaming for chat completions."""
        # Currently implement OpenAI-compatible endpoints for text chat and embeddings
        try:
            if isinstance(req, (UserChatRequest, UserImageChatRequest)):
                endpoint = "/v1/chat/completions"
                if isinstance(req, UserImageChatRequest):
                    text_content = [{"type": "text", "text": req.prompt}]  # type: ignore[attr-defined]
                    image_content = [
                        {"type": "image_url", "image_url": {"url": image}}  # type: ignore[attr-defined]
                        for image in req.image_content  # type: ignore[attr-defined]
                    ]
                    content = text_content + image_content  # type: ignore[assignment]
                else:
                    content = req.prompt  # type: ignore[assignment]

                # Build payload - prioritize max_tokens from additional_request_params if present
                # min_tokens and max_tokens are now automatically set by the sampler from the scenario
                # This matches BasetenUser's _prepare_chat_request logic
                max_tokens = req.additional_request_params.get(
                    "max_tokens", None
                ) or getattr(req, "max_tokens", None)

                payload = {
                    "model": req.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": content,
                        }
                    ],
                    "max_tokens": max_tokens,
                    "temperature": req.additional_request_params.get(
                        "temperature", 0.0
                    ),
                    "ignore_eos": req.additional_request_params.get(
                        "ignore_eos", bool(max_tokens)
                    ),
                    # Force streaming to compute TTFT/TPOT properly
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    **{
                        k: v
                        for k, v in req.additional_request_params.items()
                        if k not in {"stream"}
                    },
                }

                # For Baseten, api_base already includes the full endpoint path
                # For other backends, append the endpoint to the base URL
                if self.api_backend.lower() == "baseten":
                    request_url = self.api_base
                else:
                    request_url = f"{self.api_base}{endpoint}"

                # Log first request for debugging
                if not hasattr(self, "_logged_request_info"):
                    logger.info(f"🌐 Async runner request URL: {request_url}")
                    logger.info(f"🔑 Headers present: {bool(self.headers)}")
                    if self.headers:
                        auth_header = self.headers.get("Authorization", "None")
                        # Mask the API key for security
                        if "Api-Key" in auth_header or "Bearer" in auth_header:
                            parts = auth_header.split(" ", 1)
                            if len(parts) == 2:
                                auth_header = f"{parts[0]} [REDACTED]"
                        logger.info(f"🔐 Auth header: {auth_header}")
                    self._logged_request_info = True

                start_time = time.monotonic()
                # Reuse session if available, otherwise create new one
                if self._session is None or self._session.closed:
                    self._session = aiohttp.ClientSession(
                        headers=self.headers,
                        timeout=self._aio_timeout,
                        read_bufsize=self._aio_read_bufsize,
                        trace_configs=[self._trace_config]
                        if self._trace_config
                        else None,
                    )

                # Create a new timing context for this request (only if tracking enabled)
                timing_ctx = (
                    NetworkTimingContext() if self._track_network_timing else None
                )

                async with self._session.post(
                    url=request_url,
                    json=payload,
                    trace_request_ctx=timing_ctx,
                ) as resp:
                    if resp.status != 200:
                        # Stream entire error body for parity with tore-speed
                        error_message_bytes = b""
                        async for chunk_bytes in resp.content:
                            error_message_bytes += chunk_bytes
                        text = error_message_bytes.decode("utf-8")
                        logger.error(
                            f"❌ Request failed with status {resp.status}: {text[:500]}"
                        )
                        return UserResponse(status_code=resp.status, error_message=text)

                    stream_chunk_prefix = "data: "
                    end_chunk = b"[DONE]"

                    generated_text = ""
                    tokens_received = 0
                    time_at_first_token: Optional[float] = None
                    finish_reason: Optional[str] = None
                    num_prompt_tokens = None

                    # Read streaming response line by line
                    buffer = b""
                    async for chunk_bytes in resp.content.iter_any():
                        buffer += chunk_bytes
                        # Process complete lines
                        while b"\n" in buffer:
                            line, buffer = buffer.split(b"\n", 1)
                            chunk = line.strip()
                            if not chunk:
                                continue
                            # Gate on SSE style lines like tore-speed does
                            if not chunk.startswith(stream_chunk_prefix.encode()):
                                continue
                            chunk = chunk[len(stream_chunk_prefix) :]
                            if chunk.strip() == end_chunk:
                                break
                            try:
                                if json_lib.__name__ == "orjson":
                                    data = json_lib.loads(chunk)
                                else:
                                    data = json_lib.loads(chunk.decode("utf-8"))
                            except Exception:
                                continue

                            if data.get("error") is not None:
                                error_msg = data["error"].get(
                                    "message", "Unknown error"
                                )
                                error_code = data["error"].get("code", -1)
                                logger.error(
                                    f"❌ Error in streaming response: code={error_code}, message={error_msg}"
                                )
                                return UserResponse(
                                    status_code=error_code,
                                    error_message=error_msg,
                                )

                            if (
                                (not data.get("choices"))
                                and finish_reason
                                and data.get("usage")
                            ):
                                usage = data["usage"]
                                num_prompt_tokens = usage.get("prompt_tokens")
                                tokens_received = usage.get("completion_tokens", 0)
                                # Don't set time_at_first_token here - it should be set when content arrives
                                # If no content was received, this is likely a non-streaming response
                                if not time_at_first_token:
                                    # This shouldn't happen in streaming, but fallback for edge cases
                                    logger.warning(
                                        "⚠️ No content received before finish_reason. "
                                        "Setting time_at_first_token to end_time."
                                    )
                                    time_at_first_token = time.monotonic()
                                break

                            try:
                                delta = data["choices"][0]["delta"]
                                content_piece = delta.get("content") or delta.get(
                                    "reasoning_content"
                                )
                                usage = delta.get("usage")

                                if usage:
                                    tokens_received = usage.get(
                                        "completion_tokens", tokens_received
                                    )

                                # Set TTFT on first chunk with choices (matching vLLM's approach)
                                # This measures when the first token chunk arrives, even if content is empty
                                # which is more accurate than waiting for non-empty content
                                if not time_at_first_token:
                                    time_at_first_token = time.monotonic()

                                if content_piece:
                                    generated_text += content_piece

                                # Capture finish_reason when it appears (may appear before usage chunk)
                                if "finish_reason" in data["choices"][0]:
                                    finish_reason = data["choices"][0].get(
                                        "finish_reason", None
                                    )

                                if finish_reason and data.get("usage"):
                                    usage = data["usage"]
                                    num_prompt_tokens = usage.get("prompt_tokens")
                                    tokens_received = usage.get(
                                        "completion_tokens", tokens_received
                                    )
                                    break
                            except (IndexError, KeyError):
                                continue

                    end_time = time.monotonic()

                if not tokens_received:
                    tokens_received = self.sampler.get_token_length(
                        generated_text, add_special_tokens=False
                    )
                    logger.warning(
                        "🚨🚨🚨 There is no usage info returned from the model "
                        "server. Estimated tokens_received based on the model "
                        "tokenizer."
                    )

                # Check if min_tokens was set and if we got fewer tokens than expected
                min_tokens_expected = req.additional_request_params.get("min_tokens")
                if min_tokens_expected and tokens_received < min_tokens_expected:
                    logger.warning(
                        f"⚠️ min_tokens not respected! "
                        f"Requested min_tokens: {min_tokens_expected}, "
                        f"received: {tokens_received}, "
                        f"finish_reason: {finish_reason}. "
                        f"The server may not support min_tokens or it may have stopped early."
                    )

                # Fallback: if server didn't return prompt_tokens in usage, derive from request
                if num_prompt_tokens is None:
                    num_prompt_tokens = getattr(req, "num_prefill_tokens", None)
                    if num_prompt_tokens is None:
                        num_prompt_tokens = self.sampler.get_token_length(
                            req.prompt, add_special_tokens=False
                        )

                # Ensure time_at_first_token is always set (required for metrics calculation)
                # This handles edge cases where no chunks were received or streaming failed
                if time_at_first_token is None:
                    logger.warning(
                        "⚠️ time_at_first_token was not set during streaming. "
                        "Using end_time as fallback. This may indicate an issue with the streaming response."
                    )
                    time_at_first_token = end_time

                return UserChatResponse(
                    status_code=200,
                    generated_text=generated_text,
                    tokens_received=tokens_received,
                    time_at_first_token=time_at_first_token,
                    num_prefill_tokens=num_prompt_tokens,
                    start_time=start_time,
                    end_time=end_time,
                    # Network timing metrics from trace context (if tracking enabled)
                    network_connect_time=timing_ctx.connect_time
                    if timing_ctx
                    else None,
                    network_dns_time=timing_ctx.dns_time if timing_ctx else None,
                    network_tls_time=timing_ctx.tls_time if timing_ctx else None,
                )

            elif isinstance(req, UserEmbeddingRequest):
                endpoint = "/v1/embeddings"
                payload = {
                    "model": req.model,
                    "input": req.documents,
                    **req.additional_request_params,
                }
                # For Baseten, api_base already includes the full endpoint path
                # For other backends, append the endpoint to the base URL
                if self.api_backend.lower() == "baseten":
                    request_url = self.api_base
                else:
                    request_url = f"{self.api_base}{endpoint}"

                start_time = time.monotonic()
                # Reuse session if available, otherwise create new one
                if self._session is None or self._session.closed:
                    self._session = aiohttp.ClientSession(
                        headers=self.headers,
                        timeout=self._aio_timeout,
                        read_bufsize=self._aio_read_bufsize,
                        trace_configs=[self._trace_config]
                        if self._trace_config
                        else None,
                    )

                # Create a new timing context for this request (only if tracking enabled)
                timing_ctx = (
                    NetworkTimingContext() if self._track_network_timing else None
                )

                async with self._session.post(
                    url=request_url,
                    json=payload,
                    trace_request_ctx=timing_ctx,
                ) as resp:
                    end_time = time.monotonic()
                    if resp.status == 200:
                        data = await resp.json()
                        num_prompt_tokens = data.get("usage", {}).get("prompt_tokens")
                        return UserResponse(
                            status_code=200,
                            start_time=start_time,
                            end_time=end_time,
                            time_at_first_token=end_time,
                            num_prefill_tokens=num_prompt_tokens,
                            # Network timing metrics from trace context (if tracking enabled)
                            network_connect_time=timing_ctx.connect_time
                            if timing_ctx
                            else None,
                            network_dns_time=timing_ctx.dns_time
                            if timing_ctx
                            else None,
                            network_tls_time=timing_ctx.tls_time
                            if timing_ctx
                            else None,
                        )
                    else:
                        # Stream entire error body for parity with tore-speed
                        error_message_bytes = b""
                        async for chunk_bytes in resp.content:
                            error_message_bytes += chunk_bytes
                        text = error_message_bytes.decode("utf-8")
                        return UserResponse(status_code=resp.status, error_message=text)

            else:
                return UserResponse(
                    status_code=400, error_message="Unsupported request type"
                )
        except aiohttp.ClientConnectionError as e:
            logger.error(f"❌ Connection error: {e}")
            return UserResponse(status_code=503, error_message=f"Connection error: {e}")
        except asyncio.TimeoutError as e:
            logger.error(f"❌ Request timed out: {e}")
            return UserResponse(
                status_code=408, error_message=f"Request timed out: {e}"
            )
        except Exception as e:
            logger.exception(
                f"❌ Unexpected error in _send_request: {type(e).__name__}: {e}"
            )
            return UserResponse(status_code=500, error_message=str(e))

    async def cleanup(self) -> None:
        """Clean up resources, including closing the aiohttp session."""
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None

    def _handle_error(self, error: Exception, context: str = "") -> None:
        """
        Standardized error handling helper.

        Args:
            error: The exception that occurred
            context: Additional context about where the error occurred
        """
        error_type = type(error).__name__
        error_msg = str(error)
        context_str = f" in {context}" if context else ""

        if isinstance(error, asyncio.CancelledError):
            logger.debug(f"Task cancelled{context_str}: {error_msg}")
        elif isinstance(error, asyncio.TimeoutError):
            logger.warning(f"Timeout error{context_str}: {error_msg}")
        elif isinstance(error, aiohttp.ClientConnectionError):
            logger.error(f"Connection error{context_str}: {error_msg}")
        else:
            logger.error(f"Unexpected error{context_str} ({error_type}): {error_msg}")

    async def _send_one(self, req) -> None:
        """Send a single request and record metrics."""
        collector = RequestMetricsCollector()
        try:
            response = await self._send_request(req)
            # Convert to RequestLevelMetrics and add to collector
            if response.status_code == 200:
                collector.calculate_metrics(response)
            else:
                collector.metrics.error_code = response.status_code
                collector.metrics.error_message = response.error_message
        except asyncio.CancelledError:
            # Task was cancelled - record as error to match Locust behavior
            # Locust always collects metrics, even for cancelled/failed requests
            collector.metrics.error_code = 0  # Use 0 to indicate cancellation
            collector.metrics.error_message = "Request was cancelled"
            raise  # Re-raise to allow proper cleanup
        except Exception as e:
            # Handle other exceptions (timeouts, connection errors, etc.)
            # Match Locust behavior: always collect metrics, even for failures
            collector.metrics.error_code = 0  # Use 0 for non-HTTP errors
            collector.metrics.error_message = f"{type(e).__name__}: {str(e)}"
            self._handle_error(e, "request execution")
        finally:
            # Always record metrics (matches Locust behavior)
            # This ensures we have metrics even if request was cancelled or failed
            self.aggregated.add_single_request_metrics(collector.metrics)
            # Update dashboard live if available
            if self.dashboard is not None:
                live = self.aggregated.get_live_metrics()
                total_requests = (
                    self.aggregated.aggregated_metrics.num_completed_requests
                    + self.aggregated.aggregated_metrics.num_error_requests
                )
                self.dashboard.handle_single_request(
                    live, total_requests, collector.metrics.error_code
                )
