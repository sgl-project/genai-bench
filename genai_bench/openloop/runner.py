import asyncio
import json
import time
import random
from typing import List, Optional

import requests

from genai_bench.logging import init_logger
from genai_bench.metrics.request_metrics_collector import RequestMetricsCollector
from genai_bench.protocol import (
    UserChatRequest,
    UserEmbeddingRequest,
    UserImageChatRequest,
    UserResponse,
    UserChatResponse,
)
from genai_bench.scenarios.base import Scenario


logger = init_logger(__name__)


class OpenLoopRunner:
    """
    Open-loop QPS runner that schedules global inter-arrivals (tore-speed style)
    and emits RequestLevelMetrics via AggregatedMetricsCollector.
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
    ) -> None:
        self.sampler = sampler
        self.api_backend = api_backend
        self.api_base = api_base
        self.api_model_name = api_model_name
        self.auth_provider = auth_provider
        self.aggregated = aggregated_metrics_collector
        self.dashboard = dashboard

        self.headers = None
        if auth_provider and hasattr(auth_provider, "get_credentials"):
            token = auth_provider.get_credentials()
            if token:
                self.headers = {
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                }

    def _wait_intervals(
        self, qps_level: float, duration_s: int, random_seed: int, distribution: str
    ) -> List[float]:
        mean = 1.0 / qps_level
        random.seed(random_seed)
        out: List[float] = []
        for _ in range(int(qps_level * duration_s)):
            if distribution == "exponential":
                out.append(random.expovariate(1.0 / mean))
            elif distribution == "uniform":
                out.append(random.uniform(0, 2 * mean))
            elif distribution == "constant":
                out.append(mean)
            else:
                raise ValueError(f"Invalid distribution: {distribution}")
        return out

    def _prepare_request(self, scenario_input):
        # Accept either a prebuilt Scenario or a scenario string, for parity with Locust path
        if isinstance(scenario_input, str):
            scenario_obj = Scenario.from_string(scenario_input)
        else:
            scenario_obj = scenario_input
        req = self.sampler.sample(scenario_obj)
        return req

    def _send_request(self, req) -> UserResponse:
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
                    content = text_content + image_content
                else:
                    content = req.prompt

                payload = {
                    "model": req.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": content,
                        }
                    ],
                    "max_tokens": req.additional_request_params.get("max_tokens", None)
                    or req.__dict__.get("max_tokens"),
                    "temperature": req.additional_request_params.get("temperature", 0.0),
                    "ignore_eos": req.additional_request_params.get(
                        "ignore_eos", bool(req.__dict__.get("max_tokens"))
                    ),
                    # Force streaming to compute TTFT/TPOT properly
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    **{k: v for k, v in req.additional_request_params.items() if k not in {"stream"}},
                }

                start_time = time.monotonic()
                r = requests.post(
                    url=f"{self.api_base}{endpoint}", json=payload, headers=self.headers, stream=True
                )

                if r.status_code != 200:
                    return UserResponse(status_code=r.status_code, error_message=r.text)

                stream_chunk_prefix = "data: "
                end_chunk = b"[DONE]"

                generated_text = ""
                tokens_received = 0
                time_at_first_token: Optional[float] = None
                finish_reason = None
                previous_data = None
                num_prompt_tokens = None

                for raw_chunk in r.iter_lines(chunk_size=None):
                    chunk = (raw_chunk or b"").strip()
                    if not chunk:
                        continue
                    # SSE prefix
                    if chunk.startswith(stream_chunk_prefix.encode()):
                        chunk = chunk[len(stream_chunk_prefix) :]
                    if chunk == end_chunk:
                        break

                    try:
                        data = json.loads(chunk)
                    except Exception:
                        previous_data = chunk
                        continue

                    if data.get("error") is not None:
                        return UserResponse(
                            status_code=data["error"].get("code", -1),
                            error_message=data["error"].get("message", "Unknown error"),
                        )

                    if (not data.get("choices")) and finish_reason and data.get("usage"):
                        usage = data["usage"]
                        num_prompt_tokens = usage.get("prompt_tokens")
                        tokens_received = usage.get("completion_tokens", 0)
                        if not time_at_first_token:
                            if tokens_received > 1:
                                time_at_first_token = time.monotonic()
                            else:
                                # Not enough info to infer TTFT
                                time_at_first_token = start_time
                        break

                    try:
                        delta = data["choices"][0]["delta"]
                        content_piece = delta.get("content") or delta.get("reasoning_content")
                        usage = delta.get("usage")

                        if usage:
                            tokens_received = usage.get("completion_tokens", tokens_received)
                        if content_piece:
                            if not time_at_first_token:
                                # Mark TTFT on first tokenized arrival
                                time_at_first_token = time.monotonic()
                            generated_text += content_piece

                        finish_reason = data["choices"][0].get("finish_reason", None)
                        if finish_reason and data.get("usage"):
                            usage = data["usage"]
                            num_prompt_tokens = usage.get("prompt_tokens")
                            tokens_received = usage.get("completion_tokens", tokens_received)
                            break
                    except (IndexError, KeyError):
                        previous_data = data
                        continue

                    previous_data = data

                end_time = time.monotonic()

                if not tokens_received:
                    tokens_received = self.sampler.get_token_length(
                        generated_text, add_special_tokens=False
                    )

                if not time_at_first_token:
                    time_at_first_token = end_time

                return UserChatResponse(
                    status_code=200,
                    generated_text=generated_text,
                    tokens_received=tokens_received,
                    time_at_first_token=time_at_first_token,
                    num_prefill_tokens=num_prompt_tokens,
                    start_time=start_time,
                    end_time=end_time,
                )

            elif isinstance(req, UserEmbeddingRequest):
                endpoint = "/v1/embeddings"
                payload = {
                    "model": req.model,
                    "input": req.documents,
                    **req.additional_request_params,
                }
                start_time = time.monotonic()
                r = requests.post(
                    url=f"{self.api_base}{endpoint}", json=payload, headers=self.headers
                )
                end_time = time.monotonic()
                if r.status_code == 200:
                    data = r.json()
                    num_prompt_tokens = data.get("usage", {}).get("prompt_tokens")
                    return UserResponse(
                        status_code=200,
                        start_time=start_time,
                        end_time=end_time,
                        time_at_first_token=end_time,
                        num_prefill_tokens=num_prompt_tokens,
                    )
                else:
                    return UserResponse(status_code=r.status_code, error_message=r.text)

            else:
                return UserResponse(status_code=400, error_message="Unsupported request type")
        except requests.exceptions.ConnectionError as e:
            return UserResponse(status_code=503, error_message=f"Connection error: {e}")
        except requests.exceptions.Timeout as e:
            return UserResponse(status_code=408, error_message=f"Request timed out: {e}")
        except requests.exceptions.RequestException as e:
            return UserResponse(status_code=500, error_message=str(e))

    async def _send_one(self, req) -> None:
        response = self._send_request(req)
        # Convert to RequestLevelMetrics and add to collector
        collector = RequestMetricsCollector()
        if response.status_code == 200:
            collector.calculate_metrics(response)
        else:
            collector.metrics.error_code = response.status_code
            collector.metrics.error_message = response.error_message
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

    def run(
        self,
        *,
        qps_level: float,
        duration_s: int,
        distribution: str,
        random_seed: int,
        max_requests: Optional[int],
        max_time_s: Optional[int],
        scenario: str,
    ) -> float:
        intervals = self._wait_intervals(qps_level, duration_s, random_seed, distribution)
        n = len(intervals)
        if max_requests is not None:
            n = min(n, max_requests)
            intervals = intervals[:n]

        prepared = [self._prepare_request(scenario) for _ in range(n)]

        async def produce():
            # Periodic UI tick to advance time-based progress even before first completion
            done_flag = {"done": False}

            async def tick_progress():
                if self.dashboard is None:
                    return
                while not done_flag["done"]:
                    try:
                        progress = self.dashboard.calculate_time_based_progress()
                        self.dashboard.update_benchmark_progress_bars(progress)
                    except Exception:
                        pass
                    await asyncio.sleep(0.5)

            tick_task = None
            if self.dashboard is not None:
                tick_task = asyncio.create_task(tick_progress())
            tasks = []
            for wait_s, req in zip(intervals, prepared):
                tasks.append(asyncio.create_task(self._send_one(req)))
                await asyncio.sleep(wait_s)
            if tasks:
                await asyncio.gather(*tasks)
            if tick_task is not None:
                done_flag["done"] = True
                # Give one last update chance
                await asyncio.sleep(0)
                tick_task.cancel()

        start = time.monotonic()
        try:
            if max_time_s is not None and max_time_s > 0:
                asyncio.run(asyncio.wait_for(produce(), timeout=max_time_s))
            else:
                asyncio.run(produce())
        except asyncio.TimeoutError:
            logger.info("Open-loop run timed out per max_time_s")
        end = time.monotonic()
        return end - start


