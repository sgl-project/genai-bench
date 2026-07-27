"""User class for OCI's OpenAI-compatible endpoints."""

from locust import task

import json
import os
import time
from collections.abc import Callable, Iterator
from typing import Any, cast

import httpx
import requests
from oci_openai import (
    OciInstancePrincipalAuth,
    OciResourcePrincipalAuth,
    OciSessionAuth,
    OciUserPrincipalAuth,
)
from openai import OpenAI

from genai_bench.logging import init_logger
from genai_bench.protocol import (
    UserChatResponse,
    UserImageGenerationRequest,
    UserImageGenerationResponse,
    UserResponse,
    UserTextToSpeechRequest,
    UserTextToSpeechResponse,
)
from genai_bench.user.openai_user import OpenAIUser

logger = init_logger(__name__)

OCI_AUTH_CLASS_MAP = {
    "oci_security_token": OciSessionAuth,
    "oci_user_principal": OciUserPrincipalAuth,
    "oci_instance_principal": OciInstancePrincipalAuth,
    "oci_obo_token": OciResourcePrincipalAuth,
}

DEFAULT_STREAM_CHUNK_SIZE = 1
DEFAULT_MAX_PLAUSIBLE_OUTPUT_TPS = 2000.0
COALESCED_STREAM_STATUS_CODE = 598


class OCIOpenAIUser(OpenAIUser):
    """User class for OCI's OpenAI-compatible endpoints."""

    BACKEND_NAME = "oci-openai"
    supported_tasks = {
        "text-to-text": "chat",
        "text-to-image": "images_generations",
        "text-to-speech": "speech",
    }

    def on_start(self):
        if not self.host or not self.auth_provider:
            raise ValueError("Host and Auth is required for OCIOpenAIUser.")
        auth_type = self.auth_provider.get_auth_type()
        auth_cls = OCI_AUTH_CLASS_MAP.get(auth_type)
        if auth_cls is None:
            raise ValueError(
                f"Unsupported OCI auth type: {auth_type}. "
                f"Supported: {list(OCI_AUTH_CLASS_MAP)}"
            )

        if auth_type in ("oci_security_token", "oci_user_principal"):
            profile = getattr(self.auth_provider.oci_auth, "profile", "DEFAULT")
            config_file = getattr(self.auth_provider.oci_auth, "config_path", None)
            kwargs = {"profile_name": profile}
            if config_file:
                kwargs["config_file"] = config_file
            oci_auth = auth_cls(**kwargs)
        else:
            oci_auth = auth_cls()

        self._oci_auth = oci_auth
        self.openai_client = OpenAI(
            api_key="OCI",
            base_url=self.host,
            http_client=httpx.Client(auth=oci_auth, trust_env=False),
        )
        self.api_backend = getattr(self, "api_backend", self.BACKEND_NAME)
        super(OpenAIUser, self).on_start()

    @staticmethod
    def _is_token_bearing_sse_line(line: bytes | str) -> bool:
        text = (
            line.decode("utf-8", errors="replace") if isinstance(line, bytes) else line
        )
        if not text.startswith("data:"):
            return False

        data_text = text[5:].strip()
        if not data_text or data_text == "[DONE]":
            return False

        try:
            event = json.loads(data_text)
        except json.JSONDecodeError:
            return False

        for choice in event.get("choices") or []:
            delta = choice.get("delta") or {}
            if (
                delta.get("content")
                or delta.get("reasoning")
                or delta.get("reasoning_content")
            ):
                return True
        return False

    @classmethod
    def _use_low_latency_iter_lines(cls, response: requests.Response) -> None:
        """Yield SSE lines as bytes arrive and count independently timed events."""
        configured_chunk_size = int(
            os.environ.get(
                "OCI_OPENAI_STREAM_CHUNK_SIZE",
                str(DEFAULT_STREAM_CHUNK_SIZE),
            )
        )
        if configured_chunk_size <= 0:
            raise ValueError("OCI_OPENAI_STREAM_CHUNK_SIZE must be a positive integer")

        response_state = cast(Any, response)
        original_iter_lines = cast(
            Callable[..., Iterator[bytes | str]],
            response.iter_lines,
        )
        response_state._genai_bench_token_event_count = 0

        def iter_lines(
            chunk_size: int | None = None,
            decode_unicode: bool = False,
            delimiter: bytes | str | None = None,
        ) -> Iterator[bytes | str]:
            effective_chunk_size = (
                configured_chunk_size if chunk_size is None else chunk_size
            )
            for line in original_iter_lines(
                chunk_size=effective_chunk_size,
                decode_unicode=decode_unicode,
                delimiter=delimiter,
            ):
                if cls._is_token_bearing_sse_line(line):
                    response_state._genai_bench_token_event_count += 1
                yield line

        response_state.iter_lines = iter_lines

    @staticmethod
    def _reject_unreliable_stream_timing(
        metrics_response: UserResponse,
        response: requests.Response,
    ) -> UserResponse:
        """Reject coalesced streams instead of reporting invalid decode timing."""
        if not isinstance(metrics_response, UserChatResponse):
            return metrics_response

        first_token_time = metrics_response.time_at_first_token
        end_time = metrics_response.end_time
        tokens_received = metrics_response.tokens_received
        if first_token_time is None or end_time is None or tokens_received is None:
            return metrics_response
        if tokens_received <= 1:
            return metrics_response

        max_output_tps = float(
            os.environ.get(
                "OCI_OPENAI_MAX_PLAUSIBLE_OUTPUT_TPS",
                str(DEFAULT_MAX_PLAUSIBLE_OUTPUT_TPS),
            )
        )
        if max_output_tps <= 0:
            raise ValueError(
                "OCI_OPENAI_MAX_PLAUSIBLE_OUTPUT_TPS must be a positive number"
            )

        token_event_count = int(getattr(response, "_genai_bench_token_event_count", 0))
        output_duration_s = end_time - first_token_time
        observed_output_tps = (
            float("inf")
            if output_duration_s <= 0
            else (tokens_received - 1) / output_duration_s
        )
        if token_event_count > 1 and observed_output_tps <= max_output_tps:
            return metrics_response

        reasons = []
        if token_event_count <= 1:
            reasons.append(f"only {token_event_count} token-bearing SSE event(s)")
        if observed_output_tps > max_output_tps:
            reasons.append(
                f"{observed_output_tps:.1f} tokens/s exceeds {max_output_tps:.1f}"
            )

        return UserResponse(
            status_code=COALESCED_STREAM_STATUS_CODE,
            error_message=(
                "Unreliable coalesced-stream timing: "
                + "; ".join(reasons)
                + f"; {tokens_received} output tokens arrived over "
                f"{output_duration_s:.6f}s after the first observed content. "
                "The request is rejected instead of recording invalid "
                "TTFT/TPOT/output-speed metrics."
            ),
            start_time=metrics_response.start_time,
            time_at_first_token=first_token_time,
            end_time=end_time,
            num_prefill_tokens=metrics_response.num_prefill_tokens,
        )

    def send_request(
        self,
        stream: bool,
        endpoint: str,
        payload: dict[str, Any],
        parse_strategy: Callable[..., UserResponse],
        num_prefill_tokens: int | None = None,
    ) -> UserResponse:
        """Send an OCI-signed request without buffering streaming token events."""
        response = None
        try:
            start_time = time.monotonic()
            request_payload = dict(payload)
            compartment_id = request_payload.pop(
                "compartmentId", None
            ) or os.environ.get("OCI_GENAI_COMPARTMENT_ID")
            headers = {
                "Content-Type": "application/json",
                "Accept-Encoding": "identity",
            }
            if compartment_id:
                headers["CompartmentId"] = compartment_id
                headers["opc-compartment-id"] = compartment_id

            request = requests.Request(
                method="POST",
                url=f"{self.host}{endpoint}",
                json=request_payload,
                headers=headers,
            )
            prepared = request.prepare()
            self._oci_auth.signer.do_request_sign(prepared)

            with requests.Session() as session:
                session.trust_env = False
                response = session.send(
                    prepared,
                    stream=stream,
                    timeout=(60, 600),
                )
                request_end_time = time.monotonic()

                if response.status_code == 200:
                    if stream:
                        self._use_low_latency_iter_lines(response)
                    metrics_response = parse_strategy(
                        response,
                        start_time,
                        num_prefill_tokens,
                        request_end_time,
                    )
                    if stream:
                        metrics_response = self._reject_unreliable_stream_timing(
                            metrics_response,
                            response,
                        )
                else:
                    metrics_response = UserResponse(
                        status_code=response.status_code,
                        error_message=response.text,
                    )
        except requests.exceptions.ConnectionError as e:
            metrics_response = UserResponse(
                status_code=503, error_message=f"Connection error: {e}"
            )
        except requests.exceptions.Timeout as e:
            metrics_response = UserResponse(
                status_code=408, error_message=f"Request timed out: {e}"
            )
        except requests.exceptions.RequestException as e:
            metrics_response = UserResponse(
                status_code=500,
                error_message=str(e),
            )
        finally:
            if response is not None:
                response.close()

        self.collect_metrics(metrics_response, endpoint)
        return metrics_response

    @task
    def images_generations(self):
        user_request = self.sample()

        if not isinstance(user_request, UserImageGenerationRequest):
            raise AttributeError(
                f"user_request should be of type "
                f"UserImageGenerationRequest for OCIOpenAIUser."
                f"images_generations, got {type(user_request)}"
            )

        compartment_id = user_request.additional_request_params.get("compartmentId")
        if not compartment_id:
            raise ValueError("compartmentId missing in additional request params")

        start_time = time.monotonic()
        try:
            # Filter out keys already passed as explicit SDK args
            sdk_params = {
                k: v
                for k, v in user_request.additional_request_params.items()
                if k not in ("compartmentId", "model", "prompt", "n", "size")
            }

            response = self.openai_client.images.generate(
                model=user_request.model,
                prompt=user_request.prompt,
                n=user_request.num_images,
                size=user_request.size,
                extra_headers={"CompartmentId": compartment_id},
                **sdk_params,
            )

            end_time = time.monotonic()
            generated_images = [img.url or img.b64_json for img in response.data if img]
            revised_prompt = response.data[0].revised_prompt if response.data else None

            metrics_response = UserImageGenerationResponse(
                status_code=200,
                start_time=start_time,
                end_time=end_time,
                time_at_first_token=end_time,
                generated_images=generated_images,
                revised_prompt=revised_prompt,
                num_prefill_tokens=0,
                images_generated=len(generated_images),
            )

        except Exception as e:
            logger.error(f"OCI image generation failed: {e}")
            metrics_response = UserImageGenerationResponse(
                status_code=getattr(e, "status_code", 500),
                error_message=str(e),
                start_time=start_time,
                end_time=time.monotonic(),
                time_at_first_token=None,
                generated_images=[],
                revised_prompt=None,
                num_prefill_tokens=0,
                images_generated=0,
            )

        self.collect_metrics(metrics_response, "/v1/images/generations")
        return metrics_response

    @task
    def speech(self):
        user_request = self.sample()

        if not isinstance(user_request, UserTextToSpeechRequest):
            raise AttributeError(
                f"user_request should be of type "
                f"UserTextToSpeechRequest for OCIOpenAIUser.speech, got "
                f"{type(user_request)}"
            )

        compartment_id = user_request.additional_request_params.get("compartmentId")
        if not compartment_id:
            raise ValueError("compartmentId missing in additional request params")

        filtered_params = {
            k: v
            for k, v in user_request.additional_request_params.items()
            if k not in ("compartmentId", "model", "input", "voice")
        }

        start_time = time.monotonic()
        try:
            with self.openai_client.audio.speech.with_streaming_response.create(
                model=user_request.model,
                voice=user_request.voice,
                input=user_request.input_text,
                extra_headers={"CompartmentId": compartment_id},
                **filtered_params,
            ) as response:
                time_at_first_token = None
                total_bytes = 0
                for chunk in response.iter_bytes(1024):
                    if time_at_first_token is None:
                        time_at_first_token = time.monotonic()
                    total_bytes += len(chunk)
                end_time = time.monotonic()

            if time_at_first_token is None:
                logger.warning("TTS response returned 200 but empty audio body")
                time_at_first_token = end_time

            logger.debug(
                f"TTS response: audio_bytes={total_bytes}, "
                f"ttft={time_at_first_token - start_time:.3f}s, "
                f"e2e_latency={end_time - start_time:.3f}s"
            )

            metrics_response = UserTextToSpeechResponse(
                status_code=200,
                start_time=start_time,
                end_time=end_time,
                time_at_first_token=time_at_first_token,
                num_prefill_tokens=0,
                audio_bytes=total_bytes,
            )

        except Exception as e:
            logger.error(f"OCI TTS failed: {e}")
            metrics_response = UserTextToSpeechResponse(
                status_code=getattr(e, "status_code", 500),
                error_message=str(e),
            )

        self.collect_metrics(metrics_response, "/v1/audio/speech")
        return metrics_response
