"""Customized user for Baseten backends."""

import time
from typing import Any, Callable, Dict, Optional

import requests
from requests import Response

from genai_bench.protocol import UserResponse
from genai_bench.user.openai_user import OpenAIUser


class BasetenUser(OpenAIUser):
    """Baseten user that uses the full URL as endpoint."""
    
    BACKEND_NAME = "baseten"
    disable_streaming: bool = False
    
    def send_request(
        self,
        stream: bool,
        endpoint: str,
        payload: Dict[str, Any],
        parse_strategy: Callable[..., UserResponse],
        num_prefill_tokens: Optional[int] = None,
    ) -> UserResponse:
        """
        Override send_request to use the full URL for Baseten.
        
        For Baseten, the host is already the full endpoint URL, so we don't
        append the endpoint path.
        """
        response = None

        try:
            start_time = time.monotonic()
            # For Baseten, use the host directly as the URL
            response = requests.post(
                url=self.host,  # Use host directly instead of host + endpoint
                json=payload,
                stream=stream,
                headers=self.headers,
            )
            non_stream_post_end_time = time.monotonic()

            if response.status_code == 200:
                metrics_response = parse_strategy(
                    response,
                    start_time,
                    num_prefill_tokens,
                    non_stream_post_end_time,
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
                status_code=500,  # Assign a generic 500
                error_message=str(e),
            )
        finally:
            if response is not None:
                response.close()

        self.collect_metrics(metrics_response, endpoint)
        return metrics_response 