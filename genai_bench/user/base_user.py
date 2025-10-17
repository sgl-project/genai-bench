from locust import HttpUser

from typing import Dict, Optional, Callable

from genai_bench.logging import init_logger
from genai_bench.metrics.request_metrics_collector import RequestMetricsCollector
from genai_bench.protocol import UserRequest, UserResponse

logger = init_logger(__name__)


class BaseUser(HttpUser):
    supported_tasks: Dict[str, str] = {}
    wait_time: Optional[Callable] = None  # Can be set dynamically for QPS mode

    def __new__(cls, *args, **kwargs):
        if cls is BaseUser:
            raise TypeError("BaseUser is not meant to be instantiated directly.")
        return super().__new__(cls)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @classmethod
    def is_task_supported(cls, task: str) -> bool:
        return task in cls.supported_tasks

    def sample(self) -> UserRequest:
        if not (
            hasattr(self.environment, "scenario")
            and self.environment.scenario is not None
        ):
            raise AttributeError(
                f"Environment {self.environment} has no attribute "
                f"'scenario' or it is empty."
            )
        if not (
            hasattr(self.environment, "sampler")
            and self.environment.sampler is not None
        ):
            raise AttributeError(
                f"Environment {self.environment} has no attribute "
                f"'sampler' or it is empty."
            )
        return self.environment.sampler.sample(self.environment.scenario)

    def collect_metrics(
        self,
        user_response: UserResponse,
        endpoint: str,
    ):
        """
        Collects metrics based on the UserResponse.

        Args:
            user_response (UserResponse): The response object containing the
                request metrics.
            endpoint (str): The API endpoint for the request.
        """
        request_metrics_collector = RequestMetricsCollector()
        if user_response.status_code == 200:
            request_metrics_collector.calculate_metrics(user_response)
            self.environment.events.request.fire(
                request_type="POST",
                name=endpoint,
                response_time=request_metrics_collector.metrics.e2e_latency,
                response_length=request_metrics_collector.metrics.num_output_tokens,
            )
        else:
            # Handle error responses
            request_metrics_collector.metrics.error_code = user_response.status_code
            request_metrics_collector.metrics.error_message = (
                user_response.error_message
            )
            self.environment.events.request.fire(
                request_type="POST",
                name=endpoint,
                response_time=0,
                response_length=0,
                exception=f"Request failed with "
                f"status {user_response.status_code}, "
                f"message: {user_response.error_message}",
            )
            logger.warning(
                f"Received error response from server. Error code:"
                f" {user_response.status_code},"
                f" message: {user_response.error_message}."
            )

        # Send metrics to aggregated_metrics_collector
        self.environment.runner.send_message(
            "request_metrics", request_metrics_collector.metrics.model_dump_json()
        )  # type: ignore[attr-defined]
