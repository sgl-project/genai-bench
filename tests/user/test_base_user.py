from locust.env import Environment

from unittest.mock import MagicMock

import pytest

from genai_bench.protocol import UserChatRequest, UserChatResponse, UserResponse
from genai_bench.user.base_user import BaseUser


# Concrete subclass for testing
class ConcreteUser(BaseUser):
    host = "http://example.com"


@pytest.fixture
def locust_environment():
    """
    Set up a real Locust environment with actual stats logging to simulate
    the scenario without using mocks for runner.stats or runner.on_request.
    """
    # Initialize Locust Environment with a TaskSet
    environment = Environment(user_classes=[ConcreteUser])
    environment.create_local_runner()
    return environment


@pytest.fixture
def mock_environment(locust_environment):
    environment = locust_environment
    environment.scenario = MagicMock()
    environment.sampler = MagicMock()
    environment.sampler.sample = lambda x: UserChatRequest(
        model="gpt-3",
        prompt="Hello",
        num_prefill_tokens=1,
        max_tokens=5,
        additional_request_params={"temperature": 0.7},
    )
    return environment


class TestBaseUser:
    def test_init_raises_type_error(self):
        with pytest.raises(TypeError):
            BaseUser()

    def test_sample(self, mock_environment):
        user = ConcreteUser(environment=mock_environment)
        user_request = user.sample()
        assert user_request.model == "gpt-3"
        assert user_request.max_tokens == 5
        assert user_request.num_prefill_tokens == 1
        assert user_request.prompt == "Hello"
        assert user_request.additional_request_params == {"temperature": 0.7}

    def test_on_start_raises_attribute_error(self):
        environment = MagicMock()
        environment.sampler = None

        user = ConcreteUser(environment=environment)
        with pytest.raises(AttributeError):
            user.sample()

    def test_collect_metrics_success_with_chat(self, mock_environment):
        user_response = UserChatResponse(
            status_code=200,
            generated_text="random",
            tokens_received=2,
            time_at_first_token=2,
            num_prefill_tokens=1,
            start_time=0,
            end_time=3,
        )
        endpoint = "/test-endpoint"

        user = ConcreteUser(mock_environment)
        user.collect_metrics(user_response, endpoint)

        request_stats = mock_environment.runner.stats.get("/test-endpoint", "POST")

        assert request_stats.num_requests == 1
        assert request_stats.total_response_time > 0
        assert request_stats.total_content_length > 0

    def test_collect_metrics_success_with_embeddings(self, mock_environment):
        user_response = UserResponse(
            status_code=200,
            time_at_first_token=2,
            num_prefill_tokens=1,
            start_time=0,
            end_time=3,
        )
        endpoint = "/embeddings"

        user = ConcreteUser(mock_environment)
        user.collect_metrics(user_response, endpoint)

        request_stats = mock_environment.runner.stats.get("/embeddings", "POST")

        assert request_stats.num_requests == 1
        assert request_stats.total_response_time > 0
        assert request_stats.total_content_length == 0  # No decode returns

    def test_collect_metrics_failure(self, mock_environment):
        # Set up user response with an error
        user_response = UserResponse(
            status_code=500, error_message="Internal Server Error"
        )
        endpoint = "/test-endpoint"

        user = ConcreteUser(mock_environment)
        user.collect_metrics(user_response, endpoint)

        request_stats = mock_environment.runner.stats.get("/test-endpoint", "POST")

        assert request_stats.num_failures == 1
