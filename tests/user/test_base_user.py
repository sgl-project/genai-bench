from locust.env import Environment

import time
from unittest.mock import MagicMock, patch

import pytest

from genai_bench.protocol import UserChatRequest, UserChatResponse, UserResponse
from genai_bench.rate_limiter import TokenBucketRateLimiter
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

    def test_acquire_rate_limit_token_with_rate_limiter_success(self, mock_environment):
        """Test acquire_rate_limit_token() when rate_limiter exists."""
        rate_limiter = TokenBucketRateLimiter(rate=10.0)
        mock_environment.rate_limiter = rate_limiter

        user = ConcreteUser(environment=mock_environment)
        # Should succeed and return True
        result = user.acquire_rate_limit_token()
        assert result is True

        # Token should have been consumed
        assert rate_limiter.get_available_tokens() < rate_limiter.bucket_size

    def test_acquire_rate_limit_token_with_rate_limiter_stopped(self, mock_environment):
        """Test acquire_rate_limit_token() when rate_limiter exists but is stopped."""
        rate_limiter = TokenBucketRateLimiter(rate=10.0)
        rate_limiter.stop()
        mock_environment.rate_limiter = rate_limiter

        user = ConcreteUser(environment=mock_environment)
        # Should return False when rate limiter is stopped
        result = user.acquire_rate_limit_token()
        assert result is False

        # Should not have consumed a token since limiter is stopped
        # (acquire returns False when stopped)

    def test_acquire_rate_limit_token_with_rate_limiter_none(self, mock_environment):
        """Test acquire_rate_limit_token() when rate_limiter is None."""
        mock_environment.rate_limiter = None

        user = ConcreteUser(environment=mock_environment)
        # Should return True when no rate limiter exists
        result = user.acquire_rate_limit_token()
        assert result is True

    def test_acquire_rate_limit_token_no_rate_limiter_attribute(self, mock_environment):
        """Test acquire_rate_limit_token() when environment has no rate_limiter attr."""
        # Remove rate_limiter attribute if it exists
        if hasattr(mock_environment, "rate_limiter"):
            delattr(mock_environment, "rate_limiter")

        user = ConcreteUser(environment=mock_environment)
        # Should return True without raising AttributeError
        result = user.acquire_rate_limit_token()
        assert result is True

    def test_acquire_rate_limit_token_blocks_until_token_available(
        self, mock_environment
    ):
        """Test that acquire_rate_limit_token() blocks until token is available."""
        rate_limiter = TokenBucketRateLimiter(rate=1.0)  # 1 req/s
        mock_environment.rate_limiter = rate_limiter

        # Consume the only token
        rate_limiter.acquire()

        user = ConcreteUser(environment=mock_environment)

        # This should block for approximately 1 second until token refills
        start_time = time.monotonic()
        result = user.acquire_rate_limit_token()
        elapsed = time.monotonic() - start_time

        # Should have waited approximately 1 second and returned True
        assert 0.8 <= elapsed <= 1.5
        assert result is True

    def test_acquire_rate_limit_token_called_before_request(self):
        """Test that acquire_rate_limit_token() is called before making requests."""
        from genai_bench.user.openai_user import OpenAIUser

        mock_environment = MagicMock()
        mock_environment.scenario = MagicMock()
        mock_environment.sampler = MagicMock()
        mock_environment.sampler.sample = lambda x: UserChatRequest(
            model="gpt-3",
            prompt="Hello",
            num_prefill_tokens=1,
            max_tokens=5,
        )
        mock_environment.runner = MagicMock()
        mock_environment.runner.stats = MagicMock()

        rate_limiter = MagicMock(spec=TokenBucketRateLimiter)
        rate_limiter.acquire.return_value = True
        mock_environment.rate_limiter = rate_limiter

        # Set host as class attribute before instantiation (required by Locust)
        OpenAIUser.host = "http://test.example.com"
        user = OpenAIUser(environment=mock_environment)

        # Mock the HTTP client to avoid actual requests
        with patch.object(user.client, "post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {
                "choices": [{"message": {"content": "response"}}]
            }
            mock_post.return_value.headers = {}
            mock_post.return_value.elapsed.total_seconds.return_value = 0.1

            # Call chat task
            user.chat()

            # Verify rate limiter acquire was called
            assert rate_limiter.acquire.called

    def test_acquire_rate_limit_token_skips_request_when_false(self):
        """Test that callers skip requests when token acquisition fails."""
        from genai_bench.user.openai_user import OpenAIUser

        mock_environment = MagicMock()
        mock_environment.scenario = MagicMock()
        mock_environment.sampler = MagicMock()
        mock_environment.sampler.sample = lambda x: UserChatRequest(
            model="gpt-3",
            prompt="Hello",
            num_prefill_tokens=1,
            max_tokens=5,
        )
        mock_environment.runner = MagicMock()
        mock_environment.runner.stats = MagicMock()

        rate_limiter = MagicMock(spec=TokenBucketRateLimiter)
        rate_limiter.acquire.return_value = False  # Token acquisition fails
        mock_environment.rate_limiter = rate_limiter

        # Set host as class attribute before instantiation (required by Locust)
        OpenAIUser.host = "http://test.example.com"
        user = OpenAIUser(environment=mock_environment)

        # Mock the HTTP client to verify it's not called
        with patch.object(user, "send_request") as mock_send_request:
            # Call chat task
            user.chat()

            # Verify rate limiter acquire was called
            assert rate_limiter.acquire.called
            # Verify send_request was NOT called (request was skipped)
            mock_send_request.assert_not_called()
