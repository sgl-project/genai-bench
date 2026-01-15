"""Tests for BaseFastUser.collect_metrics with fire_event parameter."""

from locust.env import Environment

from unittest.mock import MagicMock

import pytest

from genai_bench.protocol import UserChatResponse, UserResponse
from genai_bench.user.fast_http_user import BaseFastUser


# Concrete subclass for testing
class ConcreteFastUser(BaseFastUser):
    host = "http://example.com"


@pytest.fixture
def locust_environment():
    """Set up a real Locust environment."""
    environment = Environment(user_classes=[ConcreteFastUser])
    environment.create_local_runner()
    return environment


@pytest.fixture
def mock_environment(locust_environment):
    """Set up a mocked environment with sampler."""
    environment = locust_environment
    environment.scenario = MagicMock()
    environment.sampler = MagicMock()
    return environment


class TestBaseFastUserCollectMetrics:
    """Tests for collect_metrics method with fire_event parameter."""

    def test_collect_metrics_fire_event_true_success(self, mock_environment):
        """Test that fire_event=True triggers Locust event on success."""
        user_response = UserChatResponse(
            status_code=200,
            generated_text="Hello, world!",
            tokens_received=5,
            time_at_first_token=0.2,
            num_prefill_tokens=10,
            start_time=0.0,
            end_time=1.0,
        )
        endpoint = "/v1/chat/completions"

        user = ConcreteFastUser(mock_environment)

        # Track event fires
        fire_count = 0
        original_fire = mock_environment.events.request.fire

        def count_fire(*args, **kwargs):
            nonlocal fire_count
            fire_count += 1
            return original_fire(*args, **kwargs)

        mock_environment.events.request.fire = count_fire

        # Call with fire_event=True (explicit)
        user.collect_metrics(user_response, endpoint, fire_event=True)

        # Verify event was fired
        assert fire_count == 1

        # Verify stats
        request_stats = mock_environment.runner.stats.get(endpoint, "POST")
        assert request_stats.num_requests == 1
        assert request_stats.total_response_time > 0
        assert request_stats.total_content_length == 5  # tokens_received

    def test_collect_metrics_fire_event_false_success(self, mock_environment):
        """Test that fire_event=False does not trigger Locust event on success."""
        user_response = UserChatResponse(
            status_code=200,
            generated_text="Hello, world!",
            tokens_received=5,
            time_at_first_token=0.2,
            num_prefill_tokens=10,
            start_time=0.0,
            end_time=1.0,
        )
        endpoint = "/v1/chat/completions"

        user = ConcreteFastUser(mock_environment)

        # Track event fires
        fire_count = 0
        original_fire = mock_environment.events.request.fire

        def count_fire(*args, **kwargs):
            nonlocal fire_count
            fire_count += 1
            return original_fire(*args, **kwargs)

        mock_environment.events.request.fire = count_fire

        # Call with fire_event=False
        user.collect_metrics(user_response, endpoint, fire_event=False)

        # Verify event was NOT fired
        assert fire_count == 0

        # Verify stats were NOT updated (no event fired)
        request_stats = mock_environment.runner.stats.get(endpoint, "POST")
        assert request_stats.num_requests == 0

    def test_collect_metrics_default_parameter_success(self, mock_environment):
        """Test that default fire_event=True maintains backward compatibility."""
        user_response = UserChatResponse(
            status_code=200,
            generated_text="Test",
            tokens_received=2,
            time_at_first_token=0.1,
            num_prefill_tokens=5,
            start_time=0.0,
            end_time=0.5,
        )
        endpoint = "/v1/chat/completions"

        user = ConcreteFastUser(mock_environment)

        # Call without fire_event parameter (should default to True)
        user.collect_metrics(user_response, endpoint)

        # Verify event was fired (backward compatibility)
        request_stats = mock_environment.runner.stats.get(endpoint, "POST")
        assert request_stats.num_requests == 1

    def test_collect_metrics_fire_event_true_failure(self, mock_environment):
        """Test that fire_event=True triggers Locust event on failure."""
        user_response = UserResponse(
            status_code=500, error_message="Internal Server Error"
        )
        endpoint = "/v1/chat/completions"

        user = ConcreteFastUser(mock_environment)

        # Track event fires
        fire_count = 0
        original_fire = mock_environment.events.request.fire

        def count_fire(*args, **kwargs):
            nonlocal fire_count
            fire_count += 1
            return original_fire(*args, **kwargs)

        mock_environment.events.request.fire = count_fire

        # Call with fire_event=True
        user.collect_metrics(user_response, endpoint, fire_event=True)

        # Verify event was fired
        assert fire_count == 1

        # Verify failure was recorded
        request_stats = mock_environment.runner.stats.get(endpoint, "POST")
        assert request_stats.num_failures == 1

    def test_collect_metrics_fire_event_false_failure(self, mock_environment):
        """Test that fire_event=False does not trigger Locust event on failure."""
        user_response = UserResponse(
            status_code=500, error_message="Internal Server Error"
        )
        endpoint = "/v1/chat/completions"

        user = ConcreteFastUser(mock_environment)

        # Track event fires
        fire_count = 0
        original_fire = mock_environment.events.request.fire

        def count_fire(*args, **kwargs):
            nonlocal fire_count
            fire_count += 1
            return original_fire(*args, **kwargs)

        mock_environment.events.request.fire = count_fire

        # Call with fire_event=False
        user.collect_metrics(user_response, endpoint, fire_event=False)

        # Verify event was NOT fired
        assert fire_count == 0

        # Verify stats were NOT updated
        request_stats = mock_environment.runner.stats.get(endpoint, "POST")
        assert request_stats.num_failures == 0

    def test_collect_metrics_always_sends_custom_metrics(self, mock_environment):
        """Test that runner.send_message is always called regardless of fire_event."""
        user_response = UserChatResponse(
            status_code=200,
            generated_text="Test",
            tokens_received=3,
            time_at_first_token=0.15,
            num_prefill_tokens=8,
            start_time=0.0,
            end_time=0.8,
        )
        endpoint = "/v1/chat/completions"

        user = ConcreteFastUser(mock_environment)

        # Track send_message calls
        send_count = 0
        original_send = mock_environment.runner.send_message

        def count_send(*args, **kwargs):
            nonlocal send_count
            send_count += 1
            if original_send:
                return original_send(*args, **kwargs)

        mock_environment.runner.send_message = count_send

        # Test with fire_event=True
        user.collect_metrics(user_response, endpoint, fire_event=True)
        assert send_count == 1

        # Test with fire_event=False
        user.collect_metrics(user_response, endpoint, fire_event=False)
        assert send_count == 2  # Should be called both times

    def test_collect_metrics_logger_warning_on_failure(self, mock_environment, caplog):
        """Test that logger.warning is called on failure regardless of fire_event."""
        user_response = UserResponse(status_code=404, error_message="Not Found")
        endpoint = "/v1/chat/completions"

        user = ConcreteFastUser(mock_environment)

        # Test with fire_event=True
        user.collect_metrics(user_response, endpoint, fire_event=True)
        assert "Received error response from server" in caplog.text
        assert "404" in caplog.text

        caplog.clear()

        # Test with fire_event=False
        user.collect_metrics(user_response, endpoint, fire_event=False)
        assert "Received error response from server" in caplog.text
        assert "404" in caplog.text

    def test_collect_metrics_embeddings_response(self, mock_environment):
        """
        Test collect_metrics with embeddings.

        UserResponse without generated_text.
        """
        user_response = UserResponse(
            status_code=200,
            time_at_first_token=0.1,
            num_prefill_tokens=100,
            start_time=0.0,
            end_time=0.5,
        )
        endpoint = "/v1/embeddings"

        user = ConcreteFastUser(mock_environment)
        user.collect_metrics(user_response, endpoint, fire_event=True)

        # Verify stats
        request_stats = mock_environment.runner.stats.get(endpoint, "POST")
        assert request_stats.num_requests == 1
        assert (
            request_stats.total_content_length == 0
        )  # No output tokens for embeddings
