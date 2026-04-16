from genai_bench.logging import init_logger
from genai_bench.metrics.metrics import RequestLevelMetrics
from genai_bench.protocol import (
    UserAudioTranscriptionResponse,
    UserChatResponse,
    UserImageGenerationResponse,
    UserResponse,
)

logger = init_logger(__name__)


class RequestMetricsCollector:
    """
    A class to collect and calculate metrics for individual requests.

    Attributes:
        metrics (RequestLevelMetrics): An instance to store metrics related
            to a single request.
    """

    def __init__(self):
        self.metrics = RequestLevelMetrics()

    def calculate_metrics(
        self,
        response: UserResponse,
    ):
        """
        Calculates various metrics from the response of a request.

        Args:
            response (UserResponse): The customized UserResponse object
                containing the response data needed to calculate metrics.
        """
        assert (
            response.num_prefill_tokens is not None
        ), "response.num_prefill_tokens is None"
        assert (
            response.time_at_first_token is not None
        ), "response.time_at_first_token is None"
        assert response.start_time is not None, "response.start_time is None"
        assert response.end_time is not None, "response.end_time is None"

        # Safely calculate common metrics
        self.metrics.num_input_tokens = response.num_prefill_tokens
        self.metrics.ttft = response.time_at_first_token - response.start_time
        self.metrics.e2e_latency = response.end_time - response.start_time
        self.metrics.total_tokens = self.metrics.num_input_tokens

        # Calculate prefill throughput
        self.metrics.input_throughput = (
            self.metrics.num_input_tokens / self.metrics.ttft
            if self.metrics.ttft
            else 0
        )

        # Check if the response is a UserChatResponse for output metrics
        if isinstance(response, UserChatResponse):
            self._calculate_output_metrics(response)
        elif isinstance(response, UserAudioTranscriptionResponse):
            self._calculate_audio_output_metrics(response)
        elif isinstance(response, UserImageGenerationResponse):
            # For image generation (non-streaming), use same approach as embeddings
            # to avoid filter_metrics setting tpot/output_inference_speed to None
            self._reset_output_metrics()
        else:
            # For non-chat responses, reset output metrics to avoid NoneType
            # Error in AggregatedMetricsCollector
            self._reset_output_metrics()

    def _calculate_output_metrics(self, response: UserChatResponse):
        """
        Helper function to calculate output metrics from a UserChatResponse.
        """
        assert response.tokens_received is not None, "response.tokens_received is None"
        self.metrics.num_output_tokens = response.tokens_received
        self.metrics.num_reasoning_tokens = response.reasoning_tokens or 0
        self.metrics.total_tokens += self.metrics.num_output_tokens
        self.metrics.output_latency = self.metrics.e2e_latency - self.metrics.ttft

        # Avoid divide by zero for tokens
        if self.metrics.num_output_tokens > 1:
            self.metrics.tpot = self.metrics.output_latency / (
                self.metrics.num_output_tokens - 1
            )
            self.metrics.output_inference_speed = 1 / self.metrics.tpot
            self.metrics.output_throughput = (
                (self.metrics.num_output_tokens - 1) / self.metrics.output_latency
                if self.metrics.output_latency
                else 0
            )
        else:
            logger.warning(
                f"‼️ num_output_tokens:"
                f"{self.metrics.num_output_tokens} is <= 1. Please check"
                f" your server and request!"
            )

    def _calculate_audio_output_metrics(self, response: UserAudioTranscriptionResponse):
        """
        Helper function to calculate output metrics for audio transcription.

        - output_inference_speed stores Real-Time Factor (RTF):
            RTF = audio_duration_s / e2e_latency
          Higher RTF = faster than real-time. This matches Ming's benchmark graphs.
        - num_prefill_tokens (set in parse_transcription_response) stores
          audio duration in centiseconds (1 audio-s = 100 units), so
          mean_input_throughput_tokens_per_s = audio centiseconds processed per
          wall second. Divide by 100 to convert to audio-seconds/wall-second.
        - output_throughput stores audio-seconds/wall-second (RTF equivalent
          at the server level), computed as sum(audio_duration_s) / run_duration
          via num_input_tokens sum / 100 / run_duration — but here we store the
          per-request value for aggregation.
        """
        char_count = len(response.transcribed_text or "")
        self.metrics.num_output_tokens = char_count
        self.metrics.num_reasoning_tokens = 0
        self.metrics.total_tokens += char_count
        self.metrics.output_latency = self.metrics.e2e_latency

        # Real-Time Factor: how many seconds of audio processed per second of wall time
        audio_duration_s = response.audio_duration_s or 0.0
        if self.metrics.e2e_latency and self.metrics.e2e_latency > 0:
            rtf = audio_duration_s / self.metrics.e2e_latency
        else:
            rtf = 0.0

        self.metrics.output_inference_speed = rtf
        # tpot and output_throughput are not meaningful for audio; set to 0
        self.metrics.tpot = 0
        self.metrics.output_throughput = 0

    def _reset_output_metrics(self):
        """Helper function to reset all output-related metrics to 0."""
        for field in RequestLevelMetrics.OUTPUT_METRICS_FIELDS:
            setattr(self.metrics, field, 0)
