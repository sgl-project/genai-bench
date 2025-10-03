from locust import task

import json
import time
from typing import Any, Optional

from oci.generative_ai_inference import GenerativeAiInferenceClient
from oci.generative_ai_inference.models import (
    ChatDetails,
    ChatResult,
    DedicatedServingMode,
    GenericChatRequest,
    OnDemandServingMode,
)

from genai_bench.auth.model_auth_provider import ModelAuthProvider
from genai_bench.logging import init_logger
from genai_bench.protocol import (
    UserChatRequest,
    UserChatResponse,
    UserRequest,
    UserResponse,
)
from genai_bench.user.base_user import BaseUser

logger = init_logger(__name__)

# OCI client timeout constants
OCI_CONNECT_TIMEOUT = 60
OCI_READ_TIMEOUT = 300


class OCIGenAIUser(BaseUser):
    """User class for OCI GenAI models API with OCI authentication."""

    BACKEND_NAME = "oci-genai"
    supported_tasks = {
        "text-to-text": "chat",
    }
    host: Optional[str] = None
    auth_provider: Optional[ModelAuthProvider] = None

    def on_start(self):
        """Initialize OCI client on start."""
        super().on_start()
        if not self.auth_provider:
            raise ValueError("Auth is required for OCIGenAIUser")

        # Get config and signer from auth provider
        config = self.auth_provider.get_config()
        signer = self.auth_provider.get_credentials()

        self.client = GenerativeAiInferenceClient(
            config=config,
            signer=signer,
            service_endpoint=self.host,
            timeout=(OCI_CONNECT_TIMEOUT, OCI_READ_TIMEOUT),
        )
        logger.debug("Generative AI Inference Client initialized.")

    def send_request(
        self,
        endpoint: str,
        payload: Any,
        parse_strategy: Any,
        num_prefill_tokens: Optional[int] = None,
    ) -> UserResponse:
        logger.debug(f"Sending request with payload: {payload}")
        try:
            start_time = time.monotonic()
            response = self.client.chat(payload)
            non_stream_post_end_time = time.monotonic()
            if response.status == 200:
                metrics_response = parse_strategy(
                    payload,
                    response,
                    start_time,
                    num_prefill_tokens,
                    non_stream_post_end_time,
                )
            else:
                logger.warning(
                    f"Received error status-code: {response.status} "
                    f"RequestId: {response.request_id} "
                    f"Response: {response.response}"
                )
                metrics_response = UserResponse(
                    status_code=response.status,
                    error_message="Request Failed",
                )
            self.collect_metrics(metrics_response, endpoint)
            return metrics_response
        except Exception as e:
            logger.warning(f"Error: {e}")
            return UserResponse(
                status_code=500,
                error_message=str(e),
                num_prefill_tokens=num_prefill_tokens or 0,
            )

    @task
    def chat(self):
        """Send a chat completion request using OCI GenAI Service format."""
        user_request = self.sample()

        if not isinstance(user_request, UserChatRequest):
            raise AttributeError(
                f"Expected UserChatRequest for OCIGenAIUser.chat, got "
                f"{type(user_request)}"
            )

        compartment_id = self.get_compartment_id(user_request)
        serving_mode = self.get_serving_mode(user_request)

        # Construct chat request for OCI GenAI Service format using GENERIC API format
        messages = self.build_messages(user_request)

        chat_request = GenericChatRequest(
            api_format="GENERIC",
            messages=messages,
            max_tokens=user_request.max_tokens,
            is_stream=True,
            temperature=user_request.additional_request_params.get("temperature", 0.75),
            top_p=user_request.additional_request_params.get("top_p", 0.7),
            top_k=user_request.additional_request_params.get("top_k", 1),
            frequency_penalty=user_request.additional_request_params.get(
                "frequency_penalty", None
            ),
            presence_penalty=user_request.additional_request_params.get(
                "presence_penalty", None
            ),
            num_generations=1,
        )

        # Define payload with compartment ID and serving mode
        chat_detail = ChatDetails(
            compartment_id=compartment_id,
            serving_mode=serving_mode,
            chat_request=chat_request,
        )

        return self.send_request(
            endpoint="chat",
            payload=chat_detail,
            parse_strategy=self.parse_chat_response,
            num_prefill_tokens=user_request.num_prefill_tokens,
        )

    def build_messages(self, user_request: UserChatRequest) -> list:
        """Build messages array in OCI GenAI format."""
        messages = []

        # Add system message if provided
        system_message = user_request.additional_request_params.get("system_message")
        if system_message:
            messages.append(
                {
                    "role": "SYSTEM",
                    "content": [{"text": system_message, "type": "TEXT"}],
                }
            )

        # Add conversation history if provided
        chat_history = user_request.additional_request_params.get("chat_history", [])
        for msg in chat_history:
            # Convert from OpenAI format to OCI format if needed
            if isinstance(msg.get("content"), str):
                # Convert simple string content to OCI format
                oci_msg = {
                    "role": msg["role"].upper(),
                    "content": [{"text": msg["content"], "type": "TEXT"}],
                }
            else:
                # Assume it's already in OCI format
                oci_msg = msg
            messages.append(oci_msg)

        # Add current user message
        messages.append(
            {"role": "USER", "content": [{"text": user_request.prompt, "type": "TEXT"}]}
        )

        return messages

    def get_compartment_id(self, user_request: UserRequest):
        compartment_id = user_request.additional_request_params.get("compartmentId")
        if not compartment_id:
            raise ValueError("compartmentId missing in additional request params")
        return compartment_id

    def get_serving_mode(self, user_request: UserRequest) -> Any:
        params = user_request.additional_request_params
        model_id = user_request.model
        serving_type = params.get("servingType", "ON_DEMAND")

        if serving_type == "DEDICATED":
            endpoint_id = params.get("endpointId")
            if not endpoint_id:
                raise ValueError(
                    "endpointId must be provided for DEDICATED servingType"
                )
            logger.debug(
                f"Using DedicatedServingMode {serving_type} with "
                f"endpoint ID: {endpoint_id}"
            )
            return DedicatedServingMode(endpoint_id=endpoint_id)
        else:
            logger.debug(
                f"Using OnDemandServingMode {serving_type} with model ID: {model_id}"
            )
            return OnDemandServingMode(model_id=model_id)

    def parse_chat_response(
        self,
        request: ChatDetails,
        response: ChatResult,
        start_time: float,
        num_prefill_tokens: int,
        _: float,
    ) -> UserResponse:
        """
        Parses the streaming response from OCI GenAI Service using OCI's format.

        Args:
            request (ChatDetails): OCI GenAI Chat request.
            response (ChatResult): The streaming response from the API.
            start_time (float): Timestamp of request initiation.
            num_prefill_tokens (int): Number of tokens in the prompt.
            _ (float): Placeholder for unused variable.

        Returns:
            UserResponse: Parsed response in the UserResponse format.
        """
        generated_text = ""
        tokens_received = 0
        time_at_first_token: Optional[float] = None
        finish_reason = None
        previous_data = None

        # Iterate over each event in the streaming response
        for event in response.data.events():
            # Raw event data from the stream
            event_data = event.data.strip()
            # Parse the event data as JSON
            try:
                parsed_data = json.loads(event_data)
                finish_reason = parsed_data.get("finishReason", None)
                if not finish_reason:
                    # Extract text content from OCI GenAI format
                    message = parsed_data.get("message", {})
                    content_array = message.get("content", [])
                    if content_array and len(content_array) > 0:
                        text_segment = content_array[0].get("text", "")
                        if text_segment:
                            # Capture the time at the first token
                            if not time_at_first_token:
                                time_at_first_token = time.monotonic()
                                logger.debug(
                                    f"First token received at: {time_at_first_token}"
                                )
                            generated_text += text_segment
                            tokens_received += 1  # each event contains one token
                            logger.debug(
                                f"Text: '{text_segment}', "
                                f"tokens received: {tokens_received}"
                            )
                    # Track the previous data for debugging purposes
                    previous_data = parsed_data
                else:
                    # we have reached the end
                    logger.debug(
                        f"We have reached the end of the response "
                        f"with finish reason: {finish_reason}"
                    )
                    break
            except json.JSONDecodeError:
                logger.warning(
                    f"Error decoding JSON from event data: {event_data}, "
                    f"previous data: {previous_data}, "
                    f"finish reason: {finish_reason}"
                )
                continue

        # End timing for response
        end_time = time.monotonic()
        logger.debug(
            f"Generated text: {generated_text} \n"
            f"Time at first token: {time_at_first_token} \n"
            f"Finish reason: {finish_reason}\n"
            f"Completion Tokens: {tokens_received}\n"
            f"Start Time: {start_time}\n"
            f"End Time: {end_time}"
        )
        # Log if token count was not captured accurately
        if not tokens_received:
            tokens_received = len(generated_text.split())

        # Ensure time_at_first_token is never None (fallback to end_time)
        if time_at_first_token is None:
            time_at_first_token = end_time
            logger.warning("time_at_first_token was None, using end_time as fallback")

        return UserChatResponse(
            status_code=200,
            generated_text=generated_text,
            tokens_received=tokens_received,
            time_at_first_token=time_at_first_token,
            num_prefill_tokens=num_prefill_tokens,
            start_time=start_time,
            end_time=end_time,
        )
