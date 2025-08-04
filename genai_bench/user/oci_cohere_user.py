from locust import task

import json
import time
from typing import Any, Callable, List, Optional

from oci.generative_ai_inference import GenerativeAiInferenceClient
from oci.generative_ai_inference.models import (
    ChatDetails,
    ChatResult,
    CohereChatRequest,
    DedicatedServingMode,
    EmbedTextDetails,
    EmbedTextResult,
    OnDemandServingMode,
    RerankTextDetails,
)

from genai_bench.auth.model_auth_provider import ModelAuthProvider
from genai_bench.logging import init_logger
from genai_bench.protocol import (
    UserChatRequest,
    UserChatResponse,
    UserEmbeddingRequest,
    UserImageEmbeddingRequest,
    UserRequest,
    UserReRankRequest,
    UserResponse,
)
from genai_bench.user.base_user import BaseUser

logger = init_logger(__name__)


class OCICohereUser(BaseUser):
    """User class for Cohere model API with OCI authentication."""

    BACKEND_NAME = "oci-cohere"
    supported_tasks = {
        "text-to-text": "chat",
        "text-to-rerank": "rerank",
        "text-to-embeddings": "embeddings",
        "image-to-embeddings": "embeddings",
    }
    host: Optional[str] = None
    auth_provider: Optional[ModelAuthProvider] = None

    def on_start(self):
        """Initialize OCI client on start."""
        super().on_start()
        if not self.auth_provider:
            raise ValueError("Auth is required for OCICohereUser")
        self.client = GenerativeAiInferenceClient(
            config=self.auth_provider.get_config(),
            signer=self.auth_provider.get_credentials(),
            service_endpoint=self.host,
        )
        logger.debug("Generative AI Inference Client initialized.")

    def send_request(
        self,
        make_request: Callable,
        endpoint: str,
        payload: Any,
        parse_strategy: Any,
        num_prefill_tokens: Optional[int] = None,
    ) -> UserResponse:
        logger.debug(f"Sending request with payload: {payload}")
        try:
            start_time = time.monotonic()
            response = make_request()
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
                    f"RequestId: {response.request_id}",
                    f"Response: {response.response}",
                )
                metrics_response = UserResponse(
                    status_code=response.status,
                    error_message="Request Failed",
                )
            self.collect_metrics(metrics_response, endpoint)
            return metrics_response
        except Exception as e:
            return UserResponse(
                status_code=500,
                error_message=str(e),
                num_prefill_tokens=num_prefill_tokens or 0,
            )

    @task
    def chat(self):
        """Send a chat completion request using Cohere format."""
        user_request = self.sample()

        if not isinstance(user_request, UserChatRequest):
            raise AttributeError(
                f"Expected UserChatRequest for OCICohereUser.chat, got "
                f"{type(user_request)}"
            )

        compartment_id = self.get_compartment_id(user_request)
        serving_mode = self.get_serving_mode(user_request)

        # Construct chat request
        chat_request = CohereChatRequest(
            api_format="COHERE",
            message=user_request.prompt,
            max_tokens=user_request.max_tokens,
            is_stream=True,
            temperature=user_request.additional_request_params.get("temperature", 0.1),
            top_p=user_request.additional_request_params.get("topP", 0),
            top_k=user_request.additional_request_params.get("topK", 0.75),
            frequency_penalty=user_request.additional_request_params.get(
                "frequencyPenalty", 0
            ),
            presence_penalty=user_request.additional_request_params.get(
                "presencePenalty", 0
            ),
        )

        # Add chat history and documents if provided
        if "chatHistory" in user_request.additional_request_params:
            chat_request.chat_history = user_request.additional_request_params[
                "chatHistory"
            ]
            logger.info(
                f"Chat history provided with {len(chat_request.chat_history)} items."
            )

        if "documents" in user_request.additional_request_params:
            chat_request.documents = user_request.additional_request_params["documents"]
            logger.info(f"Documents provided with {len(chat_request.documents)} items.")

        # Define payload with compartment ID and serving mode
        chat_detail = ChatDetails(
            compartment_id=compartment_id,
            serving_mode=serving_mode,
            chat_request=chat_request,
        )
        return self.send_request(
            make_request=lambda: self.client.chat(chat_detail),
            endpoint="chat",
            payload=chat_detail,
            parse_strategy=self.parse_chat_response,
            num_prefill_tokens=user_request.num_prefill_tokens,
        )

    @task
    def embeddings(self):
        """Send an embedding request using Cohere format."""
        user_request = self.sample()

        if not isinstance(user_request, UserEmbeddingRequest):
            raise AttributeError(
                f"user_request should be of type UserEmbeddingRequest for "
                f"OCICohereUser.embeddings, got {type(user_request)}"
            )

        if user_request.documents and not user_request.num_prefill_tokens:
            logger.warning(
                "Number of prefill tokens is missing or 0. Please double check."
            )

        # Retrieve compartment ID and serving mode
        compartment_id = self.get_compartment_id(user_request)
        serving_mode = self.get_serving_mode(user_request)
        input_type = self.get_embedding_input_type(user_request)
        inputs = self.get_inputs(user_request)

        embed_text_detail = EmbedTextDetails(
            compartment_id=compartment_id,
            serving_mode=serving_mode,
            inputs=inputs,
            input_type=input_type,
            truncate=user_request.additional_request_params.get("truncate", "NONE"),
        )

        response = self.send_request(
            make_request=lambda: self.client.embed_text(embed_text_detail),
            endpoint="embedText",
            payload=embed_text_detail,
            parse_strategy=self.parse_embedding_response,
            num_prefill_tokens=user_request.num_prefill_tokens,
        )

        logger.debug(f"Response received {response}")

    @task
    def rerank(self):
        """Send an rerank request using Cohere format."""
        user_request = self.sample()

        if not isinstance(user_request, UserReRankRequest):
            raise AttributeError(
                f"user_request should be of type UserReRankRequest for "
                f"OCICohereUser.rerank, got {type(user_request)}"
            )

        if user_request.documents and not user_request.num_prefill_tokens:
            logger.error(
                "Number of prefill tokens is missing or 0. Please double check."
            )

        # Retrieve compartment ID and serving mode
        compartment_id = self.get_compartment_id(user_request)
        top_n = self.get_top_n(user_request)
        serving_mode = self.get_serving_mode(user_request)
        documents = self.get_documents(user_request)
        query = self.get_query(user_request)

        # TODO: Re-rank3.5 API Changes for OCI GenAI are in progress
        rerank_text_details = RerankTextDetails(
            compartment_id=compartment_id,
            serving_mode=serving_mode,
            documents=documents,
            input=query,
            top_n=top_n,
        )

        response = self.send_request(
            make_request=lambda: self.client.rerank_text(rerank_text_details),
            endpoint="rerankText",
            payload=rerank_text_details,
            parse_strategy=self.parse_rerank_response,
            num_prefill_tokens=user_request.num_prefill_tokens,
        )

        logger.debug(f"Response received {response}")

    def get_compartment_id(self, user_request: UserRequest):
        compartment_id = user_request.additional_request_params.get("compartmentId")
        if not compartment_id:
            raise ValueError("compartmentId missing in additional request params")
        return compartment_id

    def get_top_n(self, user_request: UserRequest):
        return user_request.additional_request_params.get("topN")

    def get_embedding_input_type(self, user_request) -> str:
        input_type = "SEARCH_DOCUMENT"
        if isinstance(user_request, UserImageEmbeddingRequest):
            input_type = "IMAGE"
        return input_type

    def get_documents(self, re_rank_request: UserReRankRequest) -> List[Any]:
        return re_rank_request.documents

    def get_query(self, re_rank_request: UserReRankRequest) -> str:
        return re_rank_request.query

    def get_inputs(self, user_request) -> List[Any]:
        if isinstance(user_request, UserImageEmbeddingRequest):
            num_sampled_images = len(user_request.image_content)
            if num_sampled_images > 1:
                raise ValueError(
                    f"OCI-Cohere Image embedding supports only 1 "
                    f"image but, the value provided in traffic"
                    f"scenario is requesting {num_sampled_images}"
                )
            return user_request.image_content
        return user_request.documents

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
        Parses the streaming response from the Cohere API in OCI format.

        Args:
            request (ChatDetails): OCICohere Chat request.
            response (ChatResult): The streaming response from the Cohere API.
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
                    # Extract text content if present
                    text_segment = parsed_data.get("text", "")
                    if text_segment:
                        # Capture the time at the first token
                        if not time_at_first_token:
                            time_at_first_token = time.monotonic()
                        generated_text += text_segment
                        tokens_received += 1  # each event contains one token
                        logger.debug(f"number of tokens received: {tokens_received}")
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

        return UserChatResponse(
            status_code=200,
            generated_text=generated_text,
            tokens_received=tokens_received,
            time_at_first_token=time_at_first_token,
            num_prefill_tokens=num_prefill_tokens,
            start_time=start_time,
            end_time=end_time,
        )

    def parse_embedding_response(
        self,
        request: EmbedTextDetails,
        _: EmbedTextResult,
        start_time: float,
        num_prefill_tokens: int,
        end_time: float,
    ) -> UserResponse:
        """
        Parses a non-streaming response.

        Args:
            request (EmbedTextDetails): The request object.
            _ (EmbedTextResult): The response object.
            start_time (float): The time when the request was started.
            num_prefill_tokens (int): Number of tokens in the prefill/prompt.
            end_time(float): The time when the request was finished.

        Returns:
            UserResponse: A response object with metrics.
        """
        if num_prefill_tokens is None:
            num_prefill_tokens = len(request.inputs)

        return UserResponse(
            status_code=200,
            num_prefill_tokens=num_prefill_tokens,
            start_time=start_time,
            end_time=end_time,
            time_at_first_token=end_time,
        )

    def parse_rerank_response(
        self,
        request: RerankTextDetails,
        _: RerankTextDetails,
        start_time: float,
        num_prefill_tokens: int,
        end_time: float,
    ) -> UserResponse:
        """
        Parses a non-streaming response.

        Args:
            request (RerankTextDetails): The request object.
            _ (RerankTextDetails): The response object.
            start_time (float): The time when the request was started.
            num_prefill_tokens (int): Number of tokens in the prefill/prompt.
            end_time(float): The time when the request was finished.

        Returns:
            UserResponse: A response object with metrics.
        """

        return UserResponse(
            status_code=200,
            num_prefill_tokens=num_prefill_tokens,
            start_time=start_time,
            end_time=end_time,
            time_at_first_token=end_time,
        )
