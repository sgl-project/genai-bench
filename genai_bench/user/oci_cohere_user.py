from locust import task

import json
import time
from typing import Any, Callable, Dict, List, Optional

from oci.generative_ai_inference import GenerativeAiInferenceClient
from oci.generative_ai_inference.models import (
    ChatDetails,
    ChatResult,
    CohereChatRequest,
    CohereChatRequestV2,
    CohereImageContentV2,
    CohereImageUrlV2,
    CohereMessageV2,
    CohereSystemMessageV2,
    CohereTextContentV2,
    CohereThinkingContentV2,
    CohereUserMessageV2,
    DedicatedServingMode,
    EmbedTextDetails,
    EmbedTextResult,
    OnDemandServingMode,
    RerankTextDetails,
    StreamOptions,
)

from genai_bench.auth.model_auth_provider import ModelAuthProvider
from genai_bench.logging import init_logger
from genai_bench.protocol import (
    UserChatRequest,
    UserChatResponse,
    UserEmbeddingRequest,
    UserImageChatRequest,
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
        "image-text-to-text": "chat",
        "image-to-embeddings": "embeddings",
    }
    api_version: str = "v1"
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
            metrics_response = UserResponse(
                status_code=getattr(e, "status", 500),
                error_message=str(e),
                num_prefill_tokens=num_prefill_tokens or 0,
            )
            self.collect_metrics(metrics_response, endpoint)
            return metrics_response

    @task
    def chat(self):
        """Send a chat completion request using Cohere format."""
        user_request = self.sample()

        if not isinstance(user_request, UserChatRequest):
            raise AttributeError(
                f"Expected UserChatRequest for OCICohereUser.chat, got "
                f"{type(user_request)}"
            )

        request_builder, parse_strategy = self._select_chat_strategy(user_request)

        compartment_id = self.get_compartment_id(user_request)
        serving_mode = self.get_serving_mode(user_request)
        chat_request = request_builder(user_request)

        chat_detail = ChatDetails(
            compartment_id=compartment_id,
            serving_mode=serving_mode,
            chat_request=chat_request,
        )
        return self.send_request(
            make_request=lambda: self.client.chat(chat_detail),
            endpoint="chat",
            payload=chat_detail,
            parse_strategy=parse_strategy,
            num_prefill_tokens=user_request.num_prefill_tokens,
        )

    def is_v2_enabled(self) -> bool:
        return getattr(self, "api_version", "v1").lower() == "v2"

    def _select_chat_strategy(
        self, user_request: UserChatRequest
    ) -> tuple[Callable[[UserChatRequest], Any], Callable[..., UserResponse]]:
        if self.is_v2_enabled():
            logger.debug("Using Cohere V2 chat flow for OCI Cohere user.")
            return self.build_v2_chat_request, self.parse_chat_response_v2
        logger.debug("Using Cohere V1 chat flow for OCI Cohere user.")
        return self._build_v1_chat_request, self.parse_chat_response

    def _build_v1_chat_request(
        self, user_request: UserChatRequest
    ) -> CohereChatRequest:
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

        return chat_request

    def build_v2_chat_request(
        self, user_request: UserChatRequest
    ) -> CohereChatRequestV2:
        params: Dict[str, Any] = dict(user_request.additional_request_params or {})
        messages = self._build_v2_messages(user_request, params)

        request_kwargs: Dict[str, Any] = {
            "messages": messages,
            "max_tokens": user_request.max_tokens,
            "is_stream": True,
            "stream_options": StreamOptions(is_include_usage=True),
        }
        allowed_params = {
            "temperature",
            "top_p",
            "top_k",
            "presence_penalty",
            "frequency_penalty",
            "stop_sequences",
            "seed",
            "priority",
            "safety_mode",
            "is_search_queries_only",
            "is_log_probs_enabled",
            "is_strict_tools_enabled",
            "is_raw_prompting",
            "thinking",
            "documents",
            "tools",
            "tools_choice",
            "response_format",
        }
        for key in allowed_params:
            if key in params:
                request_kwargs[key] = params[key]
        # Remove None values to avoid overriding server defaults
        request_kwargs = {k: v for k, v in request_kwargs.items() if v is not None}

        logger.debug("Prepared CohereChatRequestV2 payload: %s", request_kwargs.keys())
        return CohereChatRequestV2(**request_kwargs)

    def _build_v2_messages(
        self, user_request: UserChatRequest, params: Dict[str, Any]
    ) -> List[CohereMessageV2]:
        messages: List[CohereMessageV2] = []

        system_message = params.pop("system_message", None)
        if system_message:
            messages.append(
                CohereSystemMessageV2(
                    content=[CohereTextContentV2(text=str(system_message))]
                )
            )

        chat_history = params.pop("chat_history", None)
        if chat_history:
            logger.info(
                "Chat history provided with %s items for Cohere V2 request.",
                len(chat_history),
            )
            for history_item in chat_history:
                converted = self._convert_history_message(history_item)
                if converted:
                    messages.append(converted)

        messages.append(self._build_v2_user_message(user_request))
        return messages

    def _build_v2_user_message(
        self, user_request: UserChatRequest
    ) -> CohereUserMessageV2:
        contents: List[Any] = []
        prompt = getattr(user_request, "prompt", "") or ""
        contents.append(CohereTextContentV2(text=prompt))

        if isinstance(user_request, UserImageChatRequest):
            images = user_request.image_content or []
            if len(images) > 1:
                raise ValueError(
                    "OCI Cohere vision chat currently supports only a single image "
                    "per request."
                )
            if images:
                contents.append(
                    CohereImageContentV2(image_url=CohereImageUrlV2(url=str(images[0])))
                )
        return CohereUserMessageV2(content=contents)

    def _convert_history_message(self, history_item: Any) -> Optional[CohereMessageV2]:
        if not isinstance(history_item, dict):
            logger.warning("Skipping invalid chat history item: %s", history_item)
            return None

        role = str(history_item.get("role", "")).upper()
        content = history_item.get("content")
        contents = self._convert_to_v2_contents(content)
        if not contents:
            logger.warning("Chat history entry missing content: %s", history_item)
            return None

        if role == "SYSTEM":
            return CohereSystemMessageV2(content=contents)
        if role == "USER":
            return CohereUserMessageV2(content=contents)
        if role == "ASSISTANT":
            return CohereMessageV2(role="ASSISTANT", content=contents)
        if role == "TOOL":
            return CohereMessageV2(role="TOOL", content=contents)

        return CohereMessageV2(role=role or "ASSISTANT", content=contents)

    def _convert_to_v2_contents(self, content: Any) -> List[Any]:
        if content is None:
            return []
        if isinstance(content, str):
            return [CohereTextContentV2(text=content)]
        if isinstance(content, list):
            converted: List[Any] = []
            for item in content:
                converted.extend(self._convert_content_item(item))
            return converted
        if isinstance(content, dict):
            return self._convert_content_item(content)
        logger.warning("Unsupported chat history content format: %s", content)
        return []

    def _convert_content_item(self, item: Any) -> List[Any]:
        if isinstance(item, str):
            return [CohereTextContentV2(text=item)]
        if not isinstance(item, dict):
            return []

        content_type = str(item.get("type", "text")).upper()

        if content_type in {"TEXT", "OUTPUT_TEXT", "OUTPUT_TEXT_DELTA"}:
            text_value = item.get("text") or item.get("data")
            if text_value:
                return [CohereTextContentV2(text=str(text_value))]
        if content_type == "THINKING":
            thinking_value = item.get("thinking") or item.get("text")
            if thinking_value:
                return [CohereThinkingContentV2(thinking=str(thinking_value))]
        if content_type == "IMAGE_URL":
            url = item.get("image_url") or item.get("url")
            if isinstance(url, dict):
                url = url.get("url")
            if url:
                return [CohereImageContentV2(image_url=CohereImageUrlV2(url=str(url)))]
        # Fallback to text if text key present without explicit type
        if "text" in item:
            return [CohereTextContentV2(text=str(item["text"]))]
        return []

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
            reasoning_tokens=0,  # OCI Cohere v1 API does not support reasoning
            time_at_first_token=time_at_first_token,
            num_prefill_tokens=num_prefill_tokens,
            start_time=start_time,
            end_time=end_time,
        )

    def parse_chat_response_v2(
        self,
        request: ChatDetails,
        response: ChatResult,
        start_time: float,
        num_prefill_tokens: Optional[int],
        _: float,
    ) -> UserResponse:
        generated_parts: List[str] = []
        reasoning_parts: List[str] = []
        time_at_first_token: Optional[float] = None
        usage_payload: Optional[Dict[str, Any]] = None
        previous_data: Optional[Any] = None

        for event in response.data.events():
            raw_data = event.data
            if isinstance(raw_data, bytes):
                raw_data = raw_data.decode("utf-8", errors="ignore")
            if not isinstance(raw_data, str):
                continue
            event_data = raw_data.strip()
            if not event_data:
                continue
            if event_data.startswith("data:"):
                event_data = event_data[len("data:") :].strip()
            if event_data == "[DONE]":
                break

            try:
                parsed_data = json.loads(event_data)
            except json.JSONDecodeError:
                logger.warning(
                    "Error decoding Cohere V2 stream event: %s, previous=%s",
                    event_data,
                    previous_data,
                )
                previous_data = event_data
                continue

            previous_data = parsed_data
            event_type = parsed_data.get("type")
            delta = parsed_data.get("delta")

            if isinstance(delta, dict):
                text_chunk, reasoning_chunk = self._extract_text_from_delta(delta)
                if text_chunk:
                    generated_parts.append(text_chunk)
                if reasoning_chunk:
                    reasoning_parts.append(reasoning_chunk)
                if (text_chunk or reasoning_chunk) and time_at_first_token is None:
                    time_at_first_token = time.monotonic()
                if event_type == "message-end":
                    usage_payload = delta.get("usage")
                    break
            elif event_type == "message-end":
                usage_payload = parsed_data.get("usage")
                message = parsed_data.get("message", {})
                text_chunk, reasoning_chunk = self._extract_text_from_delta(message)
                if text_chunk:
                    generated_parts.append(text_chunk)
                if reasoning_chunk:
                    reasoning_parts.append(reasoning_chunk)
                if (text_chunk or reasoning_chunk) and time_at_first_token is None:
                    time_at_first_token = time.monotonic()
                break

        end_time = time.monotonic()
        generated_text = "".join(generated_parts)
        reasoning_text = "".join(reasoning_parts)
        combined_text = generated_text + reasoning_text

        tokens_received = 0
        reasoning_tokens = 0

        if isinstance(usage_payload, dict):
            tokens_block = usage_payload.get("tokens", {})
            if isinstance(tokens_block, dict):
                tokens_received = tokens_block.get("output_tokens") or 0
                reasoning_tokens = tokens_block.get("reasoning_tokens") or 0
                prompt_tokens = tokens_block.get("input_tokens")
            else:
                tokens_received = usage_payload.get("completion_tokens") or 0
                reasoning_tokens = usage_payload.get("reasoning_tokens") or 0
                prompt_tokens = usage_payload.get("prompt_tokens")
            if prompt_tokens is not None:
                num_prefill_tokens = prompt_tokens

        if not tokens_received and combined_text:
            tokens_received = self.environment.sampler.get_token_length(
                combined_text, add_special_tokens=False
            )

        if not reasoning_tokens and reasoning_text:
            reasoning_tokens = self.environment.sampler.get_token_length(
                reasoning_text, add_special_tokens=False
            )

        if time_at_first_token is None:
            time_at_first_token = end_time

        return UserChatResponse(
            status_code=200,
            generated_text=combined_text,
            tokens_received=tokens_received,
            reasoning_tokens=reasoning_tokens,
            time_at_first_token=time_at_first_token,
            num_prefill_tokens=num_prefill_tokens,
            start_time=start_time,
            end_time=end_time,
        )

    def _extract_text_from_delta(self, delta: Dict[str, Any]) -> tuple[str, str]:
        if not isinstance(delta, dict):
            return "", ""

        text_parts: List[str] = []
        reasoning_parts: List[str] = []

        content_blocks: List[Any] = []
        if isinstance(delta.get("content"), list):
            content_blocks.extend(delta["content"])

        message = delta.get("message")
        if isinstance(message, dict):
            if isinstance(message.get("content"), list):
                content_blocks.extend(message["content"])
            if isinstance(message.get("reasoningContent"), list):
                content_blocks.extend(message["reasoningContent"])

        for block in content_blocks:
            if isinstance(block, str):
                text_parts.append(block)
                continue
            if not isinstance(block, dict):
                continue
            block_type = str(block.get("type", "")).lower()
            if block_type in {"text", "output_text", "output_text_delta"}:
                value = block.get("text") or block.get("data")
                if value:
                    text_parts.append(str(value))
            elif block_type == "thinking":
                value = block.get("thinking") or block.get("text")
                if value:
                    reasoning_parts.append(str(value))

        thinking_payload = delta.get("thinking")
        if isinstance(thinking_payload, str):
            reasoning_parts.append(thinking_payload)
        elif isinstance(thinking_payload, dict):
            value = thinking_payload.get("text") or thinking_payload.get("thinking")
            if value:
                reasoning_parts.append(str(value))

        return "".join(text_parts), "".join(reasoning_parts)

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
