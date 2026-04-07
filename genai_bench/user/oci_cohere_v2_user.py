from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Tuple

from oci.generative_ai_inference.models import (
    ChatDetails,
    ChatResult,
    CohereChatRequestV2,
    CohereImageContentV2,
    CohereImageUrlV2,
    CohereMessageV2,
    CohereSystemMessageV2,
    CohereTextContentV2,
    CohereThinkingContentV2,
    CohereUserMessageV2,
    StreamOptions,
)

from genai_bench.protocol import UserChatRequest, UserChatResponse, UserImageChatRequest
from genai_bench.user.oci_cohere_user import OCICohereUser, logger


class OCICohereV2User(OCICohereUser):
    """Cohere Command-A (V2) user leveraging OCI authentication."""

    BACKEND_NAME = "oci-cohere-v2"
    supported_tasks = {
        "text-to-text": "chat",
        "image-text-to-text": "chat",
    }

    def _build_chat_request(self, user_request: UserChatRequest) -> CohereChatRequestV2:
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

    def _parse_chat_response(
        self,
        request: ChatDetails,
        response: ChatResult,
        start_time: float,
        num_prefill_tokens: Optional[int],
        _: float,
    ) -> UserChatResponse:
        generated_text = ""
        reasoning_text = ""
        time_at_first_token: Optional[float] = None
        usage_payload: Optional[Dict[str, Any]] = None
        previous_data: Optional[Any] = None
        finish_reason: Optional[str] = None

        for event in response.data.events():
            raw_data = event.data
            if isinstance(raw_data, bytes):
                raw_data = raw_data.decode("utf-8", errors="ignore")
            if not isinstance(raw_data, str):
                continue
            event_data = raw_data.strip()
            if not event_data:
                continue

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

            message = parsed_data.get("message", {}) or {}
            text_chunk, reasoning_chunk = self._extract_text_from_message(message)
            if (text_chunk or reasoning_chunk) and time_at_first_token is None:
                time_at_first_token = time.monotonic()
            generated_text += text_chunk
            reasoning_text += reasoning_chunk

            finish_reason = finish_reason or parsed_data.get("finishReason")
            if finish_reason:
                usage_payload = parsed_data.get("usage")
                break

        end_time = time.monotonic()
        combined_text = generated_text + reasoning_text

        tokens_received = 0
        reasoning_tokens = 0
        usage_prompt_tokens: Optional[int] = None
        usage_completion_tokens: Optional[int] = None
        usage_reasoning_tokens: Optional[int] = None

        if isinstance(usage_payload, dict):
            (
                usage_prompt_tokens,
                usage_completion_tokens,
                usage_reasoning_tokens,
            ) = self._extract_usage_tokens(usage_payload)

        if usage_completion_tokens is not None:
            tokens_received = usage_completion_tokens

        if usage_reasoning_tokens is not None:
            reasoning_tokens = usage_reasoning_tokens

        if usage_prompt_tokens is not None:
            num_prefill_tokens = usage_prompt_tokens

        if num_prefill_tokens is None:
            prompt_text = self._gather_prompt_text(request)
            if prompt_text:
                num_prefill_tokens = self.environment.sampler.get_token_length(
                    prompt_text, add_special_tokens=False
                )
                logger.warning(
                    "OCI Cohere V2 response omitted prompt token usage; "
                    "estimated prefill tokens from textual prompt."
                )
            else:
                num_prefill_tokens = 0
                logger.warning(
                    "OCI Cohere V2 response omitted prompt token usage and prompt text "
                    "could not be inferred (likely vision-only); "
                    "prefill tokens default to 0."
                )

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
            num_prefill_tokens=num_prefill_tokens or 0,
            start_time=start_time,
            end_time=end_time,
        )

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

        chat_history = params.pop("chat_history", params.pop("chatHistory", None))
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

    def _extract_usage_tokens(
        self, usage_payload: Dict[str, Any]
    ) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        prompt_tokens: Optional[int] = None
        completion_tokens: Optional[int] = None
        reasoning_tokens: Optional[int] = None

        if not isinstance(usage_payload, dict):
            return prompt_tokens, completion_tokens, reasoning_tokens

        tokens_block = usage_payload.get("tokens")
        if isinstance(tokens_block, dict):
            prompt_tokens = (
                tokens_block.get("input_tokens")
                or tokens_block.get("prompt_tokens")
                or tokens_block.get("inputTokens")
                or tokens_block.get("promptTokens")
            )
            completion_tokens = (
                tokens_block.get("output_tokens")
                or tokens_block.get("completion_tokens")
                or tokens_block.get("outputTokens")
                or tokens_block.get("completionTokens")
            )
            reasoning_tokens = tokens_block.get("reasoning_tokens") or tokens_block.get(
                "reasoningTokens"
            )

        prompt_tokens = (
            prompt_tokens
            if prompt_tokens is not None
            else usage_payload.get("promptTokens") or usage_payload.get("prompt_tokens")
        )

        completion_tokens = (
            completion_tokens
            if completion_tokens is not None
            else usage_payload.get("completionTokens")
            or usage_payload.get("completion_tokens")
        )

        reasoning_tokens = (
            reasoning_tokens
            if reasoning_tokens is not None
            else usage_payload.get("reasoningTokens")
            or usage_payload.get("reasoning_tokens")
        )

        if reasoning_tokens is None:
            completion_details = usage_payload.get(
                "completionTokensDetails"
            ) or usage_payload.get("completion_tokens_details")
            if isinstance(completion_details, dict):
                reasoning_tokens = completion_details.get(
                    "reasoningTokens"
                ) or completion_details.get("reasoning_tokens")

        return prompt_tokens, completion_tokens, reasoning_tokens

    def _gather_prompt_text(self, request: ChatDetails) -> str:
        chat_request = getattr(request, "chat_request", None)
        if not chat_request:
            return ""
        messages = getattr(chat_request, "messages", None)
        if not messages:
            return ""

        text_segments: List[str] = []
        for message in messages:
            content_list = getattr(message, "content", None)
            if not content_list:
                continue
            for item in content_list:
                text_value = getattr(item, "text", None)
                if text_value:
                    text_segments.append(str(text_value))
        return "".join(text_segments)

    def _extract_text_from_message(self, message: Dict[str, Any]) -> tuple[str, str]:
        if not isinstance(message, dict):
            return "", ""

        text_parts: List[str] = []
        reasoning_parts: List[str] = []

        for block in message.get("content", []):
            if isinstance(block, str):
                text_parts.append(block)
                continue
            if not isinstance(block, dict):
                continue

            block_type = str(block.get("type", "TEXT")).upper()
            if block_type in {"TEXT", "OUTPUT_TEXT"}:
                value = block.get("text") or block.get("data")
                if value:
                    text_parts.append(str(value))
            elif block_type == "THINKING":
                value = block.get("thinking") or block.get("text")
                if value:
                    reasoning_parts.append(str(value))

        return "".join(text_parts), "".join(reasoning_parts)
