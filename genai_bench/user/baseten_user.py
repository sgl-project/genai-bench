"""Customized user for Baseten backends."""

import json
import time
from typing import Any, Callable, Dict, Optional

import requests
from requests import Response
from locust import task

from genai_bench.protocol import UserChatRequest, UserImageChatRequest, UserResponse, UserChatResponse
from genai_bench.user.openai_user import OpenAIUser
from genai_bench.logging import init_logger

logger = init_logger(__name__)


class BasetenUser(OpenAIUser):
    """Baseten user that uses the full URL as endpoint and supports both chat and prompt formats.
    
    Supports both OpenAI-compatible chat format and simple prompt format for non-instruct models.
    Format is controlled via use_prompt_format in additional_request_params.
    Streaming is controlled via the global --disable-streaming flag (consistent with other backends).
    """
    
    BACKEND_NAME = "baseten"
    disable_streaming: bool = False
    
    @task
    def chat(self):
        """Override chat method to support both OpenAI-compatible and prompt formats."""
        user_request = self.sample()

        if not isinstance(user_request, UserChatRequest):
            raise AttributeError(
                f"user_request should be of type "
                f"UserChatRequest for BasetenUser.chat, got "
                f"{type(user_request)}"
            )

        logger.info(f"🎯 Processing chat request - model: {user_request.model}, max_tokens: {user_request.max_tokens}")
        logger.info(f"🔧 Additional request params keys: {list(user_request.additional_request_params.keys())}")

        # Check if we should use prompt format
        use_prompt_format = user_request.additional_request_params.get("use_prompt_format", False)
        
        if use_prompt_format:
            logger.info("📋 Using prompt format (non-instruct model)")
            # Use simple prompt format for non-instruct models
            payload = self._prepare_prompt_request(user_request)
            endpoint = "prompt"  # Use different endpoint name for metrics
        else:
            logger.info("💬 Using OpenAI-compatible chat format")
            # Use OpenAI-compatible chat format (default)
            payload = self._prepare_chat_request(user_request)
            endpoint = "/v1/chat/completions"

        # Use global disable_streaming setting (consistent with other backends)
        use_streaming = not self.disable_streaming

        # Send request using the overridden send_request method
        if use_streaming:
            self.send_request(
                True,
                endpoint,
                payload,
                self.parse_chat_response,
                user_request.num_prefill_tokens,
            )
        else:
            self.send_request(
                False,
                endpoint,
                payload,
                self.parse_non_streaming_chat_response,
                user_request.num_prefill_tokens,
            )

    def _prepare_chat_request(self, user_request: UserChatRequest) -> Dict[str, Any]:
        """Prepare OpenAI-compatible chat request."""
        
        # Log the dataset prompt (truncate if too long)
        prompt_preview = user_request.prompt[:200] + "..." if len(user_request.prompt) > 200 else user_request.prompt
        logger.debug(f"📝 Dataset prompt (first 200 chars): {prompt_preview}")
        
        # Check if custom messages are provided in additional_request_params
        custom_messages = user_request.additional_request_params.get("custom_messages")
        
        if custom_messages:
            logger.info(f"✅ Using custom_messages from additional_request_params")
            logger.debug(f"📨 Custom messages received: {json.dumps(custom_messages, indent=2)}")
            # When custom_messages is provided, use them exactly as specified
            # This allows full control over the message structure
            if isinstance(custom_messages, list):
                messages = [msg.copy() if isinstance(msg, dict) else msg for msg in custom_messages]
            else:
                messages = custom_messages
            logger.info(f"💬 Using custom_messages as-is (dataset prompt ignored when custom_messages provided)")
        elif isinstance(user_request, UserImageChatRequest):
            text_content = [{"type": "text", "text": user_request.prompt}]
            image_content = [
                {
                    "type": "image_url",
                    "image_url": {"url": image},  # image already contains the full data URL
                }
                for image in user_request.image_content
            ]
            content = text_content + image_content
            messages = [{"role": "user", "content": content}]
            logger.debug(f"🖼️ Using image chat request with prompt")
        else:
            content = user_request.prompt
            messages = [{"role": "user", "content": content}]
            logger.debug(f"📄 Using dataset prompt as user message")

        logger.debug(f"💬 Final messages being sent: {json.dumps(messages, indent=2)}")

        # Use global disable_streaming setting (consistent with other backends)
        use_streaming = not self.disable_streaming

        # Build payload - prioritize max_tokens from additional_request_params if present
        max_tokens = user_request.additional_request_params.get("max_tokens") or user_request.max_tokens
        
        payload = {
            "model": user_request.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": user_request.additional_request_params.get(
                "temperature", 0.0
            ),
            "ignore_eos": user_request.additional_request_params.get(
                "ignore_eos",
                bool(max_tokens),
            ),
            "stream": use_streaming,
        }
        
        # Add additional params except use_prompt_format, stream, and custom_messages
        # Note: max_tokens and ignore_eos are already set above, but can be overridden here if needed
        for key, value in user_request.additional_request_params.items():
            if key not in ["use_prompt_format", "stream", "custom_messages"]:
                payload[key] = value
        
        # Only add stream_options if streaming is enabled
        if use_streaming:
            payload["stream_options"] = {
                "include_usage": True,
            }
        
        logger.info(f"📦 Payload summary - model: {payload['model']}, max_tokens: {payload['max_tokens']}, stream: {payload['stream']}, ignore_eos: {payload['ignore_eos']}")
        
        # Log full payload with messages (truncate message content if too long for readability)
        payload_for_logging = payload.copy()
        if "messages" in payload_for_logging:
            messages_copy = []
            for msg in payload_for_logging["messages"]:
                msg_copy = msg.copy()
                if "content" in msg_copy and isinstance(msg_copy["content"], str):
                    if len(msg_copy["content"]) > 1000:
                        msg_copy["content"] = msg_copy["content"][:1000] + "...[truncated, full length: " + str(len(msg_copy["content"])) + " chars]"
                messages_copy.append(msg_copy)
            payload_for_logging["messages"] = messages_copy
        logger.debug(f"📤 Full payload being sent: {json.dumps(payload_for_logging, indent=2)}")
            
        return payload

    def _prepare_prompt_request(self, user_request: UserChatRequest) -> Dict[str, Any]:
        """Prepare simple prompt request for non-instruct models."""
        # Use global disable_streaming setting (consistent with other backends)
        use_streaming = not self.disable_streaming

        payload = {
            "prompt": user_request.prompt,
            "max_tokens": user_request.max_tokens,
            "temperature": user_request.additional_request_params.get(
                "temperature", 0.0
            ),
            "stream": use_streaming,
        }
        
        # Add additional params except use_prompt_format and stream
        for key, value in user_request.additional_request_params.items():
            if key not in ["use_prompt_format", "stream"]:
                payload[key] = value
                
        return payload

    def parse_chat_response(
        self,
        response: Response,
        start_time: float,
        num_prefill_tokens: int,
        _: float,
    ) -> UserResponse:
        """
        Override parse_chat_response to handle both OpenAI format and plain text responses.
        """
        # Check if this is a prompt format request by looking at the endpoint
        # We can't easily detect this from the response, so we'll try both formats
        
        try:
            # First, try to parse as OpenAI streaming format
            return super().parse_chat_response(response, start_time, num_prefill_tokens, _)
        except (json.JSONDecodeError, KeyError, IndexError, AttributeError) as e:
            # If OpenAI format fails, try to parse as plain text streaming
            logger.debug(f"OpenAI format parsing failed, trying plain text: {e}")
            return self._parse_plain_text_streaming_response(
                response, start_time, num_prefill_tokens, _
            )

    def parse_non_streaming_chat_response(
        self,
        response: Response,
        start_time: float,
        num_prefill_tokens: int,
        _: float,
    ) -> UserResponse:
        """
        Override parse_non_streaming_chat_response to handle both OpenAI format and plain text responses.
        """
        # First, try to determine if this is JSON or plain text
        response_text = response.text.strip()
        
        try:
            # Try to parse as JSON
            data = json.loads(response_text)
            # If successful, try to parse as OpenAI format
            return super().parse_non_streaming_chat_response(
                response, start_time, num_prefill_tokens, _
            )
        except (json.JSONDecodeError, AttributeError) as e:
            # If JSON parsing fails, treat as plain text
            logger.debug(f"Response is not JSON, treating as plain text: {e}")
            return self._parse_plain_text_response(
                response, start_time, num_prefill_tokens, _
            )

    def _parse_plain_text_streaming_response(
        self,
        response: Response,
        start_time: float,
        num_prefill_tokens: int,
        _: float,
    ) -> UserResponse:
        """
        Parse plain text streaming response for prompt format.
        """
        generated_text = ""
        time_at_first_token = None
        end_time = None

        try:
            for chunk in response.iter_lines(chunk_size=None):
                chunk = chunk.strip()
                if not chunk:
                    continue
                
                # Try to decode as text
                try:
                    chunk_text = chunk.decode('utf-8')
                except UnicodeDecodeError:
                    continue
                
                # Set first token time
                if not time_at_first_token:
                    time_at_first_token = time.monotonic()
                
                generated_text += chunk_text
                end_time = time.monotonic()
                
        except Exception as e:
            logger.error(f"Error parsing plain text streaming response: {e}")
            return UserResponse(
                status_code=500,
                error_message=f"Failed to parse plain text streaming response: {e}",
            )

        if not end_time:
            end_time = time.monotonic()

        # Estimate tokens received
        tokens_received = self.environment.sampler.get_token_length(
            generated_text, add_special_tokens=False
        )

        return UserChatResponse(
            status_code=200,
            generated_text=generated_text,
            tokens_received=tokens_received,
            time_at_first_token=time_at_first_token or start_time,
            num_prefill_tokens=num_prefill_tokens,
            start_time=start_time,
            end_time=end_time,
        )

    def _parse_plain_text_response(
        self,
        response: Response,
        start_time: float,
        num_prefill_tokens: int,
        _: float,
    ) -> UserResponse:
        """
        Parse plain text non-streaming response for prompt format.
        """
        try:
            # Try to get the response as text
            response_text = response.text.strip()
            end_time = time.monotonic()
            
            # Try to parse as JSON first (in case it's actually JSON)
            try:
                data = json.loads(response_text)
                # If it's JSON, try to extract text from common fields
                if isinstance(data, dict):
                    generated_text = (
                        data.get("text") or 
                        data.get("output") or 
                        data.get("response") or 
                        data.get("generated_text") or
                        response_text  # fallback to full response
                    )
                else:
                    generated_text = response_text
            except json.JSONDecodeError:
                # Not JSON, treat as plain text
                generated_text = response_text
            
            # Estimate tokens received
            tokens_received = self.environment.sampler.get_token_length(
                generated_text, add_special_tokens=False
            )
            
            # For non-streaming, we can't measure TTFT, so we use a small offset
            time_at_first_token = start_time + 0.001  # 1ms offset
            
            logger.debug(
                f"Plain text response: {generated_text}\n"
                f"Estimated tokens: {tokens_received}\n"
                f"Start time: {start_time}\n"
                f"End time: {end_time}"
            )
            
            return UserChatResponse(
                status_code=200,
                generated_text=generated_text,
                tokens_received=tokens_received,
                time_at_first_token=time_at_first_token,
                num_prefill_tokens=num_prefill_tokens,
                start_time=start_time,
                end_time=end_time,
            )
            
        except Exception as e:
            logger.error(f"Error parsing plain text response: {e}")
            return UserResponse(
                status_code=500,
                error_message=f"Failed to parse plain text response: {e}",
            )
    
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
            logger.debug(f"🌐 Sending request to: {self.host}")
            # For Baseten, use the host directly as the URL
            response = requests.post(
                url=self.host,  # Use host directly instead of host + endpoint
                json=payload,
                stream=stream,
                headers=self.headers,
            )
            non_stream_post_end_time = time.monotonic()
            
            logger.info(f"📡 Response status code: {response.status_code}")

            if response.status_code == 200:
                metrics_response = parse_strategy(
                    response,
                    start_time,
                    num_prefill_tokens,
                    non_stream_post_end_time,
                )
                # Log response summary
                if hasattr(metrics_response, 'generated_text'):
                    response_text = metrics_response.generated_text
                    preview = response_text[:500] + "..." if len(response_text) > 500 else response_text
                    logger.info(f"📤 Response preview (first 500 chars): {preview}")
                    logger.info(f"📊 Response metrics - tokens_received: {getattr(metrics_response, 'tokens_received', 'N/A')}, "
                               f"status_code: {metrics_response.status_code}")
            else:
                logger.error(f"❌ Request failed with status {response.status_code}: {response.text[:500]}")
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