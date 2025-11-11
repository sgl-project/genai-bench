//! OpenAI provider implementation

use super::{ChatRequest, ChatResponse, Provider};
use crate::metrics::RequestMetrics;
use anyhow::{Context, Result};
use async_trait::async_trait;
use futures::StreamExt;
use reqwest::Client;
use serde::Deserialize;
use std::time::Instant;

/// OpenAI API provider
#[derive(Debug, Clone)]
pub struct OpenAIProvider {
    client: Client,
    api_key: String,
    base_url: String,
}

/// Streaming chunk from OpenAI
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct StreamChunk {
    id: String,
    choices: Vec<StreamChoice>,
    #[serde(default)]
    usage: Option<super::Usage>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct StreamChoice {
    index: u32,
    delta: Delta,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct Delta {
    #[serde(default)]
    role: Option<String>,
    #[serde(default)]
    content: Option<String>,
}

impl OpenAIProvider {
    /// Create a new OpenAI provider
    pub fn new(api_key: String, base_url: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
            base_url,
        }
    }

    /// Get the chat completions endpoint URL
    fn chat_url(&self) -> String {
        format!("{}/chat/completions", self.base_url)
    }

    /// Parse SSE (Server-Sent Events) line
    fn parse_sse_line(line: &str) -> Option<String> {
        if line.starts_with("data: ") {
            let data = line.strip_prefix("data: ")?;
            if data == "[DONE]" {
                None
            } else {
                Some(data.to_string())
            }
        } else {
            None
        }
    }

    /// Handle streaming response
    async fn handle_streaming(&self, request: &ChatRequest) -> Result<RequestMetrics> {
        let start = Instant::now();
        let mut ttft_ms = None;
        let mut completion_tokens = 0u32;
        let mut total_content = String::new();

        // Send streaming request
        let response = self
            .client
            .post(&self.chat_url())
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(request)
            .send()
            .await
            .context("Failed to send streaming request to OpenAI")?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("OpenAI API error ({}): {}", status, error_text);
        }

        // Process stream
        let mut stream = response.bytes_stream();
        let mut buffer = String::new();

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result.context("Failed to read stream chunk")?;
            buffer.push_str(&String::from_utf8_lossy(&chunk));

            // Process complete lines
            while let Some(newline_pos) = buffer.find('\n') {
                let line = buffer[..newline_pos].trim().to_string();
                buffer = buffer[newline_pos + 1..].to_string();

                if line.is_empty() {
                    continue;
                }

                // Parse SSE line
                if let Some(data) = Self::parse_sse_line(&line) {
                    // Parse JSON chunk
                    match serde_json::from_str::<StreamChunk>(&data) {
                        Ok(stream_chunk) => {
                            // Record TTFT on first chunk with content
                            if ttft_ms.is_none() {
                                for choice in &stream_chunk.choices {
                                    if choice.delta.content.is_some() {
                                        ttft_ms = Some(start.elapsed().as_millis() as u64);
                                        break;
                                    }
                                }
                            }

                            // Collect content
                            for choice in stream_chunk.choices {
                                if let Some(content) = choice.delta.content {
                                    total_content.push_str(&content);
                                    // OpenAI sends one token per chunk (1:1 mapping)
                                    completion_tokens += 1;
                                }
                            }

                            // Check for usage information (sent in last chunk)
                            if let Some(usage) = stream_chunk.usage {
                                completion_tokens = usage.completion_tokens;
                            }
                        }
                        Err(e) => {
                            tracing::debug!("Failed to parse stream chunk: {}, data: {}", e, data);
                        }
                    }
                }
            }
        }

        let total_time_ms = start.elapsed().as_millis() as u64;

        // Estimate prompt tokens (rough heuristic: ~4 chars per token)
        // TODO: Use tiktoken library for exact counts
        let prompt_chars: usize = request
            .messages
            .iter()
            .map(|m| m.content.len())
            .sum();
        let prompt_tokens = (prompt_chars / 4).max(1) as u32;

        Ok(RequestMetrics {
            ttft_ms: ttft_ms.unwrap_or(total_time_ms),
            total_time_ms,
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            success: true,
            status_code: status.as_u16(),
            error_message: None,
        })
    }
}

#[async_trait]
impl Provider for OpenAIProvider {
    async fn chat(&self, request: &ChatRequest) -> Result<RequestMetrics> {
        // Route to streaming or non-streaming based on request
        if request.stream.unwrap_or(false) {
            self.handle_streaming(request).await
        } else {
            self.handle_non_streaming(request).await
        }
    }

    fn name(&self) -> &str {
        "openai"
    }
}

impl OpenAIProvider {
    /// Handle non-streaming response
    async fn handle_non_streaming(&self, request: &ChatRequest) -> Result<RequestMetrics> {
        let start = Instant::now();

        // Send the request
        let response = self
            .client
            .post(&self.chat_url())
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(request)
            .send()
            .await
            .context("Failed to send request to OpenAI")?;

        let status = response.status();

        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("OpenAI API error ({}): {}", status, error_text);
        }

        let ttft = start.elapsed(); // For non-streaming, this is the full response time

        // Parse the response
        let response_body: ChatResponse = response
            .json()
            .await
            .context("Failed to parse OpenAI response")?;

        let total_time = start.elapsed();

        // Extract metrics
        let prompt_tokens = response_body
            .usage
            .as_ref()
            .map(|u| u.prompt_tokens)
            .unwrap_or(0);
        let completion_tokens = response_body
            .usage
            .as_ref()
            .map(|u| u.completion_tokens)
            .unwrap_or(0);
        let total_tokens = response_body
            .usage
            .as_ref()
            .map(|u| u.total_tokens)
            .unwrap_or(0);

        Ok(RequestMetrics {
            ttft_ms: ttft.as_millis() as u64,
            total_time_ms: total_time.as_millis() as u64,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            success: true,
            status_code: status.as_u16(),
            error_message: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openai_provider_creation() {
        let provider = OpenAIProvider::new(
            "test-key".to_string(),
            "https://api.openai.com/v1".to_string(),
        );
        assert_eq!(provider.name(), "openai");
    }

    #[test]
    fn test_chat_url() {
        let provider = OpenAIProvider::new(
            "test-key".to_string(),
            "https://api.openai.com/v1".to_string(),
        );
        assert_eq!(
            provider.chat_url(),
            "https://api.openai.com/v1/chat/completions"
        );
    }
}
