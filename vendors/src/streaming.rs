//! Streaming response processing for LLM vendor APIs.
//!
//! This module provides parsers and processors for different streaming formats
//! used by LLM vendors (SSE, JSON Lines, AWS Event Stream).

use std::sync::Arc;
use std::time::{Duration, Instant};

use genai_bench_core::{FinishReason, StreamChunk, Usage};
use reqwest::Client;

use crate::traits::StreamFormat;

// ============================================================================
// SSE Parser
// ============================================================================

/// Maximum buffer size (1MB) to prevent unbounded memory growth from malformed streams.
const MAX_BUFFER_SIZE: usize = 1024 * 1024;

/// Server-Sent Events (SSE) parser.
///
/// Handles the SSE protocol as used by OpenAI, Azure, and other vendors.
/// Buffers incoming bytes and extracts complete events.
///
/// # Buffer Limits
///
/// The internal buffer is limited to 1MB to prevent unbounded memory growth
/// from malformed streams. If exceeded, the buffer is truncated and a warning
/// is logged.
///
/// # SSE Format
///
/// ```text
/// event: message
/// data: {"content": "Hello"}
///
/// data: {"content": " world"}
///
/// data: [DONE]
/// ```
#[derive(Debug, Default)]
pub struct SSEParser {
    /// Internal buffer for incomplete events
    buffer: String,
    /// Whether the buffer has been truncated due to size limits
    truncated: bool,
}

impl SSEParser {
    /// Create a new SSE parser.
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
            truncated: false,
        }
    }

    /// Feed bytes into the parser and return any complete events.
    ///
    /// Incomplete events are buffered for the next call.
    /// If the buffer exceeds the maximum size (1MB), it will be truncated
    /// and a warning will be logged.
    pub fn feed(&mut self, bytes: &[u8]) -> Vec<SSEEvent> {
        // Append new bytes to buffer (lossy UTF-8 conversion)
        self.buffer.push_str(&String::from_utf8_lossy(bytes));

        // Prevent unbounded buffer growth
        if self.buffer.len() > MAX_BUFFER_SIZE {
            if !self.truncated {
                tracing::warn!(
                    "SSE buffer exceeded {} bytes, truncating. This may indicate a malformed stream.",
                    MAX_BUFFER_SIZE
                );
                self.truncated = true;
            }
            // Truncate at a newline boundary to avoid splitting events
            let target_start = self.buffer.len() - MAX_BUFFER_SIZE / 2;
            let start = self.buffer[target_start..]
                .find('\n')
                .map(|pos| target_start + pos + 1)
                .unwrap_or(target_start);
            self.buffer = self.buffer[start..].to_string();
        }

        let mut events = Vec::new();

        // Events are separated by double newlines
        while let Some(pos) = self.buffer.find("\n\n") {
            let event_str: String = self.buffer.drain(..pos + 2).collect();
            if let Some(event) = self.parse_event(&event_str) {
                events.push(event);
            }
            // Reset truncation flag after successfully parsing an event
            self.truncated = false;
        }

        events
    }

    /// Parse a single SSE event from its text representation.
    fn parse_event(&self, raw: &str) -> Option<SSEEvent> {
        let mut data_lines = Vec::new();
        let mut event_type = None;
        let mut id = None;
        let mut retry = None;

        for line in raw.lines() {
            if line.is_empty() {
                continue;
            }

            if let Some(value) = line.strip_prefix("data: ") {
                data_lines.push(value);
            } else if let Some(value) = line.strip_prefix("data:") {
                // Handle "data:" without space after colon
                data_lines.push(value);
            } else if let Some(value) = line.strip_prefix("event: ") {
                event_type = Some(value.to_string());
            } else if let Some(value) = line.strip_prefix("event:") {
                event_type = Some(value.to_string());
            } else if let Some(value) = line.strip_prefix("id: ") {
                id = Some(value.to_string());
            } else if let Some(value) = line.strip_prefix("id:") {
                id = Some(value.to_string());
            } else if let Some(value) = line.strip_prefix("retry: ") {
                retry = value.parse().ok();
            } else if let Some(value) = line.strip_prefix("retry:") {
                retry = value.parse().ok();
            }
        }

        if data_lines.is_empty() {
            return None;
        }

        // Join multiple data lines with newlines (per SSE spec)
        let data = data_lines.join("\n");

        // Check for [DONE] marker (OpenAI convention)
        if data.trim() == "[DONE]" {
            return Some(SSEEvent::Done);
        }

        Some(SSEEvent::Data {
            data,
            event_type,
            id,
            retry,
        })
    }

    /// Clear the internal buffer.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.truncated = false;
    }

    /// Check if the parser has buffered data.
    pub fn has_buffered_data(&self) -> bool {
        !self.buffer.is_empty()
    }

    /// Get the current buffer contents (for debugging).
    #[cfg(test)]
    pub fn buffer(&self) -> &str {
        &self.buffer
    }
}

/// A single Server-Sent Event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SSEEvent {
    /// Data event containing JSON payload.
    Data {
        /// The event data (usually JSON)
        data: String,
        /// Optional event type
        event_type: Option<String>,
        /// Optional event ID
        id: Option<String>,
        /// Optional retry timeout in milliseconds
        retry: Option<u64>,
    },
    /// End of stream marker (`[DONE]`)
    Done,
}

impl SSEEvent {
    /// Returns true if this is a Done event.
    pub fn is_done(&self) -> bool {
        matches!(self, SSEEvent::Done)
    }

    /// Returns the data if this is a Data event.
    pub fn data(&self) -> Option<&str> {
        match self {
            SSEEvent::Data { data, .. } => Some(data),
            SSEEvent::Done => None,
        }
    }

    /// Returns the event type if this is a Data event.
    pub fn event_type(&self) -> Option<&str> {
        match self {
            SSEEvent::Data { event_type, .. } => event_type.as_deref(),
            SSEEvent::Done => None,
        }
    }
}

// ============================================================================
// JSON Lines Parser
// ============================================================================

/// JSON Lines (newline-delimited JSON) parser.
///
/// Some vendors use JSON Lines format instead of SSE.
#[derive(Debug, Default)]
pub struct JsonLinesParser {
    buffer: String,
    /// Whether the buffer has been truncated due to size limits
    truncated: bool,
}

impl JsonLinesParser {
    /// Create a new JSON Lines parser.
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
            truncated: false,
        }
    }

    /// Feed bytes and return complete JSON lines.
    ///
    /// If the buffer exceeds the maximum size (1MB), it will be truncated
    /// and a warning will be logged.
    pub fn feed(&mut self, bytes: &[u8]) -> Vec<String> {
        self.buffer.push_str(&String::from_utf8_lossy(bytes));

        // Prevent unbounded buffer growth
        if self.buffer.len() > MAX_BUFFER_SIZE {
            if !self.truncated {
                tracing::warn!(
                    "JSON Lines buffer exceeded {} bytes, truncating. This may indicate a malformed stream.",
                    MAX_BUFFER_SIZE
                );
                self.truncated = true;
            }
            // Truncate at a newline boundary to avoid splitting JSON lines
            let target_start = self.buffer.len() - MAX_BUFFER_SIZE / 2;
            let start = self.buffer[target_start..]
                .find('\n')
                .map(|pos| target_start + pos + 1)
                .unwrap_or(target_start);
            self.buffer = self.buffer[start..].to_string();
        }

        let mut lines = Vec::new();

        while let Some(pos) = self.buffer.find('\n') {
            let line: String = self.buffer.drain(..pos + 1).collect();
            let trimmed = line.trim();
            if !trimmed.is_empty() {
                lines.push(trimmed.to_string());
            }
            // Reset truncation flag after successfully parsing a line
            self.truncated = false;
        }

        lines
    }

    /// Clear the internal buffer.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.truncated = false;
    }

    /// Check if the parser has buffered data.
    pub fn has_buffered_data(&self) -> bool {
        !self.buffer.is_empty()
    }
}

// ============================================================================
// Token Counter Trait
// ============================================================================

/// Trait for counting tokens in text.
///
/// Implementations can use tiktoken, HuggingFace tokenizers, or other backends.
pub trait TokenCounter: Send + Sync {
    /// Count the number of tokens in the given text.
    fn count(&self, text: &str) -> usize;
}

/// Simple whitespace-based token counter (fallback).
///
/// # Accuracy Warning
///
/// This counter splits text on whitespace and counts the resulting words.
/// For LLM tokenization, this is a **very rough approximation** that can be
/// off by 2-4x compared to actual tokenizers like tiktoken or sentencepiece.
///
/// Use this only when:
/// - A real tokenizer is unavailable
/// - Approximate counts are acceptable
/// - You're working with English text (other languages may vary more)
///
/// For accurate token counts, implement `TokenCounter` with a proper
/// tokenizer library like `tiktoken-rs` or `tokenizers`.
#[derive(Debug, Default)]
pub struct WhitespaceTokenCounter;

impl TokenCounter for WhitespaceTokenCounter {
    fn count(&self, text: &str) -> usize {
        text.split_whitespace().count()
    }
}

// ============================================================================
// JSON Response Parsing Helpers
// ============================================================================

/// Extract content from various JSON response formats.
///
/// Supports OpenAI, Cohere, and Anthropic/Bedrock response structures.
fn extract_content(value: &serde_json::Value) -> Option<String> {
    // OpenAI format: choices[0].delta.content
    if let Some(content) = value
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("delta"))
        .and_then(|d| d.get("content"))
        .and_then(|c| c.as_str())
    {
        if !content.is_empty() {
            return Some(content.to_string());
        }
    }

    // Cohere format: text
    if let Some(text) = value.get("text").and_then(|t| t.as_str()) {
        if !text.is_empty() {
            return Some(text.to_string());
        }
    }

    // Bedrock/Claude/Anthropic format: delta.text
    if let Some(text) = value
        .get("delta")
        .and_then(|d| d.get("text"))
        .and_then(|t| t.as_str())
    {
        if !text.is_empty() {
            return Some(text.to_string());
        }
    }

    None
}

/// Extract finish reason from response.
///
/// Supports OpenAI and Anthropic response structures.
fn extract_finish_reason(value: &serde_json::Value) -> Option<FinishReason> {
    // OpenAI format: choices[0].finish_reason
    let reason_str = value
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("finish_reason"))
        .and_then(|r| r.as_str());

    if let Some(reason) = reason_str.and_then(parse_finish_reason) {
        return Some(reason);
    }

    // Anthropic format: delta.stop_reason
    let stop_reason = value
        .get("delta")
        .and_then(|d| d.get("stop_reason"))
        .and_then(|r| r.as_str());

    stop_reason.and_then(parse_finish_reason)
}

/// Parse a finish reason string into a FinishReason enum.
fn parse_finish_reason(s: &str) -> Option<FinishReason> {
    match s {
        "stop" | "end_turn" => Some(FinishReason::Stop),
        "length" | "max_tokens" => Some(FinishReason::Length),
        "content_filter" => Some(FinishReason::ContentFilter),
        "tool_calls" | "function_call" | "tool_use" => Some(FinishReason::ToolCalls),
        _ => None,
    }
}

/// Extract usage statistics from response.
///
/// Supports both OpenAI (prompt_tokens/completion_tokens) and
/// Anthropic (input_tokens/output_tokens) naming conventions.
fn extract_usage(value: &serde_json::Value) -> Option<Usage> {
    let usage_obj = value.get("usage")?;

    Some(Usage {
        prompt_tokens: usage_obj
            .get("prompt_tokens")
            .or_else(|| usage_obj.get("input_tokens"))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize),
        completion_tokens: usage_obj
            .get("completion_tokens")
            .or_else(|| usage_obj.get("output_tokens"))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize),
        total_tokens: usage_obj
            .get("total_tokens")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize),
    })
}

// ============================================================================
// Stream Processor
// ============================================================================

/// High-level stream processor that handles different streaming formats.
///
/// Wraps format-specific parsers and provides a unified interface for
/// extracting stream chunks with timing information.
pub struct StreamProcessor {
    /// The streaming format being processed
    format: StreamFormat,

    /// Optional tokenizer for token counting
    tokenizer: Option<Arc<dyn TokenCounter>>,

    /// SSE parser (lazily initialized)
    sse_parser: Option<SSEParser>,

    /// JSON Lines parser (lazily initialized)
    jsonl_parser: Option<JsonLinesParser>,
}

impl StreamProcessor {
    /// Create a new stream processor for the given format.
    pub fn new(format: StreamFormat) -> Self {
        Self {
            format,
            tokenizer: None,
            sse_parser: None,
            jsonl_parser: None,
        }
    }

    /// Set a tokenizer for token counting.
    pub fn with_tokenizer(mut self, tokenizer: Arc<dyn TokenCounter>) -> Self {
        self.tokenizer = Some(tokenizer);
        self
    }

    /// Get the stream format.
    pub fn format(&self) -> StreamFormat {
        self.format
    }

    /// Process incoming bytes and return stream chunks.
    ///
    /// The timestamp in each chunk reflects when it was processed.
    pub fn process(&mut self, bytes: &[u8]) -> Vec<StreamChunk> {
        let timestamp = Instant::now();

        match self.format {
            StreamFormat::SSE => self.process_sse(bytes, timestamp),
            StreamFormat::JsonLines => self.process_jsonl(bytes, timestamp),
            StreamFormat::EventStream => self.process_event_stream(bytes, timestamp),
        }
    }

    fn process_sse(&mut self, bytes: &[u8], timestamp: Instant) -> Vec<StreamChunk> {
        let parser = self.sse_parser.get_or_insert_with(SSEParser::new);
        let events = parser.feed(bytes);

        events
            .into_iter()
            .filter_map(|event| self.sse_event_to_chunk(event, timestamp))
            .collect()
    }

    fn process_jsonl(&mut self, bytes: &[u8], timestamp: Instant) -> Vec<StreamChunk> {
        let parser = self.jsonl_parser.get_or_insert_with(JsonLinesParser::new);
        let lines = parser.feed(bytes);

        lines
            .into_iter()
            .filter_map(|line| self.json_to_chunk(&line, timestamp))
            .collect()
    }

    fn process_event_stream(&mut self, _bytes: &[u8], _timestamp: Instant) -> Vec<StreamChunk> {
        // AWS Bedrock uses a binary event stream format that requires specialized parsing.
        // This is not yet implemented - vendor clients should handle Bedrock streaming directly.
        // See: https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-streaming.html
        tracing::warn!(
            "EventStream format not implemented in generic StreamProcessor; \
             use vendor-specific Bedrock client instead"
        );
        Vec::new()
    }

    /// Convert an SSE event to a stream chunk.
    fn sse_event_to_chunk(&self, event: SSEEvent, timestamp: Instant) -> Option<StreamChunk> {
        match event {
            SSEEvent::Done => Some(StreamChunk {
                timestamp,
                content: None,
                token_count: 0,
                is_final: true,
                finish_reason: Some(FinishReason::Stop),
                usage: None,
            }),
            SSEEvent::Data { data, .. } => self.json_to_chunk(&data, timestamp),
        }
    }

    /// Parse JSON data into a stream chunk.
    ///
    /// This is vendor-agnostic and handles common response formats.
    /// Vendor-specific parsing should be done at the vendor client level.
    fn json_to_chunk(&self, json_str: &str, timestamp: Instant) -> Option<StreamChunk> {
        // Parse as generic JSON value
        let value: serde_json::Value = serde_json::from_str(json_str).ok()?;

        // Extract content from common paths
        let content = extract_content(&value);

        // Count tokens if tokenizer is available
        let token_count = content.as_ref().map(|c| self.count_tokens(c)).unwrap_or(0);

        // Check for finish reason
        let finish_reason = extract_finish_reason(&value);

        // Check for usage stats
        let usage = extract_usage(&value);

        // Determine if this is the final chunk
        let is_final = finish_reason.is_some();

        Some(StreamChunk {
            timestamp,
            content,
            token_count,
            is_final,
            finish_reason,
            usage,
        })
    }

    /// Count tokens in a string.
    fn count_tokens(&self, text: &str) -> usize {
        if let Some(tokenizer) = &self.tokenizer {
            tokenizer.count(text)
        } else {
            // Fallback: rough estimate based on whitespace
            text.split_whitespace().count()
        }
    }

    /// Reset the processor state.
    pub fn reset(&mut self) {
        if let Some(parser) = &mut self.sse_parser {
            parser.reset();
        }
        if let Some(parser) = &mut self.jsonl_parser {
            parser.reset();
        }
    }
}

impl std::fmt::Debug for StreamProcessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamProcessor")
            .field("format", &self.format)
            .field("has_tokenizer", &self.tokenizer.is_some())
            .finish()
    }
}

// ============================================================================
// HTTP Client Pool
// ============================================================================

/// Configuration for the HTTP client pool.
#[derive(Debug, Clone)]
pub struct HttpConfig {
    /// Idle connection timeout
    pub pool_idle_timeout: Duration,

    /// Maximum idle connections per host
    pub pool_max_idle_per_host: usize,

    /// Request timeout
    pub request_timeout: Duration,

    /// Connection timeout
    pub connect_timeout: Duration,

    /// TCP keepalive interval
    pub tcp_keepalive: Option<Duration>,

    /// User agent string
    pub user_agent: String,
}

impl Default for HttpConfig {
    fn default() -> Self {
        Self {
            pool_idle_timeout: Duration::from_secs(90),
            pool_max_idle_per_host: 32,
            request_timeout: Duration::from_secs(300),
            connect_timeout: Duration::from_secs(30),
            tcp_keepalive: Some(Duration::from_secs(60)),
            user_agent: format!("genai-bench/{}", env!("CARGO_PKG_VERSION")),
        }
    }
}

impl HttpConfig {
    /// Create config with custom request timeout.
    pub fn with_request_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = timeout;
        self
    }

    /// Create config with custom connect timeout.
    pub fn with_connect_timeout(mut self, timeout: Duration) -> Self {
        self.connect_timeout = timeout;
        self
    }

    /// Create config with custom pool size.
    pub fn with_pool_max_idle(mut self, max_idle: usize) -> Self {
        self.pool_max_idle_per_host = max_idle;
        self
    }
}

/// Shared HTTP client with connection pooling.
///
/// This struct manages a pool of HTTP connections for efficient
/// communication with LLM vendor endpoints.
///
/// # Connection Pooling
///
/// The client maintains a pool of idle connections that are reused
/// for subsequent requests to the same host. This significantly
/// reduces latency for high-throughput benchmarking.
///
/// # Example
///
/// ```rust,ignore
/// let config = HttpConfig::default();
/// let pool = HttpClientPool::new(&config);
///
/// // Get the underlying client for making requests
/// let client = pool.client();
/// ```
#[derive(Debug, Clone)]
pub struct HttpClientPool {
    /// The underlying reqwest client
    client: Client,

    /// Configuration used to create this pool
    config: HttpConfig,
}

impl HttpClientPool {
    /// Create a new HTTP client pool with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the HTTP client cannot be built.
    #[must_use = "this returns a Result that should be checked"]
    pub fn new(config: &HttpConfig) -> Result<Self, reqwest::Error> {
        let mut builder = Client::builder()
            .pool_idle_timeout(config.pool_idle_timeout)
            .pool_max_idle_per_host(config.pool_max_idle_per_host)
            .timeout(config.request_timeout)
            .connect_timeout(config.connect_timeout)
            .user_agent(&config.user_agent);

        if let Some(keepalive) = config.tcp_keepalive {
            builder = builder.tcp_keepalive(keepalive);
        }

        let client = builder.build()?;

        Ok(Self {
            client,
            config: config.clone(),
        })
    }

    /// Create a pool with default configuration.
    ///
    /// # Panics
    ///
    /// Panics if the default configuration fails to build a client,
    /// which should never happen under normal circumstances.
    pub fn with_defaults() -> Self {
        Self::new(&HttpConfig::default()).expect("Default HTTP config should always be valid")
    }

    /// Get a reference to the underlying HTTP client.
    pub fn client(&self) -> &Client {
        &self.client
    }

    /// Get the configuration for this pool.
    pub fn config(&self) -> &HttpConfig {
        &self.config
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sse_parser_basic() {
        let mut parser = SSEParser::new();

        let input = b"data: {\"content\": \"hello\"}\n\n";
        let events = parser.feed(input);

        assert_eq!(events.len(), 1);
        match &events[0] {
            SSEEvent::Data { data, .. } => {
                assert_eq!(data, "{\"content\": \"hello\"}");
            }
            _ => panic!("Expected Data event"),
        }
    }

    #[test]
    fn test_sse_parser_done() {
        let mut parser = SSEParser::new();

        let input = b"data: [DONE]\n\n";
        let events = parser.feed(input);

        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], SSEEvent::Done));
    }

    #[test]
    fn test_sse_parser_partial() {
        let mut parser = SSEParser::new();

        // First chunk (incomplete)
        let events1 = parser.feed(b"data: {\"content\":");
        assert!(events1.is_empty());
        assert!(parser.has_buffered_data());

        // Second chunk (completes the event)
        let events2 = parser.feed(b" \"hello\"}\n\n");
        assert_eq!(events2.len(), 1);
    }

    #[test]
    fn test_sse_parser_multiple_events() {
        let mut parser = SSEParser::new();

        let input = b"data: {\"n\": 1}\n\ndata: {\"n\": 2}\n\ndata: [DONE]\n\n";
        let events = parser.feed(input);

        assert_eq!(events.len(), 3);
        assert!(matches!(events[2], SSEEvent::Done));
    }

    #[test]
    fn test_sse_parser_with_event_type() {
        let mut parser = SSEParser::new();

        let input = b"event: message\ndata: hello\n\n";
        let events = parser.feed(input);

        assert_eq!(events.len(), 1);
        match &events[0] {
            SSEEvent::Data { event_type, .. } => {
                assert_eq!(event_type.as_deref(), Some("message"));
            }
            _ => panic!("Expected Data event"),
        }
    }

    #[test]
    fn test_sse_parser_with_id_and_retry() {
        let mut parser = SSEParser::new();

        let input = b"id: 123\nretry: 5000\ndata: test\n\n";
        let events = parser.feed(input);

        assert_eq!(events.len(), 1);
        match &events[0] {
            SSEEvent::Data { id, retry, .. } => {
                assert_eq!(id.as_deref(), Some("123"));
                assert_eq!(*retry, Some(5000));
            }
            _ => panic!("Expected Data event"),
        }
    }

    #[test]
    fn test_sse_parser_multiline_data() {
        let mut parser = SSEParser::new();

        let input = b"data: line1\ndata: line2\ndata: line3\n\n";
        let events = parser.feed(input);

        assert_eq!(events.len(), 1);
        match &events[0] {
            SSEEvent::Data { data, .. } => {
                assert_eq!(data, "line1\nline2\nline3");
            }
            _ => panic!("Expected Data event"),
        }
    }

    #[test]
    fn test_sse_parser_no_space_after_colon() {
        let mut parser = SSEParser::new();

        let input = b"data:{\"content\":\"hello\"}\n\n";
        let events = parser.feed(input);

        assert_eq!(events.len(), 1);
        match &events[0] {
            SSEEvent::Data { data, .. } => {
                assert_eq!(data, "{\"content\":\"hello\"}");
            }
            _ => panic!("Expected Data event"),
        }
    }

    #[test]
    fn test_sse_event_methods() {
        let data_event = SSEEvent::Data {
            data: "test".to_string(),
            event_type: Some("message".to_string()),
            id: None,
            retry: None,
        };

        assert!(!data_event.is_done());
        assert_eq!(data_event.data(), Some("test"));
        assert_eq!(data_event.event_type(), Some("message"));

        let done_event = SSEEvent::Done;
        assert!(done_event.is_done());
        assert_eq!(done_event.data(), None);
    }

    #[test]
    fn test_jsonl_parser() {
        let mut parser = JsonLinesParser::new();

        let input = b"{\"a\": 1}\n{\"b\": 2}\n";
        let lines = parser.feed(input);

        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0], "{\"a\": 1}");
        assert_eq!(lines[1], "{\"b\": 2}");
    }

    #[test]
    fn test_jsonl_parser_partial() {
        let mut parser = JsonLinesParser::new();

        // First chunk (incomplete)
        let lines1 = parser.feed(b"{\"partial\":");
        assert!(lines1.is_empty());
        assert!(parser.has_buffered_data());

        // Second chunk (completes the line)
        let lines2 = parser.feed(b" true}\n");
        assert_eq!(lines2.len(), 1);
        assert_eq!(lines2[0], "{\"partial\": true}");
    }

    #[test]
    fn test_jsonl_parser_empty_lines() {
        let mut parser = JsonLinesParser::new();

        let input = b"{\"a\": 1}\n\n\n{\"b\": 2}\n";
        let lines = parser.feed(input);

        assert_eq!(lines.len(), 2);
    }

    #[test]
    fn test_whitespace_token_counter() {
        let counter = WhitespaceTokenCounter;

        assert_eq!(counter.count("hello world"), 2);
        assert_eq!(counter.count("one two three four"), 4);
        assert_eq!(counter.count("  spaces  everywhere  "), 2);
        assert_eq!(counter.count(""), 0);
    }

    #[test]
    fn test_stream_processor_sse() {
        let mut processor = StreamProcessor::new(StreamFormat::SSE);

        let input = b"data: {\"choices\": [{\"delta\": {\"content\": \"Hello\"}}]}\n\n";
        let chunks = processor.process(input);

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content, Some("Hello".to_string()));
        assert!(!chunks[0].is_final);
    }

    #[test]
    fn test_stream_processor_done() {
        let mut processor = StreamProcessor::new(StreamFormat::SSE);

        let input = b"data: [DONE]\n\n";
        let chunks = processor.process(input);

        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].is_final);
        assert_eq!(chunks[0].finish_reason, Some(FinishReason::Stop));
    }

    #[test]
    fn test_stream_processor_with_finish_reason() {
        let mut processor = StreamProcessor::new(StreamFormat::SSE);

        let input = b"data: {\"choices\": [{\"delta\": {}, \"finish_reason\": \"stop\"}]}\n\n";
        let chunks = processor.process(input);

        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].is_final);
        assert_eq!(chunks[0].finish_reason, Some(FinishReason::Stop));
    }

    #[test]
    fn test_stream_processor_with_usage() {
        let mut processor = StreamProcessor::new(StreamFormat::SSE);

        let input = b"data: {\"choices\": [{\"delta\": {}, \"finish_reason\": \"stop\"}], \"usage\": {\"prompt_tokens\": 10, \"completion_tokens\": 20, \"total_tokens\": 30}}\n\n";
        let chunks = processor.process(input);

        assert_eq!(chunks.len(), 1);
        let usage = chunks[0].usage.as_ref().unwrap();
        assert_eq!(usage.prompt_tokens, Some(10));
        assert_eq!(usage.completion_tokens, Some(20));
        assert_eq!(usage.total_tokens, Some(30));
    }

    #[test]
    fn test_stream_processor_cohere_format() {
        let mut processor = StreamProcessor::new(StreamFormat::JsonLines);

        let input = b"{\"text\": \"Hello from Cohere\"}\n";
        let chunks = processor.process(input);

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content, Some("Hello from Cohere".to_string()));
    }

    #[test]
    fn test_stream_processor_reset() {
        let mut processor = StreamProcessor::new(StreamFormat::SSE);

        // Feed partial data
        let _ = processor.process(b"data: partial");
        processor.reset();

        // After reset, should process fresh input
        let chunks =
            processor.process(b"data: {\"choices\": [{\"delta\": {\"content\": \"new\"}}]}\n\n");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content, Some("new".to_string()));
    }

    #[test]
    fn test_http_client_pool_creation() {
        let config = HttpConfig::default();
        let pool = HttpClientPool::new(&config).expect("Should create pool");

        assert_eq!(pool.config().request_timeout, Duration::from_secs(300));
        assert_eq!(pool.config().pool_max_idle_per_host, 32);
    }

    #[test]
    fn test_http_client_pool_defaults() {
        let pool = HttpClientPool::with_defaults();

        assert_eq!(pool.config().connect_timeout, Duration::from_secs(30));
        assert!(pool.config().user_agent.starts_with("genai-bench/"));
    }

    #[test]
    fn test_http_config_builder() {
        let config = HttpConfig::default()
            .with_request_timeout(Duration::from_secs(60))
            .with_connect_timeout(Duration::from_secs(10))
            .with_pool_max_idle(16);

        assert_eq!(config.request_timeout, Duration::from_secs(60));
        assert_eq!(config.connect_timeout, Duration::from_secs(10));
        assert_eq!(config.pool_max_idle_per_host, 16);
    }

    // ========================================================================
    // Error case tests
    // ========================================================================

    #[test]
    fn test_sse_parser_malformed_no_data() {
        let mut parser = SSEParser::new();

        // Event with no data field should be skipped
        let input = b"event: ping\nid: 123\n\n";
        let events = parser.feed(input);
        assert!(events.is_empty());
    }

    #[test]
    fn test_sse_parser_invalid_utf8() {
        let mut parser = SSEParser::new();

        // Invalid UTF-8 bytes should be handled gracefully (lossy conversion)
        let input: &[u8] = &[b'd', b'a', b't', b'a', b':', b' ', 0xFF, 0xFE, b'\n', b'\n'];
        let events = parser.feed(input);

        assert_eq!(events.len(), 1);
        // Should contain replacement characters for invalid UTF-8
        match &events[0] {
            SSEEvent::Data { data, .. } => {
                assert!(data.contains('\u{FFFD}')); // Unicode replacement character
            }
            _ => panic!("Expected Data event"),
        }
    }

    #[test]
    fn test_jsonl_parser_invalid_json_gracefully_returns() {
        let mut parser = JsonLinesParser::new();

        // Invalid JSON is still returned as a line - parsing happens later
        let input = b"not valid json at all\n";
        let lines = parser.feed(input);

        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0], "not valid json at all");
    }

    #[test]
    fn test_stream_processor_invalid_json_skipped() {
        let mut processor = StreamProcessor::new(StreamFormat::SSE);

        // Invalid JSON in SSE data should be skipped (returns None from json_to_chunk)
        let input = b"data: not valid json\n\n";
        let chunks = processor.process(input);

        // Should be empty because invalid JSON can't be parsed
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_stream_processor_empty_content_skipped() {
        let mut processor = StreamProcessor::new(StreamFormat::SSE);

        // JSON with empty content should still produce a chunk (for metadata)
        let input = b"data: {\"choices\": [{\"delta\": {\"content\": \"\"}}]}\n\n";
        let chunks = processor.process(input);

        assert_eq!(chunks.len(), 1);
        // Empty content is returned as None
        assert_eq!(chunks[0].content, None);
    }

    #[test]
    fn test_stream_processor_missing_content_field() {
        let mut processor = StreamProcessor::new(StreamFormat::SSE);

        // JSON without any recognized content field
        let input = b"data: {\"id\": \"123\", \"object\": \"chat.completion.chunk\"}\n\n";
        let chunks = processor.process(input);

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content, None);
    }

    #[test]
    fn test_stream_processor_event_stream_not_implemented() {
        let mut processor = StreamProcessor::new(StreamFormat::EventStream);

        // EventStream returns empty vec with warning (not implemented in generic processor)
        let input = b"some binary data";
        let chunks = processor.process(input);

        assert!(chunks.is_empty());
    }
}
