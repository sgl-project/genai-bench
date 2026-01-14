//! Request types for benchmark operations

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Task types supported by the benchmark
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Task {
    /// Text-to-text generation (chat completion)
    TextToText,
    /// Text-to-embeddings (embedding generation)
    TextToEmbeddings,
    /// Text-to-rerank (reranking)
    TextToRerank,
    /// Image+text-to-text (vision)
    ImageTextToText,
    /// Image-to-embeddings (image embedding)
    ImageToEmbeddings,
}

impl std::fmt::Display for Task {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Task::TextToText => write!(f, "text_to_text"),
            Task::TextToEmbeddings => write!(f, "text_to_embeddings"),
            Task::TextToRerank => write!(f, "text_to_rerank"),
            Task::ImageTextToText => write!(f, "image_text_to_text"),
            Task::ImageToEmbeddings => write!(f, "image_to_embeddings"),
        }
    }
}

/// Unique request identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RequestId(pub u64);

impl std::fmt::Display for RequestId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<u64> for RequestId {
    fn from(id: u64) -> Self {
        Self(id)
    }
}

/// A single benchmark request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkRequest {
    /// Unique request identifier
    pub id: RequestId,

    /// Task type
    pub task: Task,

    /// Messages for chat completion
    pub messages: Vec<Message>,

    /// Target input token count
    pub input_tokens: usize,

    /// Target output token count (for generation tasks)
    pub output_tokens: Option<usize>,

    /// Sampling parameters
    pub params: SamplingParams,

    /// Request metadata
    pub metadata: RequestMetadata,
}

/// Chat message (OpenAI-compatible format)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Message role (system, user, assistant)
    pub role: Role,
    /// Message content (text or multimodal)
    pub content: Content,
}

impl Message {
    /// Create a new text message
    pub fn text(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: Content::Text(content.into()),
        }
    }

    /// Create a system message
    pub fn system(content: impl Into<String>) -> Self {
        Self::text(Role::System, content)
    }

    /// Create a user message
    pub fn user(content: impl Into<String>) -> Self {
        Self::text(Role::User, content)
    }

    /// Create an assistant message
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::text(Role::Assistant, content)
    }
}

/// Message role
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// System message (instructions)
    System,
    /// User message (input)
    User,
    /// Assistant message (output)
    Assistant,
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Role::System => write!(f, "system"),
            Role::User => write!(f, "user"),
            Role::Assistant => write!(f, "assistant"),
        }
    }
}

/// Message content (text or multimodal parts)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Content {
    /// Simple text content
    Text(String),
    /// Multimodal content parts
    Parts(Vec<ContentPart>),
}

impl Content {
    /// Get the text content if this is a simple text content
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Content::Text(text) => Some(text),
            Content::Parts(_) => None,
        }
    }

    /// Check if this content contains any images
    pub fn has_images(&self) -> bool {
        match self {
            Content::Text(_) => false,
            Content::Parts(parts) => parts
                .iter()
                .any(|p| matches!(p, ContentPart::ImageUrl { .. })),
        }
    }
}

impl From<String> for Content {
    fn from(text: String) -> Self {
        Content::Text(text)
    }
}

impl From<&str> for Content {
    fn from(text: &str) -> Self {
        Content::Text(text.to_string())
    }
}

/// Content part for multimodal messages
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentPart {
    /// Text content part
    #[serde(rename = "text")]
    Text {
        /// The text content
        text: String,
    },

    /// Image URL content part
    #[serde(rename = "image_url")]
    ImageUrl {
        /// The image URL configuration
        image_url: ImageUrl,
    },
}

impl ContentPart {
    /// Create a text content part
    pub fn text(text: impl Into<String>) -> Self {
        ContentPart::Text { text: text.into() }
    }

    /// Create an image URL content part
    pub fn image_url(url: impl Into<String>) -> Self {
        ContentPart::ImageUrl {
            image_url: ImageUrl {
                url: url.into(),
                detail: None,
            },
        }
    }

    /// Create an image URL content part with detail level
    pub fn image_url_with_detail(url: impl Into<String>, detail: ImageDetail) -> Self {
        ContentPart::ImageUrl {
            image_url: ImageUrl {
                url: url.into(),
                detail: Some(detail),
            },
        }
    }
}

/// Image URL with optional detail level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrl {
    /// The URL of the image
    pub url: String,
    /// Detail level for processing
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<ImageDetail>,
}

/// Image detail level for vision models
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ImageDetail {
    /// Automatic detail level
    #[default]
    Auto,
    /// Low detail (faster, cheaper)
    Low,
    /// High detail (slower, more accurate)
    High,
}

/// Sampling parameters for generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingParams {
    /// Temperature for sampling (0.0 = deterministic)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Top-p (nucleus) sampling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    /// Maximum tokens to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<usize>,

    /// Stop sequences
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,

    /// Whether to stream the response
    pub stream: bool,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: Some(0.0), // Deterministic by default for reproducible benchmarks
            top_p: None,
            max_tokens: None,
            stop: None,
            stream: true, // Always stream for token-level metrics
        }
    }
}

impl SamplingParams {
    /// Create params for deterministic generation
    pub fn deterministic() -> Self {
        Self {
            temperature: Some(0.0),
            ..Default::default()
        }
    }

    /// Create params with specific max tokens
    pub fn with_max_tokens(max_tokens: usize) -> Self {
        Self {
            max_tokens: Some(max_tokens),
            ..Default::default()
        }
    }
}

/// Request metadata (not sent to vendor, used for tracking)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestMetadata {
    /// Which sampler generated this request
    pub sampler_name: String,

    /// Scenario name (e.g., "input_512_output_128")
    pub scenario_name: String,

    /// When the request was created
    pub created_at: chrono::DateTime<chrono::Utc>,

    /// Additional custom metadata
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub extra: HashMap<String, serde_json::Value>,
}

impl RequestMetadata {
    /// Create new metadata with default values
    pub fn new(sampler_name: impl Into<String>, scenario_name: impl Into<String>) -> Self {
        Self {
            sampler_name: sampler_name.into(),
            scenario_name: scenario_name.into(),
            created_at: chrono::Utc::now(),
            extra: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_serialization() {
        let msg = Message::user("Hello, world!");
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"role\":\"user\""));
        assert!(json.contains("\"content\":\"Hello, world!\""));
    }

    #[test]
    fn test_multimodal_content_serialization() {
        let content = Content::Parts(vec![
            ContentPart::text("Describe this image:"),
            ContentPart::image_url("https://example.com/image.png"),
        ]);
        let json = serde_json::to_string(&content).unwrap();
        assert!(json.contains("\"type\":\"text\""));
        assert!(json.contains("\"type\":\"image_url\""));
    }

    #[test]
    fn test_task_display() {
        assert_eq!(Task::TextToText.to_string(), "text_to_text");
        assert_eq!(Task::ImageTextToText.to_string(), "image_text_to_text");
    }

    #[test]
    fn test_request_id_from_u64() {
        let id: RequestId = 42u64.into();
        assert_eq!(id.0, 42);
        assert_eq!(id.to_string(), "42");
    }

    #[test]
    fn test_content_has_images() {
        let text_content = Content::Text("Hello".to_string());
        assert!(!text_content.has_images());

        let image_content =
            Content::Parts(vec![ContentPart::image_url("https://example.com/img.png")]);
        assert!(image_content.has_images());
    }

    #[test]
    fn test_sampling_params_default() {
        let params = SamplingParams::default();
        assert_eq!(params.temperature, Some(0.0));
        assert!(params.stream);
    }
}
