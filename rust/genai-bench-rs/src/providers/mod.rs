//! LLM provider implementations
//!
//! Each provider module implements the `Provider` trait to interact with
//! different LLM APIs (OpenAI, Azure, GCP, AWS, Anthropic).

pub mod openai;

use crate::metrics::RequestMetrics;
use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Common request structure for chat completions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub stream: Option<bool>,
}

/// Message in a chat conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

/// Common response structure for chat completions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    pub id: String,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Option<Usage>,
}

/// A single completion choice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Choice {
    pub index: u32,
    pub message: Message,
    pub finish_reason: Option<String>,
}

/// Token usage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// Provider trait that all LLM providers must implement
#[async_trait]
pub trait Provider: Send + Sync {
    /// Send a chat completion request and return metrics
    async fn chat(&self, request: &ChatRequest) -> Result<RequestMetrics>;

    /// Get the provider name
    fn name(&self) -> &str;
}
