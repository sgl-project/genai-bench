//! Vendor abstraction layer for genai-bench.
//!
//! This crate provides a unified interface for interacting with various LLM providers,
//! along with streaming response processing infrastructure.
//!
//! # Supported Vendors
//!
//! - OpenAI (and OpenAI-compatible APIs like vLLM, SGLang)
//! - Azure OpenAI Service
//! - AWS Bedrock
//! - Google Cloud Vertex AI
//! - Oracle Cloud Infrastructure (OCI) Generative AI
//! - Cohere
//! - Together AI
//!
//! # Architecture
//!
//! The `VendorClient` trait (defined in `genai-bench-core`) provides the core abstraction.
//! This crate provides:
//!
//! - [`Vendor`] - Enumeration of supported vendors
//! - [`VendorConfig`] - Configuration for vendor clients
//! - [`StreamFormat`] - Streaming response format types
//! - Streaming parsers ([`SSEParser`], [`JsonLinesParser`])
//! - [`StreamProcessor`] - Unified stream processing
//! - [`HttpClientPool`] - Connection pooling for efficient HTTP requests
//!
//! # Example
//!
//! ```rust,ignore
//! use genai_bench_vendors::{Vendor, VendorConfig, HttpClientPool, StreamProcessor};
//!
//! // Create vendor configuration
//! let config = VendorConfig::new(Vendor::OpenAI, "gpt-4")
//!     .with_endpoint("https://api.openai.com/v1");
//!
//! // Create HTTP client pool
//! let pool = HttpClientPool::with_defaults();
//!
//! // Create stream processor for SSE format
//! let mut processor = StreamProcessor::new(config.vendor.default_stream_format());
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod streaming;
pub mod traits;

// Re-export main types from traits
pub use traits::{ConfigValidationError, StreamFormat, Vendor, VendorConfig};

// Re-export streaming types
pub use streaming::{
    HttpClientPool, HttpConfig, JsonLinesParser, SSEEvent, SSEParser, StreamProcessor,
    TokenCounter, WhitespaceTokenCounter,
};

// Re-export core vendor types for convenience
pub use genai_bench_core::{FinishReason, StreamChunk, Usage, VendorClient, VendorError};
