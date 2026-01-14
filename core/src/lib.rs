//! genai-bench-core: Core data structures for benchmarking LLM APIs
//!
//! This crate provides the foundational types used across all genai-bench components,
//! including:
//!
//! - Protocol data structures (requests, responses)
//! - Core traits (VendorClient, Sampler)
//! - Metrics collection and aggregation
//! - Error handling

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod error;
pub mod metrics;
pub mod request;
pub mod response;
pub mod traits;
pub mod worker;

pub use error::*;
pub use metrics::*;
pub use request::*;
pub use response::*;
pub use traits::*;
pub use worker::{RequestRateLimiter, Worker, WorkerBuilder, WorkerStats};

#[cfg(test)]
mod integration_tests {
    use super::*;

    // =========================================================================
    // Round-trip serialization tests
    // =========================================================================

    #[test]
    fn test_message_roundtrip() {
        let msg = Message::user("Hello, world!");
        let json = serde_json::to_string(&msg).unwrap();
        let deserialized: Message = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.role, Role::User);
        assert_eq!(deserialized.content.as_text(), Some("Hello, world!"));
    }

    #[test]
    fn test_multimodal_content_roundtrip() {
        let content = Content::Parts(vec![
            ContentPart::text("Describe this image:"),
            ContentPart::image_url("https://example.com/image.png"),
        ]);
        let json = serde_json::to_string(&content).unwrap();
        let deserialized: Content = serde_json::from_str(&json).unwrap();

        match deserialized {
            Content::Parts(parts) => {
                assert_eq!(parts.len(), 2);
            }
            _ => panic!("Expected Parts variant"),
        }
    }

    #[test]
    fn test_task_roundtrip() {
        for task in [
            Task::TextToText,
            Task::TextToEmbeddings,
            Task::TextToRerank,
            Task::ImageTextToText,
            Task::ImageToEmbeddings,
        ] {
            let json = serde_json::to_string(&task).unwrap();
            let deserialized: Task = serde_json::from_str(&json).unwrap();
            assert_eq!(deserialized, task);
        }
    }

    #[test]
    fn test_error_kind_roundtrip() {
        for kind in [
            ErrorKind::Timeout,
            ErrorKind::RateLimited,
            ErrorKind::InvalidRequest,
            ErrorKind::AuthenticationFailed,
            ErrorKind::ServerError,
            ErrorKind::ConnectionError,
            ErrorKind::StreamingError,
            ErrorKind::ContentFiltered,
            ErrorKind::ModelNotFound,
            ErrorKind::QuotaExceeded,
            ErrorKind::Unknown,
        ] {
            let json = serde_json::to_string(&kind).unwrap();
            let deserialized: ErrorKind = serde_json::from_str(&json).unwrap();
            assert_eq!(deserialized, kind);
        }
    }

    #[test]
    fn test_token_counts_roundtrip() {
        let counts = TokenCounts::new(100, 50);
        let json = serde_json::to_string(&counts).unwrap();
        let deserialized: TokenCounts = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.input, 100);
        assert_eq!(deserialized.output, 50);
        assert_eq!(deserialized.total, 150);
    }

    #[test]
    fn test_response_status_roundtrip() {
        let success = ResponseStatus::Success;
        let json = serde_json::to_string(&success).unwrap();
        let deserialized: ResponseStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, success);

        let error = ResponseStatus::Error(ErrorKind::Timeout);
        let json = serde_json::to_string(&error).unwrap();
        let deserialized: ResponseStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, error);
    }

    // =========================================================================
    // OpenAI JSON format compatibility tests
    // =========================================================================

    #[test]
    fn test_message_json_format() {
        let msg = Message::user("Hello, world!");
        let json = serde_json::to_string(&msg).unwrap();

        // Verify OpenAI-compatible format
        assert!(json.contains("\"role\":\"user\""));
        assert!(json.contains("\"content\":\"Hello, world!\""));
    }

    #[test]
    fn test_multimodal_content_json_format() {
        let content = Content::Parts(vec![
            ContentPart::text("Describe this image:"),
            ContentPart::image_url("https://example.com/image.png"),
        ]);
        let json = serde_json::to_string(&content).unwrap();

        // Verify OpenAI-compatible tagged format
        assert!(json.contains("\"type\":\"text\""));
        assert!(json.contains("\"type\":\"image_url\""));
        assert!(json.contains("\"text\":\"Describe this image:\""));
        assert!(json.contains("\"url\":\"https://example.com/image.png\""));
    }

    #[test]
    fn test_image_detail_json_format() {
        let part =
            ContentPart::image_url_with_detail("https://example.com/img.png", ImageDetail::High);
        let json = serde_json::to_string(&part).unwrap();

        assert!(json.contains("\"detail\":\"high\""));
    }

    #[test]
    fn test_task_snake_case_serialization() {
        assert_eq!(
            serde_json::to_string(&Task::TextToText).unwrap(),
            "\"text_to_text\""
        );
        assert_eq!(
            serde_json::to_string(&Task::ImageTextToText).unwrap(),
            "\"image_text_to_text\""
        );
    }

    #[test]
    fn test_role_lowercase_serialization() {
        assert_eq!(serde_json::to_string(&Role::System).unwrap(), "\"system\"");
        assert_eq!(serde_json::to_string(&Role::User).unwrap(), "\"user\"");
        assert_eq!(
            serde_json::to_string(&Role::Assistant).unwrap(),
            "\"assistant\""
        );
    }

    // =========================================================================
    // Integration tests from task spec
    // =========================================================================

    #[test]
    fn test_token_counts_arithmetic() {
        let a = TokenCounts::new(100, 50);
        let b = TokenCounts::new(200, 100);
        let sum = a + b;

        assert_eq!(sum.input, 300);
        assert_eq!(sum.output, 150);
        assert_eq!(sum.total, 450);
    }

    #[test]
    fn test_response_timing_tpot() {
        let mut timing = ResponseTiming::from_timestamps(
            chrono::Utc::now(),
            None,
            Some(chrono::Utc::now()),
            chrono::Utc::now(),
        );
        timing.token_timings_ms = vec![100.0, 120.0, 140.0, 160.0, 180.0];

        let tpot = timing.tpot_ms().unwrap();
        assert!((tpot - 20.0).abs() < 0.01); // Mean of inter-token times
    }

    #[test]
    fn test_skip_serializing_if_works() {
        // Test that optional fields are skipped when None/empty
        let params = SamplingParams {
            temperature: None,
            top_p: None,
            max_tokens: None,
            stop: None,
            stream: true,
        };
        let json = serde_json::to_string(&params).unwrap();

        // Only stream should be present
        assert!(!json.contains("temperature"));
        assert!(!json.contains("top_p"));
        assert!(!json.contains("max_tokens"));
        assert!(!json.contains("stop"));
        assert!(json.contains("\"stream\":true"));
    }

    #[test]
    fn test_content_untagged_serialization() {
        // Text content should serialize as just a string
        let text = Content::Text("Hello".to_string());
        let json = serde_json::to_string(&text).unwrap();
        assert_eq!(json, "\"Hello\"");

        // Parts should serialize as an array
        let parts = Content::Parts(vec![ContentPart::text("Hello")]);
        let json = serde_json::to_string(&parts).unwrap();
        assert!(json.starts_with('['));
    }
}
