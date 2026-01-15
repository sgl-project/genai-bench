//! Vendor types and configuration
//!
//! This module provides vendor enumeration, configuration structures,
//! and streaming format definitions.

use std::time::Duration;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Configuration validation error.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ConfigValidationError {
    /// A required configuration field is missing.
    #[error("missing required field: {0}")]
    MissingField(&'static str),

    /// A timeout value is out of acceptable range.
    #[error("invalid timeout: {0:?}")]
    InvalidTimeout(Duration),
}

// ============================================================================
// Vendor Enumeration
// ============================================================================

/// Enumeration of supported LLM vendors.
///
/// Used for configuration parsing and vendor client factory dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Vendor {
    /// OpenAI API (also compatible with vLLM, SGLang, etc.)
    #[serde(rename = "openai")]
    OpenAI,
    /// Azure OpenAI Service
    #[serde(rename = "azure_openai")]
    AzureOpenAI,
    /// AWS Bedrock
    #[serde(rename = "aws_bedrock")]
    AwsBedrock,
    /// Google Cloud Vertex AI
    #[serde(rename = "gcp_vertex")]
    GcpVertex,
    /// Oracle Cloud Infrastructure Generative AI
    #[serde(rename = "oci_genai")]
    OciGenAI,
    /// OCI-hosted Cohere models
    #[serde(rename = "oci_cohere")]
    OciCohere,
    /// Cohere API
    #[serde(rename = "cohere")]
    Cohere,
    /// Together AI
    #[serde(rename = "together")]
    Together,
}

impl Vendor {
    /// Returns the display name for this vendor.
    pub fn display_name(&self) -> &'static str {
        match self {
            Vendor::OpenAI => "OpenAI",
            Vendor::AzureOpenAI => "Azure OpenAI",
            Vendor::AwsBedrock => "AWS Bedrock",
            Vendor::GcpVertex => "GCP Vertex AI",
            Vendor::OciGenAI => "OCI Generative AI",
            Vendor::OciCohere => "OCI Cohere",
            Vendor::Cohere => "Cohere",
            Vendor::Together => "Together AI",
        }
    }

    /// Returns the identifier string for this vendor.
    pub fn id(&self) -> &'static str {
        match self {
            Vendor::OpenAI => "openai",
            Vendor::AzureOpenAI => "azure_openai",
            Vendor::AwsBedrock => "aws_bedrock",
            Vendor::GcpVertex => "gcp_vertex",
            Vendor::OciGenAI => "oci_genai",
            Vendor::OciCohere => "oci_cohere",
            Vendor::Cohere => "cohere",
            Vendor::Together => "together",
        }
    }

    /// Returns the default streaming format for this vendor.
    pub fn default_stream_format(&self) -> StreamFormat {
        match self {
            Vendor::OpenAI | Vendor::AzureOpenAI | Vendor::Cohere | Vendor::Together => {
                StreamFormat::SSE
            }
            Vendor::AwsBedrock => StreamFormat::EventStream,
            Vendor::GcpVertex => StreamFormat::SSE,
            Vendor::OciGenAI | Vendor::OciCohere => StreamFormat::JsonLines,
        }
    }

    /// Returns all supported vendors.
    pub fn all() -> &'static [Vendor] {
        &[
            Vendor::OpenAI,
            Vendor::AzureOpenAI,
            Vendor::AwsBedrock,
            Vendor::GcpVertex,
            Vendor::OciGenAI,
            Vendor::OciCohere,
            Vendor::Cohere,
            Vendor::Together,
        ]
    }
}

impl std::fmt::Display for Vendor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

impl std::str::FromStr for Vendor {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "openai" => Ok(Vendor::OpenAI),
            "azure_openai" | "azure-openai" | "azureopenai" => Ok(Vendor::AzureOpenAI),
            "aws_bedrock" | "aws-bedrock" | "bedrock" => Ok(Vendor::AwsBedrock),
            "gcp_vertex" | "gcp-vertex" | "vertex" | "vertexai" => Ok(Vendor::GcpVertex),
            "oci_genai" | "oci-genai" | "ocigenai" => Ok(Vendor::OciGenAI),
            "oci_cohere" | "oci-cohere" | "ocicohere" => Ok(Vendor::OciCohere),
            "cohere" => Ok(Vendor::Cohere),
            "together" | "together_ai" | "together-ai" => Ok(Vendor::Together),
            _ => Err(format!("Unknown vendor: {}", s)),
        }
    }
}

// ============================================================================
// Stream Format
// ============================================================================

/// Streaming response format used by different vendors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StreamFormat {
    /// Server-Sent Events (OpenAI, Azure, Cohere, Together, GCP)
    #[serde(rename = "sse")]
    SSE,
    /// JSON Lines / newline-delimited JSON (OCI)
    #[serde(rename = "json_lines")]
    JsonLines,
    /// AWS event stream format (Bedrock)
    #[serde(rename = "event_stream")]
    EventStream,
}

impl StreamFormat {
    /// Returns true if this format uses SSE protocol.
    pub fn is_sse(&self) -> bool {
        matches!(self, StreamFormat::SSE)
    }

    /// Returns the expected Content-Type header for this format.
    pub fn content_type(&self) -> &'static str {
        match self {
            StreamFormat::SSE => "text/event-stream",
            StreamFormat::JsonLines => "application/x-ndjson",
            StreamFormat::EventStream => "application/vnd.amazon.eventstream",
        }
    }
}

// ============================================================================
// Vendor Configuration
// ============================================================================

/// Configuration for creating a vendor client.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VendorConfig {
    /// Target vendor
    pub vendor: Vendor,

    /// Model identifier
    pub model: String,

    /// Base URL or endpoint (vendor-specific)
    #[serde(default)]
    pub endpoint: Option<String>,

    /// API version (Azure, GCP)
    #[serde(default)]
    pub api_version: Option<String>,

    /// Deployment name (Azure)
    #[serde(default)]
    pub deployment: Option<String>,

    /// Project ID (GCP)
    #[serde(default)]
    pub project: Option<String>,

    /// Region/location
    #[serde(default)]
    pub region: Option<String>,

    /// Compartment ID (OCI)
    #[serde(default)]
    pub compartment_id: Option<String>,

    /// Organization ID (OpenAI)
    #[serde(default)]
    pub organization: Option<String>,

    /// Request timeout
    #[serde(default = "default_request_timeout")]
    #[serde(with = "humantime_serde")]
    pub request_timeout: Duration,

    /// Connection timeout
    #[serde(default = "default_connect_timeout")]
    #[serde(with = "humantime_serde")]
    pub connect_timeout: Duration,
}

fn default_request_timeout() -> Duration {
    Duration::from_secs(300)
}

fn default_connect_timeout() -> Duration {
    Duration::from_secs(30)
}

impl VendorConfig {
    /// Create a new vendor config with required fields.
    pub fn new(vendor: Vendor, model: impl Into<String>) -> Self {
        Self {
            vendor,
            model: model.into(),
            endpoint: None,
            api_version: None,
            deployment: None,
            project: None,
            region: None,
            compartment_id: None,
            organization: None,
            request_timeout: default_request_timeout(),
            connect_timeout: default_connect_timeout(),
        }
    }

    /// Set the endpoint URL.
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = Some(endpoint.into());
        self
    }

    /// Set the API version.
    pub fn with_api_version(mut self, version: impl Into<String>) -> Self {
        self.api_version = Some(version.into());
        self
    }

    /// Set the deployment name (Azure).
    pub fn with_deployment(mut self, deployment: impl Into<String>) -> Self {
        self.deployment = Some(deployment.into());
        self
    }

    /// Set the project ID (GCP).
    pub fn with_project(mut self, project: impl Into<String>) -> Self {
        self.project = Some(project.into());
        self
    }

    /// Set the region.
    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.region = Some(region.into());
        self
    }

    /// Set the compartment ID (OCI).
    pub fn with_compartment_id(mut self, id: impl Into<String>) -> Self {
        self.compartment_id = Some(id.into());
        self
    }

    /// Set the organization ID (OpenAI).
    pub fn with_organization(mut self, org: impl Into<String>) -> Self {
        self.organization = Some(org.into());
        self
    }

    /// Set the request timeout.
    pub fn with_request_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = timeout;
        self
    }

    /// Set the connection timeout.
    pub fn with_connect_timeout(mut self, timeout: Duration) -> Self {
        self.connect_timeout = timeout;
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), ConfigValidationError> {
        if self.model.is_empty() {
            return Err(ConfigValidationError::MissingField("model"));
        }

        // Validate timeouts (1s to 1h for request, 1s to 5m for connect)
        if self.request_timeout < Duration::from_secs(1)
            || self.request_timeout > Duration::from_secs(3600)
        {
            return Err(ConfigValidationError::InvalidTimeout(self.request_timeout));
        }
        if self.connect_timeout < Duration::from_secs(1)
            || self.connect_timeout > Duration::from_secs(300)
        {
            return Err(ConfigValidationError::InvalidTimeout(self.connect_timeout));
        }

        match self.vendor {
            Vendor::AzureOpenAI => {
                if self.endpoint.is_none() {
                    return Err(ConfigValidationError::MissingField("endpoint"));
                }
                if self.deployment.is_none() {
                    return Err(ConfigValidationError::MissingField("deployment"));
                }
            }
            Vendor::GcpVertex => {
                if self.project.is_none() {
                    return Err(ConfigValidationError::MissingField("project"));
                }
                if self.region.is_none() {
                    return Err(ConfigValidationError::MissingField("region"));
                }
            }
            Vendor::OciGenAI | Vendor::OciCohere => {
                if self.compartment_id.is_none() {
                    return Err(ConfigValidationError::MissingField("compartment_id"));
                }
            }
            Vendor::AwsBedrock => {
                if self.region.is_none() {
                    return Err(ConfigValidationError::MissingField("region"));
                }
            }
            Vendor::OpenAI | Vendor::Cohere | Vendor::Together => {}
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vendor_display() {
        assert_eq!(Vendor::OpenAI.to_string(), "OpenAI");
        assert_eq!(Vendor::AwsBedrock.to_string(), "AWS Bedrock");
        assert_eq!(Vendor::GcpVertex.to_string(), "GCP Vertex AI");
    }

    #[test]
    fn test_vendor_id() {
        assert_eq!(Vendor::OpenAI.id(), "openai");
        assert_eq!(Vendor::AzureOpenAI.id(), "azure_openai");
        assert_eq!(Vendor::AwsBedrock.id(), "aws_bedrock");
    }

    #[test]
    fn test_vendor_from_str() {
        assert_eq!("openai".parse::<Vendor>().unwrap(), Vendor::OpenAI);
        assert_eq!(
            "azure-openai".parse::<Vendor>().unwrap(),
            Vendor::AzureOpenAI
        );
        assert_eq!("bedrock".parse::<Vendor>().unwrap(), Vendor::AwsBedrock);
        assert_eq!("vertex".parse::<Vendor>().unwrap(), Vendor::GcpVertex);
        assert!("unknown".parse::<Vendor>().is_err());
    }

    #[test]
    fn test_vendor_stream_format() {
        assert_eq!(Vendor::OpenAI.default_stream_format(), StreamFormat::SSE);
        assert_eq!(
            Vendor::AwsBedrock.default_stream_format(),
            StreamFormat::EventStream
        );
        assert_eq!(
            Vendor::OciGenAI.default_stream_format(),
            StreamFormat::JsonLines
        );
    }

    #[test]
    fn test_vendor_serialization() {
        // Serialization
        let json = serde_json::to_string(&Vendor::OpenAI).unwrap();
        assert_eq!(json, "\"openai\"");

        let json = serde_json::to_string(&Vendor::AzureOpenAI).unwrap();
        assert_eq!(json, "\"azure_openai\"");

        let json = serde_json::to_string(&Vendor::AwsBedrock).unwrap();
        assert_eq!(json, "\"aws_bedrock\"");

        // Deserialization
        let vendor: Vendor = serde_json::from_str("\"openai\"").unwrap();
        assert_eq!(vendor, Vendor::OpenAI);

        let vendor: Vendor = serde_json::from_str("\"azure_openai\"").unwrap();
        assert_eq!(vendor, Vendor::AzureOpenAI);

        let vendor: Vendor = serde_json::from_str("\"aws_bedrock\"").unwrap();
        assert_eq!(vendor, Vendor::AwsBedrock);
    }

    #[test]
    fn test_stream_format_content_type() {
        assert_eq!(StreamFormat::SSE.content_type(), "text/event-stream");
        assert_eq!(
            StreamFormat::JsonLines.content_type(),
            "application/x-ndjson"
        );
    }

    #[test]
    fn test_stream_format_serialization() {
        // Serialization
        assert_eq!(
            serde_json::to_string(&StreamFormat::SSE).unwrap(),
            "\"sse\""
        );
        assert_eq!(
            serde_json::to_string(&StreamFormat::JsonLines).unwrap(),
            "\"json_lines\""
        );
        assert_eq!(
            serde_json::to_string(&StreamFormat::EventStream).unwrap(),
            "\"event_stream\""
        );

        // Deserialization
        assert_eq!(
            serde_json::from_str::<StreamFormat>("\"sse\"").unwrap(),
            StreamFormat::SSE
        );
        assert_eq!(
            serde_json::from_str::<StreamFormat>("\"json_lines\"").unwrap(),
            StreamFormat::JsonLines
        );
        assert_eq!(
            serde_json::from_str::<StreamFormat>("\"event_stream\"").unwrap(),
            StreamFormat::EventStream
        );
    }

    #[test]
    fn test_vendor_config_builder() {
        let config = VendorConfig::new(Vendor::OpenAI, "gpt-4")
            .with_endpoint("https://api.openai.com/v1")
            .with_organization("org-123")
            .with_request_timeout(Duration::from_secs(120));

        assert_eq!(config.vendor, Vendor::OpenAI);
        assert_eq!(config.model, "gpt-4");
        assert_eq!(
            config.endpoint,
            Some("https://api.openai.com/v1".to_string())
        );
        assert_eq!(config.organization, Some("org-123".to_string()));
        assert_eq!(config.request_timeout, Duration::from_secs(120));
    }

    #[test]
    fn test_vendor_config_serialization() {
        let config = VendorConfig::new(Vendor::AzureOpenAI, "gpt-4")
            .with_deployment("my-deployment")
            .with_api_version("2024-02-15-preview");

        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("\"vendor\":\"azure_openai\""));
        assert!(json.contains("\"deployment\":\"my-deployment\""));
        assert!(json.contains("\"model\":\"gpt-4\""));

        let parsed: VendorConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.vendor, Vendor::AzureOpenAI);
        assert_eq!(parsed.model, "gpt-4");
        assert_eq!(parsed.deployment, Some("my-deployment".to_string()));
        assert_eq!(parsed.api_version, Some("2024-02-15-preview".to_string()));
    }

    #[test]
    fn test_vendor_all() {
        let all = Vendor::all();
        assert_eq!(all.len(), 8);
        assert!(all.contains(&Vendor::OpenAI));
        assert!(all.contains(&Vendor::AwsBedrock));
    }

    #[test]
    fn test_vendor_config_validate() {
        // Valid configs
        assert!(VendorConfig::new(Vendor::OpenAI, "gpt-4")
            .validate()
            .is_ok());
        assert!(VendorConfig::new(Vendor::AzureOpenAI, "gpt-4")
            .with_endpoint("https://x.openai.azure.com")
            .with_deployment("dep")
            .validate()
            .is_ok());
        assert!(VendorConfig::new(Vendor::GcpVertex, "gemini")
            .with_project("proj")
            .with_region("us-central1")
            .validate()
            .is_ok());
        assert!(VendorConfig::new(Vendor::AwsBedrock, "claude")
            .with_region("us-east-1")
            .validate()
            .is_ok());
        assert!(VendorConfig::new(Vendor::OciGenAI, "cmd")
            .with_compartment_id("ocid")
            .validate()
            .is_ok());

        // Missing fields
        assert!(VendorConfig::new(Vendor::OpenAI, "").validate().is_err());
        assert!(VendorConfig::new(Vendor::AzureOpenAI, "gpt-4")
            .validate()
            .is_err());
        assert!(VendorConfig::new(Vendor::GcpVertex, "gemini")
            .validate()
            .is_err());
        assert!(VendorConfig::new(Vendor::AwsBedrock, "claude")
            .validate()
            .is_err());
        assert!(VendorConfig::new(Vendor::OciGenAI, "cmd")
            .validate()
            .is_err());

        // Invalid timeouts
        assert!(VendorConfig::new(Vendor::OpenAI, "gpt-4")
            .with_request_timeout(Duration::ZERO)
            .validate()
            .is_err());
        assert!(VendorConfig::new(Vendor::OpenAI, "gpt-4")
            .with_request_timeout(Duration::from_secs(7200))
            .validate()
            .is_err());
    }
}
