//! Request sampling from datasets
//!
//! This module handles loading and sampling from datasets of prompts.
//! For the MVP, we'll support simple text files with one prompt per line.

use anyhow::{Context, Result};
use rand::seq::SliceRandom;
use std::fs;
use std::path::Path;

/// Simple sampler that loads prompts from a file
pub struct PromptSampler {
    prompts: Vec<String>,
}

impl PromptSampler {
    /// Create a new prompt sampler from a file
    pub fn from_file(path: &Path) -> Result<Self> {
        let content = fs::read_to_string(path)
            .with_context(|| format!("Failed to read prompt file: {}", path.display()))?;

        let prompts: Vec<String> = content
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| line.to_string())
            .collect();

        if prompts.is_empty() {
            anyhow::bail!("No prompts found in file: {}", path.display());
        }

        Ok(Self { prompts })
    }

    /// Create a sampler with a single prompt
    pub fn from_prompt(prompt: String) -> Self {
        Self {
            prompts: vec![prompt],
        }
    }

    /// Sample a random prompt
    pub fn sample(&self) -> &str {
        let mut rng = rand::thread_rng();
        self.prompts
            .choose(&mut rng)
            .expect("Prompts should not be empty")
    }

    /// Get a specific prompt by index
    pub fn get(&self, index: usize) -> Option<&str> {
        self.prompts.get(index).map(|s| s.as_str())
    }

    /// Get the number of prompts
    pub fn len(&self) -> usize {
        self.prompts.len()
    }

    /// Check if the sampler has no prompts
    pub fn is_empty(&self) -> bool {
        self.prompts.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_prompt_sampler_from_prompt() {
        let sampler = PromptSampler::from_prompt("Hello, world!".to_string());
        assert_eq!(sampler.len(), 1);
        assert_eq!(sampler.sample(), "Hello, world!");
    }

    #[test]
    fn test_prompt_sampler_from_file() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "Prompt 1").unwrap();
        writeln!(file, "Prompt 2").unwrap();
        writeln!(file, "Prompt 3").unwrap();

        let sampler = PromptSampler::from_file(file.path()).unwrap();
        assert_eq!(sampler.len(), 3);

        // Verify we can sample
        let sample = sampler.sample();
        assert!(
            sample == "Prompt 1" || sample == "Prompt 2" || sample == "Prompt 3"
        );
    }

    #[test]
    fn test_prompt_sampler_get() {
        let sampler = PromptSampler::from_prompt("Test".to_string());
        assert_eq!(sampler.get(0), Some("Test"));
        assert_eq!(sampler.get(1), None);
    }
}
