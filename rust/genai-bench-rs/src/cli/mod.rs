//! CLI argument parsing and command handling

use crate::metrics::MetricsCollector;
use crate::output::{CsvExporter, ExcelExporter, JsonExporter};
use crate::providers::openai::OpenAIProvider;
use crate::runner::{BenchmarkConfig, BenchmarkRunner};
use crate::sampling::PromptSampler;
use crate::scenarios::{DeterministicScenario, NormalScenario, Scenario, UniformScenario};
use crate::visualization::{HistogramPlotter, PercentilePlotter, ThroughputPlotter};
use anyhow::{Context, Result};
use clap::Parser;
use std::path::Path;

/// GenAI Bench - High-performance LLM benchmarking tool
#[derive(Parser, Debug)]
#[command(name = "genai-bench")]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    /// Provider to benchmark (openai, azure, gcp, aws, anthropic)
    #[arg(short, long, default_value = "openai")]
    pub provider: String,

    /// API key for the provider
    #[arg(short = 'k', long, env = "API_KEY")]
    pub api_key: String,

    /// Base URL for the API
    #[arg(short, long, default_value = "https://api.openai.com/v1")]
    pub base_url: String,

    /// Model name
    #[arg(short, long, default_value = "gpt-3.5-turbo")]
    pub model: String,

    /// Number of requests to send
    #[arg(short, long, default_value = "100")]
    pub num_requests: usize,

    /// Traffic scenario (e.g., "D(100,100)" for deterministic, "N(100,10)" for normal, "U(50,150)" for uniform)
    #[arg(short, long, default_value = "D(0,0)")]
    pub scenario: String,

    /// Concurrency level (number of concurrent requests)
    #[arg(short, long, default_value = "1")]
    pub concurrency: usize,

    /// Path to dataset file (text file with one prompt per line)
    #[arg(short, long)]
    pub dataset_path: Option<String>,

    /// Default prompt to use if no dataset is provided
    #[arg(long, default_value = "Hello, how are you today?")]
    pub default_prompt: String,

    /// Maximum tokens in response
    #[arg(long, default_value = "100")]
    pub max_tokens: u32,

    /// Temperature for sampling (0.0-2.0)
    #[arg(long, default_value = "1.0")]
    pub temperature: f32,

    /// Enable streaming mode (important for accurate TTFT measurement)
    #[arg(long)]
    pub stream: bool,

    /// Output directory for results
    #[arg(long, default_value = "results")]
    pub output_dir: String,

    /// Export results to Excel (.xlsx)
    #[arg(long)]
    pub excel: bool,

    /// Export results to CSV
    #[arg(long)]
    pub csv: bool,

    /// Export results to JSON
    #[arg(long)]
    pub json: bool,

    /// Generate plots (histograms, throughput, percentiles)
    #[arg(long)]
    pub plot: bool,

    /// Enable live dashboard (TUI mode)
    #[arg(long)]
    pub ui: bool,

    /// Enable verbose logging
    #[arg(short, long)]
    pub verbose: bool,
}

impl Cli {
    /// Run the benchmark based on CLI arguments
    pub async fn run(&self) -> Result<()> {
        tracing::info!("Starting GenAI Bench");
        tracing::info!("Provider: {}", self.provider);
        tracing::info!("Model: {}", self.model);
        tracing::info!("Requests: {}", self.num_requests);
        tracing::info!("Concurrency: {}", self.concurrency);

        // Print banner
        println!("\n{}", "=".repeat(70));
        println!("   GenAI Bench - High-Performance LLM Benchmarking");
        println!("{}", "=".repeat(70));
        println!();
        println!("Configuration:");
        println!("  Provider:     {}", self.provider);
        println!("  Model:        {}", self.model);
        println!("  Requests:     {}", self.num_requests);
        println!("  Concurrency:  {}", self.concurrency);
        println!("  Scenario:     {}", self.scenario);
        println!("  Streaming:    {}", if self.stream { "enabled" } else { "disabled" });
        println!("{}", "=".repeat(70));
        println!();

        // 1. Create provider
        let provider = match self.provider.as_str() {
            "openai" => OpenAIProvider::new(self.api_key.clone(), self.base_url.clone()),
            _ => anyhow::bail!("Unsupported provider: {}. Currently only 'openai' is supported.", self.provider),
        };

        // 2. Create sampler
        let sampler = if let Some(ref path) = self.dataset_path {
            PromptSampler::from_file(Path::new(path))
                .with_context(|| format!("Failed to load dataset from: {}", path))?
        } else {
            PromptSampler::from_prompt(self.default_prompt.clone())
        };

        tracing::info!("Loaded {} prompts", sampler.len());

        // 3. Parse scenario
        let scenario = self.parse_scenario()
            .with_context(|| format!("Failed to parse scenario: {}", self.scenario))?;

        tracing::info!("Using scenario: {}", scenario.name());

        // 4. Create benchmark config
        let config = BenchmarkConfig {
            num_requests: self.num_requests,
            model: self.model.clone(),
            max_tokens: Some(self.max_tokens),
            temperature: Some(self.temperature),
            stream: self.stream,
        };

        // 5. Create runner
        let runner = BenchmarkRunner::new(provider, sampler, scenario, config);

        // 6. Run benchmark
        let collector = if self.ui {
            // UI mode - run with live dashboard
            use crate::ui;
            use tokio::sync::mpsc;

            let (tx, rx) = mpsc::channel(100);
            let num_requests = self.num_requests;

            // Spawn UI task
            let ui_handle = tokio::spawn(async move {
                ui::run_dashboard(rx, num_requests).await
            });

            // Run benchmark with UI updates
            let collector = if self.concurrency > 1 {
                runner.run_concurrent_with_ui(self.concurrency, tx).await?
            } else {
                // For sequential, use concurrent with concurrency=1
                runner.run_concurrent_with_ui(1, tx).await?
            };

            // Wait for UI to finish
            ui_handle.await??;

            collector
        } else {
            // Non-UI mode - use progress bars
            println!("Starting benchmark...\n");

            let collector = if self.concurrency > 1 {
                runner.run_concurrent(self.concurrency).await?
            } else {
                runner.run_sequential().await?
            };

            // Print results
            println!();
            self.print_results(&collector)?;

            collector
        };

        // 7. Export results if requested
        if self.excel || self.csv || self.json || self.plot {
            std::fs::create_dir_all(&self.output_dir)
                .with_context(|| format!("Failed to create output directory: {}", self.output_dir))?;

            if !self.ui {
                println!("\n{}", "=".repeat(70));
                println!("   Exporting Results");
                println!("{}", "=".repeat(70));
            }
        }

        if self.excel {
            let path = format!("{}/results.xlsx", self.output_dir);
            ExcelExporter::export(&collector, &path)
                .with_context(|| format!("Failed to export Excel to: {}", path))?;
            println!("✓ Excel exported to: {}", path);
        }

        if self.csv {
            let path = format!("{}/results.csv", self.output_dir);
            CsvExporter::export(&collector, &path)
                .with_context(|| format!("Failed to export CSV to: {}", path))?;
            println!("✓ CSV exported to: {}", path);

            // Also export summary
            let summary_path = format!("{}/summary.csv", self.output_dir);
            CsvExporter::export_summary(&collector, &summary_path)
                .with_context(|| format!("Failed to export CSV summary to: {}", summary_path))?;
            println!("✓ CSV summary exported to: {}", summary_path);
        }

        if self.json {
            let path = format!("{}/results.json", self.output_dir);
            JsonExporter::export(&collector, &path)
                .with_context(|| format!("Failed to export JSON to: {}", path))?;
            println!("✓ JSON exported to: {}", path);
        }

        if self.plot {
            println!("\nGenerating plots...");

            let ttft_path = format!("{}/ttft_histogram.png", self.output_dir);
            HistogramPlotter::plot_ttft(&collector, &ttft_path)
                .with_context(|| format!("Failed to generate TTFT histogram: {}", ttft_path))?;
            println!("✓ TTFT histogram: {}", ttft_path);

            let total_path = format!("{}/total_time_histogram.png", self.output_dir);
            HistogramPlotter::plot_total_time(&collector, &total_path)
                .with_context(|| format!("Failed to generate total time histogram: {}", total_path))?;
            println!("✓ Total time histogram: {}", total_path);

            let tokens_path = format!("{}/tokens_histogram.png", self.output_dir);
            HistogramPlotter::plot_tokens(&collector, &tokens_path)
                .with_context(|| format!("Failed to generate tokens histogram: {}", tokens_path))?;
            println!("✓ Tokens histogram: {}", tokens_path);

            let throughput_path = format!("{}/throughput.png", self.output_dir);
            ThroughputPlotter::plot(&collector, &throughput_path)
                .with_context(|| format!("Failed to generate throughput plot: {}", throughput_path))?;
            println!("✓ Throughput plot: {}", throughput_path);

            let rps_path = format!("{}/rps.png", self.output_dir);
            ThroughputPlotter::plot_rps(&collector, &rps_path, 1000) // 1 second windows
                .with_context(|| format!("Failed to generate RPS plot: {}", rps_path))?;
            println!("✓ RPS plot: {}", rps_path);

            let percentiles_path = format!("{}/percentiles.png", self.output_dir);
            PercentilePlotter::plot(&collector, &percentiles_path)
                .with_context(|| format!("Failed to generate percentiles chart: {}", percentiles_path))?;
            println!("✓ Percentiles chart: {}", percentiles_path);

            let cdf_path = format!("{}/cdf.png", self.output_dir);
            PercentilePlotter::plot_cdf(&collector, &cdf_path)
                .with_context(|| format!("Failed to generate CDF chart: {}", cdf_path))?;
            println!("✓ CDF chart: {}", cdf_path);
        }

        if self.excel || self.csv || self.json || self.plot {
            println!("{}", "=".repeat(70));
            println!();
        }

        Ok(())
    }

    /// Parse scenario string like "D(100,100)" or "N(100,10)" or "U(50,150)"
    fn parse_scenario(&self) -> Result<Box<dyn Scenario>> {
        let scenario_str = self.scenario.trim();

        if scenario_str.starts_with("D(") || scenario_str.starts_with("d(") {
            // Deterministic: D(delay_ms)
            let params = self.extract_params(scenario_str)?;
            if params.len() != 2 {
                anyhow::bail!("Deterministic scenario requires 2 parameters: D(delay_ms,delay_ms), got: {}", scenario_str);
            }
            let delay_ms = params[0].parse::<u64>()
                .with_context(|| format!("Invalid delay value: {}", params[0]))?;
            Ok(Box::new(DeterministicScenario::new(delay_ms)))
        } else if scenario_str.starts_with("N(") || scenario_str.starts_with("n(") {
            // Normal: N(mean_ms, std_dev_ms)
            let params = self.extract_params(scenario_str)?;
            if params.len() != 2 {
                anyhow::bail!("Normal scenario requires 2 parameters: N(mean_ms,std_dev_ms), got: {}", scenario_str);
            }
            let mean = params[0].parse::<f64>()
                .with_context(|| format!("Invalid mean value: {}", params[0]))?;
            let std_dev = params[1].parse::<f64>()
                .with_context(|| format!("Invalid std_dev value: {}", params[1]))?;
            Ok(Box::new(NormalScenario::new(mean, std_dev)?))
        } else if scenario_str.starts_with("U(") || scenario_str.starts_with("u(") {
            // Uniform: U(min_ms, max_ms)
            let params = self.extract_params(scenario_str)?;
            if params.len() != 2 {
                anyhow::bail!("Uniform scenario requires 2 parameters: U(min_ms,max_ms), got: {}", scenario_str);
            }
            let min = params[0].parse::<u64>()
                .with_context(|| format!("Invalid min value: {}", params[0]))?;
            let max = params[1].parse::<u64>()
                .with_context(|| format!("Invalid max value: {}", params[1]))?;
            Ok(Box::new(UniformScenario::new(min, max)))
        } else {
            anyhow::bail!(
                "Invalid scenario format: {}. Expected D(delay,delay), N(mean,std), or U(min,max)",
                scenario_str
            );
        }
    }

    /// Extract parameters from scenario string like "D(100,100)" -> ["100", "100"]
    fn extract_params(&self, scenario_str: &str) -> Result<Vec<String>> {
        let start = scenario_str.find('(')
            .ok_or_else(|| anyhow::anyhow!("Missing opening parenthesis in scenario: {}", scenario_str))?;
        let end = scenario_str.rfind(')')
            .ok_or_else(|| anyhow::anyhow!("Missing closing parenthesis in scenario: {}", scenario_str))?;

        if start >= end {
            anyhow::bail!("Invalid parentheses in scenario: {}", scenario_str);
        }

        let params_str = &scenario_str[start + 1..end];
        let params: Vec<String> = params_str
            .split(',')
            .map(|s| s.trim().to_string())
            .collect();

        Ok(params)
    }

    /// Print benchmark results in a nice format
    fn print_results(&self, collector: &MetricsCollector) -> Result<()> {
        let agg = collector.aggregate();

        println!("{}", "=".repeat(70));
        println!("   Benchmark Results");
        println!("{}", "=".repeat(70));
        println!();
        println!("📊 Overall Statistics:");
        println!("  Total Requests:       {}", agg.total_requests);
        println!("  Successful:           {} ({:.1}%)",
            agg.successful_requests,
            (agg.successful_requests as f64 / agg.total_requests as f64) * 100.0
        );
        println!("  Failed:               {}", agg.failed_requests);
        println!();

        println!("⏱️  Time to First Token (TTFT):");
        println!("  Average:              {:.2} ms", agg.avg_ttft_ms);
        println!("  Median (P50):         {:.2} ms", agg.p50_ttft_ms);
        println!("  95th Percentile:      {:.2} ms", agg.p95_ttft_ms);
        println!("  99th Percentile:      {:.2} ms", agg.p99_ttft_ms);
        println!();

        println!("⏰ Total Request Time:");
        println!("  Average:              {:.2} ms", agg.avg_total_time_ms);
        println!("  Median (P50):         {:.2} ms", agg.p50_total_time_ms);
        println!("  95th Percentile:      {:.2} ms", agg.p95_total_time_ms);
        println!("  99th Percentile:      {:.2} ms", agg.p99_total_time_ms);
        println!();

        println!("🔢 Token Statistics:");
        println!("  Total Input Tokens:   {}", agg.total_prompt_tokens);
        println!("  Total Output Tokens:  {}", agg.total_completion_tokens);
        println!("  Total Tokens:         {}", agg.total_tokens);
        println!("  Avg Tokens/Second:    {:.2}", agg.avg_tokens_per_second);
        println!();

        println!("{}", "=".repeat(70));
        println!();

        Ok(())
    }
}
