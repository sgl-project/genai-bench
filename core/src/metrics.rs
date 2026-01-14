//! Metrics aggregation and percentile calculation

use crate::request::{RequestId, Task};
use crate::response::ResponseStatus;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

/// Aggregated metrics for an experiment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentMetrics {
    /// Experiment metadata
    pub metadata: ExperimentMetadata,

    /// Summary statistics
    pub summary: MetricsSummary,

    /// Per-scenario breakdown
    pub scenarios: HashMap<String, ScenarioMetrics>,

    /// Time-series data (for plotting)
    pub timeseries: Vec<TimeseriesPoint>,

    /// Individual request records
    pub requests: Vec<RequestRecord>,
}

impl ExperimentMetrics {
    /// Create new metrics with metadata
    pub fn new(metadata: ExperimentMetadata) -> Self {
        Self {
            metadata,
            summary: MetricsSummary::default(),
            scenarios: HashMap::new(),
            timeseries: Vec::new(),
            requests: Vec::new(),
        }
    }
}

/// Experiment metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentMetadata {
    /// Unique experiment identifier
    pub experiment_id: String,
    /// Vendor name (e.g., "openai", "azure")
    pub vendor: String,
    /// Model name
    pub model: String,
    /// Task type
    pub task: Task,
    /// When the experiment started
    pub start_time: chrono::DateTime<chrono::Utc>,
    /// When the experiment ended
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_time: Option<chrono::DateTime<chrono::Utc>>,
    /// Experiment configuration
    pub config: ExperimentConfig,
}

/// Experiment configuration (subset of full config relevant for metrics)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentConfig {
    /// Concurrency level
    pub concurrency: usize,
    /// Number of requests to run
    pub num_requests: usize,
    /// Duration limit in seconds
    pub duration_secs: Option<u64>,
    /// Additional configuration
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub extra: HashMap<String, serde_json::Value>,
}

impl Default for ExperimentConfig {
    fn default() -> Self {
        Self {
            concurrency: 1,
            num_requests: 100,
            duration_secs: None,
            extra: HashMap::new(),
        }
    }
}

/// Summary statistics for an experiment
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MetricsSummary {
    /// Total requests made
    pub total_requests: usize,
    /// Successful requests
    pub successful_requests: usize,
    /// Failed requests
    pub failed_requests: usize,
    /// Error rate (0.0 - 1.0)
    pub error_rate: f64,

    /// Total input tokens
    pub total_input_tokens: usize,
    /// Total output tokens
    pub total_output_tokens: usize,

    /// Requests per second
    pub requests_per_second: f64,
    /// Input tokens per second
    pub input_tokens_per_second: f64,
    /// Output tokens per second
    pub output_tokens_per_second: f64,

    /// Time to first token percentiles (milliseconds)
    pub ttft: LatencyPercentiles,
    /// Time per output token percentiles (milliseconds)
    pub tpot: LatencyPercentiles,
    /// End-to-end latency percentiles (milliseconds)
    pub e2e: LatencyPercentiles,

    /// Total duration in seconds
    pub total_duration_secs: f64,
}

impl MetricsSummary {
    /// Calculate summary from request records
    pub fn from_records(records: &[RequestRecord], duration: Duration) -> Self {
        let total_requests = records.len();
        let successful_requests = records.iter().filter(|r| r.status.is_success()).count();
        let failed_requests = total_requests - successful_requests;
        let error_rate = if total_requests > 0 {
            failed_requests as f64 / total_requests as f64
        } else {
            0.0
        };

        let total_input_tokens: usize = records.iter().map(|r| r.input_tokens).sum();
        let total_output_tokens: usize = records.iter().map(|r| r.output_tokens).sum();

        let duration_secs = duration.as_secs_f64();
        let requests_per_second = if duration_secs > 0.0 {
            total_requests as f64 / duration_secs
        } else {
            0.0
        };
        let input_tokens_per_second = if duration_secs > 0.0 {
            total_input_tokens as f64 / duration_secs
        } else {
            0.0
        };
        let output_tokens_per_second = if duration_secs > 0.0 {
            total_output_tokens as f64 / duration_secs
        } else {
            0.0
        };

        // Calculate latency percentiles
        let ttft_values: Vec<f64> = records.iter().filter_map(|r| r.ttft_ms).collect();
        let tpot_values: Vec<f64> = records.iter().filter_map(|r| r.tpot_ms).collect();
        let e2e_values: Vec<f64> = records.iter().map(|r| r.e2e_ms).collect();

        Self {
            total_requests,
            successful_requests,
            failed_requests,
            error_rate,
            total_input_tokens,
            total_output_tokens,
            requests_per_second,
            input_tokens_per_second,
            output_tokens_per_second,
            ttft: LatencyPercentiles::from_values(&ttft_values),
            tpot: LatencyPercentiles::from_values(&tpot_values),
            e2e: LatencyPercentiles::from_values(&e2e_values),
            total_duration_secs: duration_secs,
        }
    }
}

/// Latency percentiles (all values in milliseconds)
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct LatencyPercentiles {
    /// Minimum value
    pub min: f64,
    /// 50th percentile (median)
    pub p50: f64,
    /// 75th percentile
    pub p75: f64,
    /// 90th percentile
    pub p90: f64,
    /// 95th percentile
    pub p95: f64,
    /// 99th percentile
    pub p99: f64,
    /// Maximum value
    pub max: f64,
    /// Mean value
    pub mean: f64,
    /// Standard deviation
    pub stddev: f64,
}

impl LatencyPercentiles {
    /// Calculate percentiles from a slice of values
    pub fn from_values(values: &[f64]) -> Self {
        if values.is_empty() {
            return Self::default();
        }

        let mut sorted: Vec<f64> = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let len = sorted.len();
        let mean = sorted.iter().sum::<f64>() / len as f64;

        let variance = if len > 1 {
            sorted.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (len - 1) as f64
        } else {
            0.0
        };
        let stddev = variance.sqrt();

        Self {
            min: sorted[0],
            p50: percentile(&sorted, 0.50),
            p75: percentile(&sorted, 0.75),
            p90: percentile(&sorted, 0.90),
            p95: percentile(&sorted, 0.95),
            p99: percentile(&sorted, 0.99),
            max: sorted[len - 1],
            mean,
            stddev,
        }
    }
}

/// Calculate percentile from sorted values using linear interpolation
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }

    let idx = p * (sorted.len() - 1) as f64;
    let lower = idx.floor() as usize;
    let upper = idx.ceil() as usize;
    let frac = idx - lower as f64;

    if upper >= sorted.len() {
        sorted[sorted.len() - 1]
    } else {
        sorted[lower] * (1.0 - frac) + sorted[upper] * frac
    }
}

/// In-memory histogram for efficient percentile calculation
/// Uses HdrHistogram for memory-efficient storage of large datasets
pub struct LatencyHistogram {
    histogram: hdrhistogram::Histogram<u64>,
}

impl LatencyHistogram {
    /// Create a new histogram
    /// Configured for microsecond precision with max 1 hour latency
    pub fn new() -> Self {
        // Histogram with microsecond precision, max 1 hour (3,600,000,000 microseconds)
        let histogram = hdrhistogram::Histogram::new_with_bounds(1, 3_600_000_000, 3)
            .expect("Failed to create histogram");
        Self { histogram }
    }

    /// Record a duration
    pub fn record(&mut self, duration: Duration) {
        let micros = duration.as_micros() as u64;
        let _ = self.histogram.record(micros);
    }

    /// Record a value in milliseconds
    pub fn record_ms(&mut self, ms: f64) {
        let micros = (ms * 1000.0) as u64;
        let _ = self.histogram.record(micros);
    }

    /// Get the number of recorded values
    pub fn len(&self) -> u64 {
        self.histogram.len()
    }

    /// Check if the histogram is empty
    pub fn is_empty(&self) -> bool {
        self.histogram.is_empty()
    }

    /// Calculate percentiles from the histogram
    pub fn percentiles(&self) -> LatencyPercentiles {
        if self.histogram.is_empty() {
            return LatencyPercentiles::default();
        }

        LatencyPercentiles {
            min: self.histogram.min() as f64 / 1000.0,
            p50: self.histogram.value_at_quantile(0.50) as f64 / 1000.0,
            p75: self.histogram.value_at_quantile(0.75) as f64 / 1000.0,
            p90: self.histogram.value_at_quantile(0.90) as f64 / 1000.0,
            p95: self.histogram.value_at_quantile(0.95) as f64 / 1000.0,
            p99: self.histogram.value_at_quantile(0.99) as f64 / 1000.0,
            max: self.histogram.max() as f64 / 1000.0,
            mean: self.histogram.mean() / 1000.0,
            stddev: self.histogram.stdev() / 1000.0,
        }
    }

    /// Reset the histogram
    pub fn reset(&mut self) {
        self.histogram.reset();
    }
}

impl Default for LatencyHistogram {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-scenario metrics breakdown
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ScenarioMetrics {
    /// Scenario name
    pub scenario_name: String,
    /// Number of requests
    pub request_count: usize,
    /// Number of successful requests
    pub success_count: usize,
    /// Number of failed requests
    pub error_count: usize,
    /// Total input tokens
    pub input_tokens: usize,
    /// Total output tokens
    pub output_tokens: usize,
    /// Time to first token percentiles
    pub ttft: LatencyPercentiles,
    /// Time per output token percentiles
    pub tpot: LatencyPercentiles,
    /// End-to-end latency percentiles
    pub e2e: LatencyPercentiles,
}

/// Time-series data point for plotting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeseriesPoint {
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Cumulative requests completed
    pub requests_completed: usize,
    /// Requests per second at this point
    pub requests_per_second: f64,
    /// Tokens per second at this point
    pub tokens_per_second: f64,
    /// Average TTFT at this point (ms)
    pub avg_ttft_ms: f64,
    /// Average TPOT at this point (ms)
    pub avg_tpot_ms: f64,
    /// Cumulative error count
    pub error_count: usize,
}

/// Individual request record for detailed analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestRecord {
    /// Request ID
    pub request_id: RequestId,
    /// Scenario name
    pub scenario: String,
    /// Input tokens
    pub input_tokens: usize,
    /// Output tokens
    pub output_tokens: usize,
    /// Time to first token (ms)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttft_ms: Option<f64>,
    /// Time per output token (ms)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tpot_ms: Option<f64>,
    /// End-to-end latency (ms)
    pub e2e_ms: f64,
    /// Response status
    pub status: ResponseStatus,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

// ============================================================================
// Running Counters
// ============================================================================

/// Running counters for metrics aggregation
#[derive(Debug, Clone, Default)]
pub struct MetricsCounters {
    /// Total requests processed
    pub total_requests: usize,
    /// Successful requests
    pub successful_requests: usize,
    /// Failed requests
    pub failed_requests: usize,
    /// Total input tokens processed
    pub total_input_tokens: usize,
    /// Total output tokens generated
    pub total_output_tokens: usize,
    /// Error counts by type
    pub error_counts: HashMap<crate::error::ErrorKind, usize>,
}

impl MetricsCounters {
    /// Create new counters
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the success rate (0.0 - 1.0)
    pub fn success_rate(&self) -> f64 {
        if self.total_requests > 0 {
            self.successful_requests as f64 / self.total_requests as f64
        } else {
            0.0
        }
    }

    /// Get the error rate (0.0 - 1.0)
    pub fn error_rate(&self) -> f64 {
        if self.total_requests > 0 {
            self.failed_requests as f64 / self.total_requests as f64
        } else {
            0.0
        }
    }
}

// ============================================================================
// UI Update Message
// ============================================================================

/// UI update message sent periodically to the dashboard
#[derive(Debug, Clone)]
pub struct UIUpdate {
    /// Total requests completed
    pub requests_completed: usize,
    /// Current requests per second
    pub requests_per_second: f64,
    /// Current output tokens per second
    pub tokens_per_second: f64,
    /// Median time to first token (ms)
    pub ttft_p50: f64,
    /// Median time per output token (ms)
    pub tpot_p50: f64,
    /// Total error count
    pub error_count: usize,
    /// Elapsed time since start
    pub elapsed: Duration,
    /// Estimated remaining time (if determinable)
    pub estimated_remaining: Option<Duration>,
}

impl Default for UIUpdate {
    fn default() -> Self {
        Self {
            requests_completed: 0,
            requests_per_second: 0.0,
            tokens_per_second: 0.0,
            ttft_p50: 0.0,
            tpot_p50: 0.0,
            error_count: 0,
            elapsed: Duration::ZERO,
            estimated_remaining: None,
        }
    }
}

impl UIUpdate {
    /// Create a new UI update
    pub fn new() -> Self {
        Self::default()
    }

    /// Calculate progress percentage (0.0 - 1.0) if determinable
    pub fn progress(&self, total_requests: Option<usize>) -> Option<f64> {
        total_requests.map(|total| {
            if total > 0 {
                (self.requests_completed as f64 / total as f64).min(1.0)
            } else {
                0.0
            }
        })
    }

    /// Calculate success rate (0.0 - 1.0)
    pub fn success_rate(&self) -> f64 {
        let total = self.requests_completed + self.error_count;
        if total > 0 {
            self.requests_completed as f64 / total as f64
        } else {
            1.0 // No failures yet
        }
    }
}

// ============================================================================
// Metrics Aggregator
// ============================================================================

/// Per-scenario aggregation state
struct ScenarioState {
    counters: MetricsCounters,
    ttft_hist: LatencyHistogram,
    tpot_hist: LatencyHistogram,
    e2e_hist: LatencyHistogram,
}

impl ScenarioState {
    fn new() -> Self {
        Self {
            counters: MetricsCounters::default(),
            ttft_hist: LatencyHistogram::new(),
            tpot_hist: LatencyHistogram::new(),
            e2e_hist: LatencyHistogram::new(),
        }
    }
}

/// Aggregates metrics from workers and produces final experiment metrics
pub struct MetricsAggregator {
    /// Receiver for request records from workers
    rx: mpsc::Receiver<RequestRecord>,
    /// UI update sender (optional)
    ui_tx: Option<mpsc::Sender<UIUpdate>>,
    /// Histograms for percentile calculation
    ttft_hist: LatencyHistogram,
    tpot_hist: LatencyHistogram,
    e2e_hist: LatencyHistogram,
    /// Running counters
    counters: MetricsCounters,
    /// Per-scenario aggregation state
    scenario_states: HashMap<String, ScenarioState>,
    /// Request records (for final report)
    records: Vec<RequestRecord>,
    /// Time-series buffer
    timeseries: Vec<TimeseriesPoint>,
    /// Last UI update time
    last_ui_update: Instant,
    /// Last timeseries recording time
    last_timeseries: Instant,
    /// Start time (monotonic)
    start_time: Instant,
    /// Start time (wall clock)
    start_time_utc: chrono::DateTime<chrono::Utc>,
    /// Target request count (for progress calculation)
    target_requests: Option<usize>,
}

impl MetricsAggregator {
    /// Create a new metrics aggregator
    pub fn new(rx: mpsc::Receiver<RequestRecord>) -> Self {
        let now = Instant::now();
        Self {
            rx,
            ui_tx: None,
            ttft_hist: LatencyHistogram::new(),
            tpot_hist: LatencyHistogram::new(),
            e2e_hist: LatencyHistogram::new(),
            counters: MetricsCounters::default(),
            scenario_states: HashMap::new(),
            records: Vec::new(),
            timeseries: Vec::new(),
            last_ui_update: now,
            last_timeseries: now,
            start_time: now,
            start_time_utc: chrono::Utc::now(),
            target_requests: None,
        }
    }

    /// Calculate rate multiplier from elapsed seconds
    fn rate_multiplier(elapsed_secs: f64) -> f64 {
        if elapsed_secs > 0.0 {
            1.0 / elapsed_secs
        } else {
            0.0
        }
    }

    /// Set the UI update sender
    pub fn with_ui_sender(mut self, ui_tx: mpsc::Sender<UIUpdate>) -> Self {
        self.ui_tx = Some(ui_tx);
        self
    }

    /// Set the target request count for progress calculation
    pub fn with_target_requests(mut self, target: usize) -> Self {
        self.target_requests = Some(target);
        self
    }

    /// Run the aggregator until the channel closes
    pub async fn run(mut self) -> ExperimentMetrics {
        let ui_update_interval = Duration::from_millis(100);
        let timeseries_interval = Duration::from_secs(1);

        while let Some(record) = self.rx.recv().await {
            self.process_record(record);

            // Update UI periodically
            if self.last_ui_update.elapsed() >= ui_update_interval {
                self.send_ui_update().await;
            }

            // Record timeseries point
            if self.last_timeseries.elapsed() >= timeseries_interval {
                self.record_timeseries_point();
            }
        }

        // Send final UI update
        self.send_ui_update().await;

        self.build_final_metrics()
    }

    /// Process a single request record
    fn process_record(&mut self, record: RequestRecord) {
        // Update global counters
        self.counters.total_requests += 1;

        // Get or create scenario state
        let scenario_state = self
            .scenario_states
            .entry(record.scenario.clone())
            .or_insert_with(ScenarioState::new);
        scenario_state.counters.total_requests += 1;

        match &record.status {
            ResponseStatus::Success => {
                // Update global counters
                self.counters.successful_requests += 1;
                self.counters.total_input_tokens += record.input_tokens;
                self.counters.total_output_tokens += record.output_tokens;

                // Update scenario counters
                scenario_state.counters.successful_requests += 1;
                scenario_state.counters.total_input_tokens += record.input_tokens;
                scenario_state.counters.total_output_tokens += record.output_tokens;

                // Record latencies in global histograms
                if let Some(ttft) = record.ttft_ms {
                    self.ttft_hist.record_ms(ttft);
                    scenario_state.ttft_hist.record_ms(ttft);
                }
                if let Some(tpot) = record.tpot_ms {
                    self.tpot_hist.record_ms(tpot);
                    scenario_state.tpot_hist.record_ms(tpot);
                }
                self.e2e_hist.record_ms(record.e2e_ms);
                scenario_state.e2e_hist.record_ms(record.e2e_ms);
            }
            ResponseStatus::Error(kind) => {
                self.counters.failed_requests += 1;
                *self.counters.error_counts.entry(*kind).or_insert(0) += 1;

                scenario_state.counters.failed_requests += 1;
                *scenario_state
                    .counters
                    .error_counts
                    .entry(*kind)
                    .or_insert(0) += 1;
            }
        }

        // Store record for final report
        self.records.push(record);
    }

    /// Record a timeseries data point
    fn record_timeseries_point(&mut self) {
        let elapsed_secs = self.start_time.elapsed().as_secs_f64();
        let rate = Self::rate_multiplier(elapsed_secs);

        let point = TimeseriesPoint {
            timestamp: chrono::Utc::now(),
            requests_completed: self.counters.total_requests,
            requests_per_second: self.counters.total_requests as f64 * rate,
            tokens_per_second: self.counters.total_output_tokens as f64 * rate,
            avg_ttft_ms: self.ttft_hist.percentiles().mean,
            avg_tpot_ms: self.tpot_hist.percentiles().mean,
            error_count: self.counters.failed_requests,
        };

        self.timeseries.push(point);
        self.last_timeseries = Instant::now();
    }

    /// Send a UI update
    async fn send_ui_update(&mut self) {
        let Some(ui_tx) = &self.ui_tx else {
            self.last_ui_update = Instant::now();
            return;
        };

        let elapsed = self.start_time.elapsed();
        let elapsed_secs = elapsed.as_secs_f64();
        let rate = Self::rate_multiplier(elapsed_secs);
        let estimated_remaining = self.estimate_remaining_time(elapsed);

        let update = UIUpdate {
            requests_completed: self.counters.total_requests,
            requests_per_second: self.counters.total_requests as f64 * rate,
            tokens_per_second: self.counters.total_output_tokens as f64 * rate,
            ttft_p50: self.ttft_hist.percentiles().p50,
            tpot_p50: self.tpot_hist.percentiles().p50,
            error_count: self.counters.failed_requests,
            elapsed,
            estimated_remaining,
        };

        // Non-blocking send - drop if receiver is full
        let _ = ui_tx.try_send(update);
        self.last_ui_update = Instant::now();
    }

    /// Estimate remaining time based on progress
    fn estimate_remaining_time(&self, elapsed: Duration) -> Option<Duration> {
        let target = self.target_requests?;
        if target == 0 || self.counters.total_requests == 0 {
            return None;
        }

        let progress = self.counters.total_requests as f64 / target as f64;
        if progress >= 1.0 {
            return Some(Duration::ZERO);
        }

        let estimated_total = elapsed.as_secs_f64() / progress;
        let remaining = estimated_total - elapsed.as_secs_f64();

        if remaining > 0.0 {
            Some(Duration::from_secs_f64(remaining))
        } else {
            Some(Duration::ZERO)
        }
    }

    /// Build the final experiment metrics
    fn build_final_metrics(self) -> ExperimentMetrics {
        let elapsed = self.start_time.elapsed();
        let duration_secs = elapsed.as_secs_f64();
        let rate = Self::rate_multiplier(duration_secs);

        // Build summary from pre-aggregated counters and histograms
        let summary = MetricsSummary {
            total_requests: self.counters.total_requests,
            successful_requests: self.counters.successful_requests,
            failed_requests: self.counters.failed_requests,
            error_rate: self.counters.error_rate(),
            total_input_tokens: self.counters.total_input_tokens,
            total_output_tokens: self.counters.total_output_tokens,
            requests_per_second: self.counters.total_requests as f64 * rate,
            input_tokens_per_second: self.counters.total_input_tokens as f64 * rate,
            output_tokens_per_second: self.counters.total_output_tokens as f64 * rate,
            ttft: self.ttft_hist.percentiles(),
            tpot: self.tpot_hist.percentiles(),
            e2e: self.e2e_hist.percentiles(),
            total_duration_secs: duration_secs,
        };

        ExperimentMetrics {
            metadata: ExperimentMetadata {
                experiment_id: String::new(), // Set by caller
                vendor: String::new(),        // Set by caller
                model: String::new(),         // Set by caller
                task: crate::request::Task::TextToText,
                start_time: self.start_time_utc,
                end_time: Some(chrono::Utc::now()),
                config: ExperimentConfig::default(),
            },
            summary,
            scenarios: self.build_scenario_metrics(),
            timeseries: self.timeseries,
            requests: self.records,
        }
    }

    /// Build per-scenario metrics from pre-aggregated state
    fn build_scenario_metrics(&self) -> HashMap<String, ScenarioMetrics> {
        self.scenario_states
            .iter()
            .map(|(name, state)| {
                let metrics = ScenarioMetrics {
                    scenario_name: name.clone(),
                    request_count: state.counters.total_requests,
                    success_count: state.counters.successful_requests,
                    error_count: state.counters.failed_requests,
                    input_tokens: state.counters.total_input_tokens,
                    output_tokens: state.counters.total_output_tokens,
                    ttft: state.ttft_hist.percentiles(),
                    tpot: state.tpot_hist.percentiles(),
                    e2e: state.e2e_hist.percentiles(),
                };
                (name.clone(), metrics)
            })
            .collect()
    }
}

impl std::fmt::Debug for MetricsAggregator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetricsAggregator")
            .field("total_requests", &self.counters.total_requests)
            .field("records_count", &self.records.len())
            .field("timeseries_points", &self.timeseries.len())
            .finish()
    }
}

// ============================================================================
// TPOT Calculation Utility
// ============================================================================

/// Calculate Time Per Output Token (TPOT) from token timings
///
/// TPOT is calculated as the mean of inter-token intervals.
/// Requires at least 2 token timings to compute.
pub fn calculate_tpot(token_timings: &[Duration]) -> Option<Duration> {
    if token_timings.len() < 2 {
        return None;
    }

    // Calculate inter-token intervals
    // Note: intervals is guaranteed non-empty since token_timings.len() >= 2
    let intervals: Vec<Duration> = token_timings
        .windows(2)
        .map(|w| w[1].saturating_sub(w[0]))
        .collect();

    // Calculate mean interval
    let total: Duration = intervals.iter().sum();
    Some(total / intervals.len() as u32)
}

/// Calculate TPOT from millisecond timings
pub fn calculate_tpot_ms(token_timings_ms: &[f64]) -> Option<f64> {
    if token_timings_ms.len() < 2 {
        return None;
    }

    // Calculate inter-token intervals
    // Note: intervals is guaranteed non-empty since token_timings_ms.len() >= 2
    let intervals: Vec<f64> = token_timings_ms
        .windows(2)
        .map(|w| (w[1] - w[0]).max(0.0))
        .collect();

    // Calculate mean interval
    let sum: f64 = intervals.iter().sum();
    Some(sum / intervals.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_percentile_calculation() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let percentiles = LatencyPercentiles::from_values(&values);

        assert_eq!(percentiles.min, 1.0);
        assert_eq!(percentiles.max, 10.0);
        assert!((percentiles.p50 - 5.5).abs() < 0.01);
        assert!((percentiles.mean - 5.5).abs() < 0.01);
    }

    #[test]
    fn test_percentile_single_value() {
        let values = vec![42.0];
        let percentiles = LatencyPercentiles::from_values(&values);

        assert_eq!(percentiles.min, 42.0);
        assert_eq!(percentiles.max, 42.0);
        assert_eq!(percentiles.p50, 42.0);
        assert_eq!(percentiles.mean, 42.0);
        assert_eq!(percentiles.stddev, 0.0);
    }

    #[test]
    fn test_percentile_empty() {
        let values: Vec<f64> = vec![];
        let percentiles = LatencyPercentiles::from_values(&values);

        assert_eq!(percentiles.min, 0.0);
        assert_eq!(percentiles.max, 0.0);
        assert_eq!(percentiles.mean, 0.0);
    }

    #[test]
    fn test_histogram_percentiles() {
        let mut histogram = LatencyHistogram::new();

        for i in 1..=100 {
            histogram.record_ms(i as f64);
        }

        let percentiles = histogram.percentiles();
        assert!((percentiles.min - 1.0).abs() < 0.1);
        assert!((percentiles.max - 100.0).abs() < 0.1);
        assert!((percentiles.p50 - 50.0).abs() < 1.0);
    }

    #[test]
    fn test_empty_histogram() {
        let histogram = LatencyHistogram::new();
        let percentiles = histogram.percentiles();

        assert_eq!(percentiles.min, 0.0);
        assert_eq!(percentiles.max, 0.0);
        assert_eq!(percentiles.mean, 0.0);
    }

    #[test]
    fn test_histogram_record_duration() {
        let mut histogram = LatencyHistogram::new();
        histogram.record(Duration::from_millis(100));
        histogram.record(Duration::from_millis(200));

        assert_eq!(histogram.len(), 2);
        assert!(!histogram.is_empty());
    }

    #[test]
    fn test_metrics_summary_from_records() {
        let records = vec![
            RequestRecord {
                request_id: 1u64.into(),
                scenario: "test".to_string(),
                input_tokens: 100,
                output_tokens: 50,
                ttft_ms: Some(100.0),
                tpot_ms: Some(10.0),
                e2e_ms: 500.0,
                status: ResponseStatus::Success,
                timestamp: chrono::Utc::now(),
            },
            RequestRecord {
                request_id: 2u64.into(),
                scenario: "test".to_string(),
                input_tokens: 200,
                output_tokens: 100,
                ttft_ms: Some(150.0),
                tpot_ms: Some(15.0),
                e2e_ms: 750.0,
                status: ResponseStatus::Success,
                timestamp: chrono::Utc::now(),
            },
        ];

        let summary = MetricsSummary::from_records(&records, Duration::from_secs(10));

        assert_eq!(summary.total_requests, 2);
        assert_eq!(summary.successful_requests, 2);
        assert_eq!(summary.failed_requests, 0);
        assert_eq!(summary.error_rate, 0.0);
        assert_eq!(summary.total_input_tokens, 300);
        assert_eq!(summary.total_output_tokens, 150);
        assert!((summary.requests_per_second - 0.2).abs() < 0.01);
    }

    // =========================================================================
    // MetricsCounters Tests
    // =========================================================================

    #[test]
    fn test_metrics_counters_default() {
        let counters = MetricsCounters::new();
        assert_eq!(counters.total_requests, 0);
        assert_eq!(counters.successful_requests, 0);
        assert_eq!(counters.failed_requests, 0);
        assert_eq!(counters.success_rate(), 0.0);
        assert_eq!(counters.error_rate(), 0.0);
    }

    #[test]
    fn test_metrics_counters_rates() {
        let counters = MetricsCounters {
            total_requests: 100,
            successful_requests: 90,
            failed_requests: 10,
            ..Default::default()
        };
        assert!((counters.success_rate() - 0.9).abs() < 0.001);
        assert!((counters.error_rate() - 0.1).abs() < 0.001);
    }

    // =========================================================================
    // UIUpdate Tests
    // =========================================================================

    #[test]
    fn test_ui_update_default() {
        let update = UIUpdate::new();
        assert_eq!(update.requests_completed, 0);
        assert_eq!(update.requests_per_second, 0.0);
        assert_eq!(update.error_count, 0);
    }

    #[test]
    fn test_ui_update_progress() {
        let mut update = UIUpdate::new();
        update.requests_completed = 50;

        assert_eq!(update.progress(Some(100)), Some(0.5));
        assert_eq!(update.progress(Some(50)), Some(1.0));
        assert_eq!(update.progress(None), None);
    }

    #[test]
    fn test_ui_update_success_rate() {
        let mut update = UIUpdate::new();
        update.requests_completed = 90;
        update.error_count = 10;

        assert!((update.success_rate() - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_ui_update_success_rate_no_requests() {
        let update = UIUpdate::new();
        assert_eq!(update.success_rate(), 1.0);
    }

    // =========================================================================
    // TPOT Calculation Tests
    // =========================================================================

    #[test]
    fn test_calculate_tpot_basic() {
        let timings = vec![
            Duration::from_millis(0),
            Duration::from_millis(10),
            Duration::from_millis(20),
            Duration::from_millis(30),
        ];

        let tpot = calculate_tpot(&timings).unwrap();
        assert_eq!(tpot, Duration::from_millis(10));
    }

    #[test]
    fn test_calculate_tpot_single_value() {
        let timings = vec![Duration::from_millis(100)];
        assert!(calculate_tpot(&timings).is_none());
    }

    #[test]
    fn test_calculate_tpot_empty() {
        let timings: Vec<Duration> = vec![];
        assert!(calculate_tpot(&timings).is_none());
    }

    #[test]
    fn test_calculate_tpot_ms_basic() {
        let timings = vec![0.0, 10.0, 20.0, 30.0];
        let tpot = calculate_tpot_ms(&timings).unwrap();
        assert!((tpot - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_calculate_tpot_ms_varying_intervals() {
        let timings = vec![0.0, 10.0, 30.0, 60.0];
        // Intervals: 10, 20, 30 -> mean = 20
        let tpot = calculate_tpot_ms(&timings).unwrap();
        assert!((tpot - 20.0).abs() < 0.001);
    }

    // =========================================================================
    // MetricsAggregator Tests
    // =========================================================================

    #[tokio::test]
    async fn test_aggregator_basic() {
        let (tx, rx) = mpsc::channel(100);
        let aggregator = MetricsAggregator::new(rx);

        // Send some records
        for i in 0..5 {
            let record = RequestRecord {
                request_id: (i as u64).into(),
                scenario: "test".to_string(),
                input_tokens: 100,
                output_tokens: 50,
                ttft_ms: Some(100.0),
                tpot_ms: Some(10.0),
                e2e_ms: 500.0,
                status: ResponseStatus::Success,
                timestamp: chrono::Utc::now(),
            };
            tx.send(record).await.unwrap();
        }

        // Close channel to signal completion
        drop(tx);

        // Run aggregator
        let metrics = aggregator.run().await;

        assert_eq!(metrics.summary.total_requests, 5);
        assert_eq!(metrics.summary.successful_requests, 5);
        assert_eq!(metrics.summary.failed_requests, 0);
        assert_eq!(metrics.requests.len(), 5);
    }

    #[tokio::test]
    async fn test_aggregator_with_errors() {
        let (tx, rx) = mpsc::channel(100);
        let aggregator = MetricsAggregator::new(rx);

        // Send success
        tx.send(RequestRecord {
            request_id: 1u64.into(),
            scenario: "test".to_string(),
            input_tokens: 100,
            output_tokens: 50,
            ttft_ms: Some(100.0),
            tpot_ms: Some(10.0),
            e2e_ms: 500.0,
            status: ResponseStatus::Success,
            timestamp: chrono::Utc::now(),
        })
        .await
        .unwrap();

        // Send error
        tx.send(RequestRecord {
            request_id: 2u64.into(),
            scenario: "test".to_string(),
            input_tokens: 100,
            output_tokens: 0,
            ttft_ms: None,
            tpot_ms: None,
            e2e_ms: 100.0,
            status: ResponseStatus::Error(crate::error::ErrorKind::Timeout),
            timestamp: chrono::Utc::now(),
        })
        .await
        .unwrap();

        drop(tx);

        let metrics = aggregator.run().await;

        assert_eq!(metrics.summary.total_requests, 2);
        assert_eq!(metrics.summary.successful_requests, 1);
        assert_eq!(metrics.summary.failed_requests, 1);
        assert!((metrics.summary.error_rate - 0.5).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_aggregator_per_scenario_metrics() {
        let (tx, rx) = mpsc::channel(100);
        let aggregator = MetricsAggregator::new(rx);

        // Send records for different scenarios
        for scenario in ["scenario_a", "scenario_b"] {
            for i in 0..3 {
                tx.send(RequestRecord {
                    request_id: (i as u64).into(),
                    scenario: scenario.to_string(),
                    input_tokens: 100,
                    output_tokens: 50,
                    ttft_ms: Some(100.0),
                    tpot_ms: Some(10.0),
                    e2e_ms: 500.0,
                    status: ResponseStatus::Success,
                    timestamp: chrono::Utc::now(),
                })
                .await
                .unwrap();
            }
        }

        drop(tx);

        let metrics = aggregator.run().await;

        assert_eq!(metrics.scenarios.len(), 2);
        assert!(metrics.scenarios.contains_key("scenario_a"));
        assert!(metrics.scenarios.contains_key("scenario_b"));
        assert_eq!(metrics.scenarios["scenario_a"].request_count, 3);
        assert_eq!(metrics.scenarios["scenario_b"].request_count, 3);
    }

    #[tokio::test]
    async fn test_aggregator_with_ui_updates() {
        let (tx, rx) = mpsc::channel(100);
        let (ui_tx, mut ui_rx) = mpsc::channel(100);

        let aggregator = MetricsAggregator::new(rx)
            .with_ui_sender(ui_tx)
            .with_target_requests(10);

        // Send records
        for i in 0..5 {
            tx.send(RequestRecord {
                request_id: (i as u64).into(),
                scenario: "test".to_string(),
                input_tokens: 100,
                output_tokens: 50,
                ttft_ms: Some(100.0),
                tpot_ms: Some(10.0),
                e2e_ms: 500.0,
                status: ResponseStatus::Success,
                timestamp: chrono::Utc::now(),
            })
            .await
            .unwrap();
        }

        drop(tx);

        let metrics = aggregator.run().await;
        assert_eq!(metrics.summary.total_requests, 5);

        // Should have received at least one UI update (final update)
        let mut update_count = 0;
        while ui_rx.try_recv().is_ok() {
            update_count += 1;
        }
        assert!(update_count >= 1);
    }

    #[test]
    fn test_aggregator_debug() {
        let (_tx, rx) = mpsc::channel(100);
        let aggregator = MetricsAggregator::new(rx);
        let debug = format!("{:?}", aggregator);
        assert!(debug.contains("MetricsAggregator"));
    }
}
