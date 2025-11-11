//! CSV export functionality

use crate::metrics::MetricsCollector;
use anyhow::Result;
use csv::Writer;
use std::fs::File;

pub struct CsvExporter;

impl CsvExporter {
    /// Export metrics to CSV file
    pub fn export(collector: &MetricsCollector, path: &str) -> Result<()> {
        let file = File::create(path)?;
        let mut wtr = Writer::from_writer(file);

        // Write headers
        wtr.write_record(&[
            "request_num",
            "ttft_ms",
            "total_time_ms",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "success",
            "status_code",
            "error_message",
        ])?;

        // Write data rows
        for (idx, metric) in collector.iter().enumerate() {
            wtr.write_record(&[
                (idx + 1).to_string(),
                metric.ttft_ms.to_string(),
                metric.total_time_ms.to_string(),
                metric.prompt_tokens.to_string(),
                metric.completion_tokens.to_string(),
                metric.total_tokens.to_string(),
                metric.success.to_string(),
                metric.status_code.to_string(),
                metric
                    .error_message
                    .as_ref()
                    .unwrap_or(&String::new())
                    .clone(),
            ])?;
        }

        wtr.flush()?;
        Ok(())
    }

    /// Export summary statistics to CSV
    pub fn export_summary(collector: &MetricsCollector, path: &str) -> Result<()> {
        let agg = collector.aggregate();
        let file = File::create(path)?;
        let mut wtr = Writer::from_writer(file);

        // Write headers
        wtr.write_record(&["metric", "value"])?;

        // Write summary data
        wtr.write_record(&[
            "total_requests",
            &agg.total_requests.to_string(),
        ])?;
        wtr.write_record(&[
            "successful_requests",
            &agg.successful_requests.to_string(),
        ])?;
        wtr.write_record(&["failed_requests", &agg.failed_requests.to_string()])?;
        wtr.write_record(&[
            "success_rate_percent",
            &format!(
                "{:.2}",
                (agg.successful_requests as f64 / agg.total_requests as f64) * 100.0
            ),
        ])?;

        wtr.write_record(&["avg_ttft_ms", &format!("{:.2}", agg.avg_ttft_ms)])?;
        wtr.write_record(&["p50_ttft_ms", &format!("{:.2}", agg.p50_ttft_ms)])?;
        wtr.write_record(&["p95_ttft_ms", &format!("{:.2}", agg.p95_ttft_ms)])?;
        wtr.write_record(&["p99_ttft_ms", &format!("{:.2}", agg.p99_ttft_ms)])?;

        wtr.write_record(&[
            "avg_total_time_ms",
            &format!("{:.2}", agg.avg_total_time_ms),
        ])?;
        wtr.write_record(&[
            "p50_total_time_ms",
            &format!("{:.2}", agg.p50_total_time_ms),
        ])?;
        wtr.write_record(&[
            "p95_total_time_ms",
            &format!("{:.2}", agg.p95_total_time_ms),
        ])?;
        wtr.write_record(&[
            "p99_total_time_ms",
            &format!("{:.2}", agg.p99_total_time_ms),
        ])?;

        wtr.write_record(&[
            "total_prompt_tokens",
            &agg.total_prompt_tokens.to_string(),
        ])?;
        wtr.write_record(&[
            "total_completion_tokens",
            &agg.total_completion_tokens.to_string(),
        ])?;
        wtr.write_record(&["total_tokens", &agg.total_tokens.to_string()])?;
        wtr.write_record(&[
            "avg_tokens_per_second",
            &format!("{:.2}", agg.avg_tokens_per_second),
        ])?;

        wtr.flush()?;
        Ok(())
    }
}
