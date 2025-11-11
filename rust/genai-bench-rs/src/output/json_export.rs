//! JSON export functionality

use crate::metrics::MetricsCollector;
use anyhow::Result;
use serde_json::json;
use std::fs::File;
use std::io::Write;

pub struct JsonExporter;

impl JsonExporter {
    /// Export metrics to JSON file
    pub fn export(collector: &MetricsCollector, path: &str) -> Result<()> {
        let agg = collector.aggregate();

        // Collect raw data
        let raw_data: Vec<_> = collector
            .iter()
            .enumerate()
            .map(|(idx, m)| {
                json!({
                    "request_num": idx + 1,
                    "ttft_ms": m.ttft_ms,
                    "total_time_ms": m.total_time_ms,
                    "prompt_tokens": m.prompt_tokens,
                    "completion_tokens": m.completion_tokens,
                    "total_tokens": m.total_tokens,
                    "success": m.success,
                    "status_code": m.status_code,
                    "error_message": m.error_message,
                })
            })
            .collect();

        // Create complete output
        let output = json!({
            "summary": {
                "overall": {
                    "total_requests": agg.total_requests,
                    "successful_requests": agg.successful_requests,
                    "failed_requests": agg.failed_requests,
                    "success_rate_percent": (agg.successful_requests as f64 / agg.total_requests as f64) * 100.0,
                },
                "ttft": {
                    "avg_ms": agg.avg_ttft_ms,
                    "p50_ms": agg.p50_ttft_ms,
                    "p95_ms": agg.p95_ttft_ms,
                    "p99_ms": agg.p99_ttft_ms,
                },
                "total_time": {
                    "avg_ms": agg.avg_total_time_ms,
                    "p50_ms": agg.p50_total_time_ms,
                    "p95_ms": agg.p95_total_time_ms,
                    "p99_ms": agg.p99_total_time_ms,
                },
                "tokens": {
                    "total_prompt": agg.total_prompt_tokens,
                    "total_completion": agg.total_completion_tokens,
                    "total": agg.total_tokens,
                    "avg_per_second": agg.avg_tokens_per_second,
                }
            },
            "raw_data": raw_data,
        });

        let file = File::create(path)?;
        serde_json::to_writer_pretty(file, &output)?;

        Ok(())
    }

    /// Export summary only (smaller file)
    pub fn export_summary(collector: &MetricsCollector, path: &str) -> Result<()> {
        let agg = collector.aggregate();

        let output = json!({
            "total_requests": agg.total_requests,
            "successful_requests": agg.successful_requests,
            "failed_requests": agg.failed_requests,
            "success_rate_percent": (agg.successful_requests as f64 / agg.total_requests as f64) * 100.0,
            "ttft": {
                "avg_ms": agg.avg_ttft_ms,
                "p50_ms": agg.p50_ttft_ms,
                "p95_ms": agg.p95_ttft_ms,
                "p99_ms": agg.p99_ttft_ms,
            },
            "total_time": {
                "avg_ms": agg.avg_total_time_ms,
                "p50_ms": agg.p50_total_time_ms,
                "p95_ms": agg.p95_total_time_ms,
                "p99_ms": agg.p99_total_time_ms,
            },
            "tokens": {
                "total_prompt": agg.total_prompt_tokens,
                "total_completion": agg.total_completion_tokens,
                "total": agg.total_tokens,
                "avg_per_second": agg.avg_tokens_per_second,
            }
        });

        let mut file = File::create(path)?;
        file.write_all(serde_json::to_string_pretty(&output)?.as_bytes())?;

        Ok(())
    }
}
