//! Excel export functionality

use crate::metrics::MetricsCollector;
use anyhow::Result;
use rust_xlsxwriter::*;

pub struct ExcelExporter;

impl ExcelExporter {
    /// Export metrics to Excel file with multiple sheets
    pub fn export(collector: &MetricsCollector, path: &str) -> Result<()> {
        let mut workbook = Workbook::new();

        // Summary sheet
        let summary_sheet = workbook.add_worksheet();
        summary_sheet.set_name("Summary")?;
        Self::write_summary(summary_sheet, collector)?;

        // Raw data sheet
        let raw_sheet = workbook.add_worksheet();
        raw_sheet.set_name("Raw Data")?;
        Self::write_raw_data(raw_sheet, collector)?;

        // Percentiles sheet
        let percentile_sheet = workbook.add_worksheet();
        percentile_sheet.set_name("Percentiles")?;
        Self::write_percentiles(percentile_sheet, collector)?;

        workbook.save(path)?;
        Ok(())
    }

    fn write_summary(sheet: &mut Worksheet, collector: &MetricsCollector) -> Result<()> {
        let agg = collector.aggregate();

        // Create bold format for headers
        let bold = Format::new().set_bold();
        let number_format = Format::new().set_num_format("0.00");

        // Headers
        sheet.write_with_format(0, 0, "Metric", &bold)?;
        sheet.write_with_format(0, 1, "Value", &bold)?;

        let mut row = 1u32;

        // Overall statistics
        sheet.write(row, 0, "Total Requests")?;
        sheet.write(row, 1, agg.total_requests as f64)?;
        row += 1;

        sheet.write(row, 0, "Successful Requests")?;
        sheet.write(row, 1, agg.successful_requests as f64)?;
        row += 1;

        sheet.write(row, 0, "Failed Requests")?;
        sheet.write(row, 1, agg.failed_requests as f64)?;
        row += 1;

        sheet.write(row, 0, "Success Rate (%)")?;
        let success_rate = (agg.successful_requests as f64 / agg.total_requests as f64) * 100.0;
        sheet.write_with_format(row, 1, success_rate, &number_format)?;
        row += 1;

        // Empty row
        row += 1;

        // TTFT metrics
        sheet.write(row, 0, "Avg TTFT (ms)")?;
        sheet.write_with_format(row, 1, agg.avg_ttft_ms, &number_format)?;
        row += 1;

        sheet.write(row, 0, "P50 TTFT (ms)")?;
        sheet.write_with_format(row, 1, agg.p50_ttft_ms, &number_format)?;
        row += 1;

        sheet.write(row, 0, "P95 TTFT (ms)")?;
        sheet.write_with_format(row, 1, agg.p95_ttft_ms, &number_format)?;
        row += 1;

        sheet.write(row, 0, "P99 TTFT (ms)")?;
        sheet.write_with_format(row, 1, agg.p99_ttft_ms, &number_format)?;
        row += 1;

        // Empty row
        row += 1;

        // Total time metrics
        sheet.write(row, 0, "Avg Total Time (ms)")?;
        sheet.write_with_format(row, 1, agg.avg_total_time_ms, &number_format)?;
        row += 1;

        sheet.write(row, 0, "P50 Total Time (ms)")?;
        sheet.write_with_format(row, 1, agg.p50_total_time_ms, &number_format)?;
        row += 1;

        sheet.write(row, 0, "P95 Total Time (ms)")?;
        sheet.write_with_format(row, 1, agg.p95_total_time_ms, &number_format)?;
        row += 1;

        sheet.write(row, 0, "P99 Total Time (ms)")?;
        sheet.write_with_format(row, 1, agg.p99_total_time_ms, &number_format)?;
        row += 1;

        // Empty row
        row += 1;

        // Token statistics
        sheet.write(row, 0, "Total Input Tokens")?;
        sheet.write(row, 1, agg.total_prompt_tokens as f64)?;
        row += 1;

        sheet.write(row, 0, "Total Output Tokens")?;
        sheet.write(row, 1, agg.total_completion_tokens as f64)?;
        row += 1;

        sheet.write(row, 0, "Total Tokens")?;
        sheet.write(row, 1, agg.total_tokens as f64)?;
        row += 1;

        sheet.write(row, 0, "Avg Tokens/Second")?;
        sheet.write_with_format(row, 1, agg.avg_tokens_per_second, &number_format)?;

        // Set column widths
        sheet.set_column_width(0, 25)?;
        sheet.set_column_width(1, 15)?;

        Ok(())
    }

    fn write_raw_data(sheet: &mut Worksheet, collector: &MetricsCollector) -> Result<()> {
        let bold = Format::new().set_bold();

        // Headers
        let headers = vec![
            "Request #",
            "TTFT (ms)",
            "Total Time (ms)",
            "Prompt Tokens",
            "Completion Tokens",
            "Total Tokens",
            "Success",
            "Status Code",
            "Error Message",
        ];

        for (col, header) in headers.iter().enumerate() {
            sheet.write_with_format(0, col as u16, *header, &bold)?;
        }

        // Data rows
        for (idx, metric) in collector.iter().enumerate() {
            let row = (idx + 1) as u32;
            sheet.write(row, 0, (idx + 1) as f64)?;
            sheet.write(row, 1, metric.ttft_ms as f64)?;
            sheet.write(row, 2, metric.total_time_ms as f64)?;
            sheet.write(row, 3, metric.prompt_tokens as f64)?;
            sheet.write(row, 4, metric.completion_tokens as f64)?;
            sheet.write(row, 5, metric.total_tokens as f64)?;
            sheet.write(row, 6, if metric.success { "Yes" } else { "No" })?;
            sheet.write(row, 7, metric.status_code as f64)?;
            sheet.write(row, 8, metric.error_message.as_deref().unwrap_or(""))?;
        }

        // Set column widths
        sheet.set_column_width(0, 12)?;
        sheet.set_column_width(1, 12)?;
        sheet.set_column_width(2, 15)?;
        sheet.set_column_width(3, 15)?;
        sheet.set_column_width(4, 18)?;
        sheet.set_column_width(5, 13)?;
        sheet.set_column_width(6, 10)?;
        sheet.set_column_width(7, 12)?;
        sheet.set_column_width(8, 30)?;

        Ok(())
    }

    fn write_percentiles(sheet: &mut Worksheet, collector: &MetricsCollector) -> Result<()> {
        let bold = Format::new().set_bold();
        let number_format = Format::new().set_num_format("0.00");

        // Headers
        sheet.write_with_format(0, 0, "Percentile", &bold)?;
        sheet.write_with_format(0, 1, "TTFT (ms)", &bold)?;
        sheet.write_with_format(0, 2, "Total Time (ms)", &bold)?;

        // Collect and sort values
        let mut ttft_values: Vec<u64> = collector.iter().map(|m| m.ttft_ms).collect();
        let mut total_values: Vec<u64> = collector.iter().map(|m| m.total_time_ms).collect();
        ttft_values.sort_unstable();
        total_values.sort_unstable();

        // Calculate percentiles
        let percentiles = vec![10, 25, 50, 75, 90, 95, 99];
        for (idx, &p) in percentiles.iter().enumerate() {
            let row = (idx + 1) as u32;
            let percentile_idx =
                ((p as f64 / 100.0) * (ttft_values.len() - 1) as f64).round() as usize;

            sheet.write(row, 0, format!("P{}", p))?;
            sheet.write_with_format(row, 1, ttft_values[percentile_idx] as f64, &number_format)?;
            sheet.write_with_format(
                row,
                2,
                total_values[percentile_idx] as f64,
                &number_format,
            )?;
        }

        // Set column widths
        sheet.set_column_width(0, 12)?;
        sheet.set_column_width(1, 15)?;
        sheet.set_column_width(2, 18)?;

        Ok(())
    }
}
