//! Throughput plots over time

use crate::metrics::MetricsCollector;
use anyhow::Result;
use plotters::prelude::*;

pub struct ThroughputPlotter;

impl ThroughputPlotter {
    /// Plot throughput over time (moving average)
    pub fn plot(collector: &MetricsCollector, path: &str) -> Result<()> {
        let root = BitMapBackend::new(path, (1400, 900)).into_drawing_area();
        root.fill(&WHITE)?;

        if collector.is_empty() {
            return Ok(());
        }

        // Calculate cumulative time and tokens for each request
        let mut data_points: Vec<(f64, f64)> = Vec::new();
        let mut cumulative_time_ms = 0u64;
        let mut cumulative_tokens = 0u64;

        for (idx, metric) in collector.iter().enumerate() {
            cumulative_time_ms += metric.total_time_ms;
            cumulative_tokens += metric.completion_tokens as u64;

            // Calculate throughput as tokens per second
            let time_seconds = cumulative_time_ms as f64 / 1000.0;
            let throughput = if time_seconds > 0.0 {
                cumulative_tokens as f64 / time_seconds
            } else {
                0.0
            };

            data_points.push((idx as f64, throughput));
        }

        let max_throughput = data_points
            .iter()
            .map(|(_, t)| *t)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);

        let max_requests = data_points.len() as f64;

        let mut chart = ChartBuilder::on(&root)
            .caption("Token Throughput Over Time", ("sans-serif", 40))
            .margin(15)
            .x_label_area_size(50)
            .y_label_area_size(70)
            .build_cartesian_2d(0f64..max_requests, 0f64..(max_throughput * 1.1))?;

        chart
            .configure_mesh()
            .x_desc("Request Number")
            .y_desc("Tokens per Second (Cumulative)")
            .x_label_formatter(&|x| format!("{:.0}", x))
            .y_label_formatter(&|y| format!("{:.1}", y))
            .draw()?;

        // Draw line
        chart
            .draw_series(LineSeries::new(data_points.clone(), &BLUE))?
            .label("Throughput")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

        chart
            .configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()?;

        root.present()?;
        Ok(())
    }

    /// Plot requests per second over time windows
    pub fn plot_rps(collector: &MetricsCollector, path: &str, window_size_ms: u64) -> Result<()> {
        let root = BitMapBackend::new(path, (1400, 900)).into_drawing_area();
        root.fill(&WHITE)?;

        if collector.is_empty() {
            return Ok(());
        }

        // Group requests by time windows
        let mut time_windows = std::collections::HashMap::new();
        let mut max_time = 0u64;

        for metric in collector.iter() {
            let window = metric.total_time_ms / window_size_ms;
            *time_windows.entry(window).or_insert(0u32) += 1;
            max_time = max_time.max(metric.total_time_ms);
        }

        let max_window = max_time / window_size_ms;
        let max_count = *time_windows.values().max().unwrap_or(&0);

        // Convert to sorted data points
        let data_points: Vec<(f64, f64)> = (0..=max_window)
            .map(|w| {
                let count = *time_windows.get(&w).unwrap_or(&0);
                let time_seconds = (w * window_size_ms) as f64 / 1000.0;
                let rps = count as f64 / (window_size_ms as f64 / 1000.0);
                (time_seconds, rps)
            })
            .collect();

        let max_time_seconds = (max_window * window_size_ms) as f64 / 1000.0;

        let mut chart = ChartBuilder::on(&root)
            .caption("Requests Per Second Over Time", ("sans-serif", 40))
            .margin(15)
            .x_label_area_size(50)
            .y_label_area_size(70)
            .build_cartesian_2d(
                0f64..max_time_seconds,
                0f64..(max_count as f64 * 1.2 / (window_size_ms as f64 / 1000.0)),
            )?;

        chart
            .configure_mesh()
            .x_desc("Time (seconds)")
            .y_desc("Requests per Second")
            .x_label_formatter(&|x| format!("{:.1}", x))
            .y_label_formatter(&|y| format!("{:.1}", y))
            .draw()?;

        // Draw bars
        chart.draw_series(
            data_points.iter().map(|(time, rps)| {
                let x0 = *time;
                let x1 = time + (window_size_ms as f64 / 1000.0);
                Rectangle::new([(x0, 0.0), (x1, *rps)], GREEN.mix(0.6).filled())
            }),
        )?;

        root.present()?;
        Ok(())
    }
}
