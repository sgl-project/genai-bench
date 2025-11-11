//! Histogram plots for latency distributions

use crate::metrics::MetricsCollector;
use anyhow::Result;
use plotters::prelude::*;

pub struct HistogramPlotter;

impl HistogramPlotter {
    /// Plot TTFT histogram
    pub fn plot_ttft(collector: &MetricsCollector, path: &str) -> Result<()> {
        let root = BitMapBackend::new(path, (1200, 800)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut ttft_values: Vec<u64> = collector.iter().map(|m| m.ttft_ms).collect();
        ttft_values.sort_unstable();

        if ttft_values.is_empty() {
            return Ok(());
        }

        let max_ttft = *ttft_values.last().unwrap();
        let min_ttft = *ttft_values.first().unwrap();

        // Create histogram bins
        let num_bins = 50;
        let bin_size = ((max_ttft - min_ttft) / num_bins).max(1);
        let mut bins = vec![0u32; num_bins as usize];

        for &value in &ttft_values {
            let bin_idx = (((value - min_ttft) / bin_size).min(num_bins - 1)) as usize;
            bins[bin_idx] += 1;
        }

        let max_count = *bins.iter().max().unwrap_or(&0);

        let mut chart = ChartBuilder::on(&root)
            .caption("Time to First Token (TTFT) Distribution", ("sans-serif", 40))
            .margin(15)
            .x_label_area_size(50)
            .y_label_area_size(60)
            .build_cartesian_2d(
                min_ttft as f64..(max_ttft as f64 + bin_size as f64),
                0f64..(max_count as f64 * 1.1),
            )?;

        chart
            .configure_mesh()
            .x_desc("TTFT (ms)")
            .y_desc("Frequency")
            .x_label_formatter(&|x| format!("{:.0}", x))
            .draw()?;

        // Draw bars
        chart.draw_series(
            bins.iter().enumerate().map(|(idx, &count)| {
                let x0 = min_ttft + idx as u64 * bin_size;
                let x1 = min_ttft + (idx + 1) as u64 * bin_size;
                Rectangle::new(
                    [(x0 as f64, 0.0), (x1 as f64, count as f64)],
                    BLUE.mix(0.6).filled(),
                )
            }),
        )?;

        root.present()?;
        Ok(())
    }

    /// Plot total time histogram
    pub fn plot_total_time(collector: &MetricsCollector, path: &str) -> Result<()> {
        let root = BitMapBackend::new(path, (1200, 800)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut total_values: Vec<u64> = collector.iter().map(|m| m.total_time_ms).collect();
        total_values.sort_unstable();

        if total_values.is_empty() {
            return Ok(());
        }

        let max_time = *total_values.last().unwrap();
        let min_time = *total_values.first().unwrap();

        // Create histogram bins
        let num_bins = 50;
        let bin_size = ((max_time - min_time) / num_bins).max(1);
        let mut bins = vec![0u32; num_bins as usize];

        for &value in &total_values {
            let bin_idx = (((value - min_time) / bin_size).min(num_bins - 1)) as usize;
            bins[bin_idx] += 1;
        }

        let max_count = *bins.iter().max().unwrap_or(&0);

        let mut chart = ChartBuilder::on(&root)
            .caption("Total Request Time Distribution", ("sans-serif", 40))
            .margin(15)
            .x_label_area_size(50)
            .y_label_area_size(60)
            .build_cartesian_2d(
                min_time as f64..(max_time as f64 + bin_size as f64),
                0f64..(max_count as f64 * 1.1),
            )?;

        chart
            .configure_mesh()
            .x_desc("Total Time (ms)")
            .y_desc("Frequency")
            .x_label_formatter(&|x| format!("{:.0}", x))
            .draw()?;

        // Draw bars
        chart.draw_series(
            bins.iter().enumerate().map(|(idx, &count)| {
                let x0 = min_time + idx as u64 * bin_size;
                let x1 = min_time + (idx + 1) as u64 * bin_size;
                Rectangle::new(
                    [(x0 as f64, 0.0), (x1 as f64, count as f64)],
                    GREEN.mix(0.6).filled(),
                )
            }),
        )?;

        root.present()?;
        Ok(())
    }

    /// Plot token count histogram
    pub fn plot_tokens(collector: &MetricsCollector, path: &str) -> Result<()> {
        let root = BitMapBackend::new(path, (1200, 800)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut token_values: Vec<u32> = collector.iter().map(|m| m.completion_tokens).collect();
        token_values.sort_unstable();

        if token_values.is_empty() {
            return Ok(());
        }

        let max_tokens = *token_values.last().unwrap();
        let min_tokens = *token_values.first().unwrap();

        // Create histogram bins
        let num_bins = 50;
        let bin_size = ((max_tokens - min_tokens) / num_bins).max(1);
        let mut bins = vec![0u32; num_bins as usize];

        for &value in &token_values {
            let bin_idx = (((value - min_tokens) / bin_size).min(num_bins - 1)) as usize;
            bins[bin_idx] += 1;
        }

        let max_count = *bins.iter().max().unwrap_or(&0);

        let mut chart = ChartBuilder::on(&root)
            .caption("Completion Tokens Distribution", ("sans-serif", 40))
            .margin(15)
            .x_label_area_size(50)
            .y_label_area_size(60)
            .build_cartesian_2d(
                min_tokens as f64..(max_tokens as f64 + bin_size as f64),
                0f64..(max_count as f64 * 1.1),
            )?;

        chart
            .configure_mesh()
            .x_desc("Completion Tokens")
            .y_desc("Frequency")
            .x_label_formatter(&|x| format!("{:.0}", x))
            .draw()?;

        // Draw bars
        chart.draw_series(
            bins.iter().enumerate().map(|(idx, &count)| {
                let x0 = min_tokens + idx as u32 * bin_size;
                let x1 = min_tokens + (idx as u32 + 1) * bin_size;
                Rectangle::new(
                    [(x0 as f64, 0.0), (x1 as f64, count as f64)],
                    RED.mix(0.6).filled(),
                )
            }),
        )?;

        root.present()?;
        Ok(())
    }
}
