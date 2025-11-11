//! Percentile charts

use crate::metrics::MetricsCollector;
use anyhow::Result;
use plotters::prelude::*;

pub struct PercentilePlotter;

impl PercentilePlotter {
    /// Plot percentile comparison chart
    pub fn plot(collector: &MetricsCollector, path: &str) -> Result<()> {
        let root = BitMapBackend::new(path, (1200, 800)).into_drawing_area();
        root.fill(&WHITE)?;

        if collector.is_empty() {
            return Ok(());
        }

        // Collect and sort values
        let mut ttft_values: Vec<u64> = collector.iter().map(|m| m.ttft_ms).collect();
        let mut total_values: Vec<u64> = collector.iter().map(|m| m.total_time_ms).collect();
        ttft_values.sort_unstable();
        total_values.sort_unstable();

        // Calculate percentiles
        let percentiles = vec![10, 25, 50, 75, 90, 95, 99];
        let mut ttft_percentiles = Vec::new();
        let mut total_percentiles = Vec::new();

        for &p in &percentiles {
            let idx = ((p as f64 / 100.0) * (ttft_values.len() - 1) as f64).round() as usize;
            ttft_percentiles.push((p, ttft_values[idx] as f64));

            let idx = ((p as f64 / 100.0) * (total_values.len() - 1) as f64).round() as usize;
            total_percentiles.push((p, total_values[idx] as f64));
        }

        let max_value = ttft_percentiles
            .iter()
            .chain(total_percentiles.iter())
            .map(|(_, v)| *v)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);

        let mut chart = ChartBuilder::on(&root)
            .caption("Latency Percentiles", ("sans-serif", 40))
            .margin(15)
            .x_label_area_size(50)
            .y_label_area_size(70)
            .build_cartesian_2d(0f64..100f64, 0f64..(max_value * 1.1))?;

        chart
            .configure_mesh()
            .x_desc("Percentile")
            .y_desc("Latency (ms)")
            .x_label_formatter(&|x| format!("P{:.0}", x))
            .y_label_formatter(&|y| format!("{:.0}", y))
            .draw()?;

        // Draw TTFT line
        chart
            .draw_series(LineSeries::new(
                ttft_percentiles.iter().map(|(p, v)| (*p as f64, *v)),
                &BLUE,
            ))?
            .label("TTFT")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

        // Draw Total Time line
        chart
            .draw_series(LineSeries::new(
                total_percentiles.iter().map(|(p, v)| (*p as f64, *v)),
                &RED,
            ))?
            .label("Total Time")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

        // Draw markers
        chart.draw_series(
            ttft_percentiles
                .iter()
                .map(|(p, v)| Circle::new((*p as f64, *v), 4, BLUE.filled())),
        )?;

        chart.draw_series(
            total_percentiles
                .iter()
                .map(|(p, v)| Circle::new((*p as f64, *v), 4, RED.filled())),
        )?;

        chart
            .configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()?;

        root.present()?;
        Ok(())
    }

    /// Plot CDF (Cumulative Distribution Function)
    pub fn plot_cdf(collector: &MetricsCollector, path: &str) -> Result<()> {
        let root = BitMapBackend::new(path, (1200, 800)).into_drawing_area();
        root.fill(&WHITE)?;

        if collector.is_empty() {
            return Ok(());
        }

        // Collect and sort TTFT values
        let mut ttft_values: Vec<u64> = collector.iter().map(|m| m.ttft_ms).collect();
        ttft_values.sort_unstable();

        // Create CDF data points
        let cdf_points: Vec<(f64, f64)> = ttft_values
            .iter()
            .enumerate()
            .map(|(idx, &value)| {
                let percentile = (idx + 1) as f64 / ttft_values.len() as f64 * 100.0;
                (value as f64, percentile)
            })
            .collect();

        let max_ttft = *ttft_values.last().unwrap() as f64;

        let mut chart = ChartBuilder::on(&root)
            .caption(
                "TTFT Cumulative Distribution Function (CDF)",
                ("sans-serif", 40),
            )
            .margin(15)
            .x_label_area_size(50)
            .y_label_area_size(70)
            .build_cartesian_2d(0f64..(max_ttft * 1.05), 0f64..100f64)?;

        chart
            .configure_mesh()
            .x_desc("TTFT (ms)")
            .y_desc("Percentile")
            .x_label_formatter(&|x| format!("{:.0}", x))
            .y_label_formatter(&|y| format!("{:.0}%", y))
            .draw()?;

        // Draw CDF line
        chart
            .draw_series(LineSeries::new(cdf_points, &BLUE))?
            .label("CDF")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

        // Draw P50, P95, P99 markers
        let p50_idx = (0.50 * (ttft_values.len() - 1) as f64).round() as usize;
        let p95_idx = (0.95 * (ttft_values.len() - 1) as f64).round() as usize;
        let p99_idx = (0.99 * (ttft_values.len() - 1) as f64).round() as usize;

        chart.draw_series(vec![
            Circle::new((ttft_values[p50_idx] as f64, 50.0), 5, RED.filled()),
            Circle::new((ttft_values[p95_idx] as f64, 95.0), 5, RED.filled()),
            Circle::new((ttft_values[p99_idx] as f64, 99.0), 5, RED.filled()),
        ])?;

        chart
            .configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()?;

        root.present()?;
        Ok(())
    }
}
