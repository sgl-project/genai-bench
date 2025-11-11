//! Chart widgets for the TUI

use ratatui::{
    style::{Color, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
};

/// Create a simple horizontal bar histogram
pub fn create_histogram<'a>(
    values: &[f64],
    title: &'a str,
    max_bins: usize,
) -> Paragraph<'a> {
    let lines = if values.is_empty() {
        vec![Line::from("No data yet...")]
    } else {
        render_simple_histogram(values, max_bins, 50)
    };

    let border_color = if title.contains("Input") {
        Color::Green
    } else {
        Color::Blue
    };

    Paragraph::new(lines).block(
        Block::default()
            .title(title)
            .borders(Borders::ALL)
            .border_style(Style::default().fg(border_color)),
    )
}

/// Render histogram as text lines
fn render_simple_histogram(values: &[f64], max_bins: usize, width: usize) -> Vec<Line<'static>> {
    if values.is_empty() {
        return vec![Line::from("No data")];
    }

    let min_val = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max_val = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    if (max_val - min_val).abs() < f64::EPSILON {
        return vec![Line::from("All values identical")];
    }

    // Calculate bins
    let range = max_val - min_val;
    let bin_width = range / max_bins as f64;
    let mut bins = vec![0u64; max_bins];

    for &value in values {
        let bin_idx = ((value - min_val) / bin_width).floor() as usize;
        let bin_idx = bin_idx.min(max_bins - 1);
        bins[bin_idx] += 1;
    }

    let max_count = *bins.iter().max().unwrap_or(&1);
    let mut lines = Vec::new();

    for (i, &count) in bins.iter().enumerate() {
        let bin_start = min_val + i as f64 * bin_width;
        let bar_length = if max_count > 0 {
            (count as f64 / max_count as f64 * (width - 15) as f64).round() as usize
        } else {
            0
        };

        // Color based on count (Red -> Yellow -> Green)
        let color = if count < max_count / 3 {
            Color::Red
        } else if count < max_count * 2 / 3 {
            Color::Yellow
        } else {
            Color::Green
        };

        let bar = "█".repeat(bar_length);
        lines.push(Line::from(vec![
            Span::styled(format!("{:>6.0}ms │ ", bin_start), Style::default().fg(Color::Gray)),
            Span::styled(bar, Style::default().fg(color)),
            Span::styled(format!(" {}", count), Style::default().fg(Color::White)),
        ]));
    }

    lines
}

/// Create a simple scatter plot representation
pub fn create_scatter_chart<'a>(
    x_values: &[f64],
    y_values: &[f64],
    title: &'a str,
    x_label: &str,
    y_label: &str,
    color: Color,
) -> Paragraph<'a> {
    let lines = if x_values.len() < 2 || y_values.len() < 2 {
        vec![
            Line::from(""),
            Line::from(Span::styled(
                "Collecting data...",
                Style::default().fg(Color::DarkGray),
            )),
            Line::from(""),
            Line::from(format!("X: {}", x_label)),
            Line::from(format!("Y: {}", y_label)),
        ]
    } else {
        // Simple text representation of scatter plot
        let x_min = x_values.iter().copied().fold(f64::INFINITY, f64::min);
        let x_max = x_values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let y_min = y_values.iter().copied().fold(f64::INFINITY, f64::min);
        let y_max = y_values.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        vec![
            Line::from(""),
            Line::from(vec![
                Span::styled("Range: ", Style::default().fg(Color::Gray)),
                Span::raw(format!(
                    "X=[{:.0}, {:.0}], Y=[{:.0}, {:.0}]",
                    x_min, x_max, y_min, y_max
                )),
            ]),
            Line::from(vec![
                Span::styled("Points: ", Style::default().fg(Color::Gray)),
                Span::styled(format!("{}", x_values.len()), Style::default().fg(color)),
            ]),
            Line::from(""),
            Line::from(format!("X: {}", x_label)),
            Line::from(format!("Y: {}", y_label)),
        ]
    };

    Paragraph::new(lines).block(
        Block::default()
            .title(title)
            .borders(Borders::ALL)
            .border_style(Style::default().fg(color)),
    )
}
