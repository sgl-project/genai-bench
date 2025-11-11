//! Dashboard rendering for the TUI - TGI-style with Ratatui widgets

use super::charts::{create_histogram, create_scatter_chart};
use super::UiState;
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, Paragraph},
    Frame,
};

pub struct Dashboard {}

impl Dashboard {
    pub fn new() -> Self {
        Self {}
    }

    /// Render the dashboard - matches Python layout
    pub fn render(&self, frame: &mut Frame, state: &UiState) {
        let size = frame.size();

        // Main layout: 4 rows matching Python
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),  // Row 1: Progress bars
                Constraint::Percentage(35), // Row 2: Metrics + Histograms
                Constraint::Percentage(30), // Row 3: Scatter plots
                Constraint::Min(5),     // Row 4: Footer/Status
            ])
            .split(size);

        // Row 1: Progress bars
        self.render_progress_bars(frame, chunks[0], state);

        // Row 2: Input/Output metrics with histograms
        self.render_metrics_and_histograms(frame, chunks[1], state);

        // Row 3: Scatter plots
        self.render_scatter_plots(frame, chunks[2], state);

        // Row 4: Footer
        self.render_footer(frame, chunks[3], state);
    }

    fn render_progress_bars(&self, frame: &mut Frame, area: Rect, state: &UiState) {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(area);

        // Total Progress (left)
        let total_progress = Gauge::default()
            .block(
                Block::default()
                    .title("Total Progress")
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::Magenta)),
            )
            .gauge_style(Style::default().fg(Color::Magenta))
            .ratio(state.progress_percent() / 100.0)
            .label(format!("{:.0}%", state.progress_percent()));

        frame.render_widget(total_progress, chunks[0]);

        // Current Run Progress (right)
        let run_progress = Gauge::default()
            .block(
                Block::default()
                    .title("Current Run Progress")
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::Cyan)),
            )
            .gauge_style(Style::default().fg(Color::Cyan))
            .ratio(state.progress_percent() / 100.0)
            .label(format!(
                "{}/{} ({:.0}%)",
                state.completed,
                state.total,
                state.progress_percent()
            ));

        frame.render_widget(run_progress, chunks[1]);
    }

    fn render_metrics_and_histograms(&self, frame: &mut Frame, area: Rect, state: &UiState) {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(area);

        // Left: Input metrics
        self.render_input_section(frame, chunks[0], state);

        // Right: Output metrics
        self.render_output_section(frame, chunks[1], state);
    }

    fn render_input_section(&self, frame: &mut Frame, area: Rect, state: &UiState) {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(30), Constraint::Percentage(70)])
            .split(area);

        // Left: Metrics panels
        let metric_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(chunks[0]);

        // Input Throughput
        self.render_throughput_panel(frame, metric_chunks[0], "Input", state, true);

        // Input Latency (TTFT)
        self.render_latency_panel(frame, metric_chunks[1], "Input", state, true);

        // Right: Histogram
        self.render_histogram_panel(frame, chunks[1], "Input Latency Histogram", state, true);
    }

    fn render_output_section(&self, frame: &mut Frame, area: Rect, state: &UiState) {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(30), Constraint::Percentage(70)])
            .split(area);

        // Left: Metrics panels
        let metric_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(chunks[0]);

        // Output Throughput
        self.render_throughput_panel(frame, metric_chunks[0], "Output", state, false);

        // Output Latency
        self.render_latency_panel(frame, metric_chunks[1], "Output", state, false);

        // Right: Histogram
        self.render_histogram_panel(frame, chunks[1], "Output Latency Histogram", state, false);
    }

    fn render_throughput_panel(
        &self,
        frame: &mut Frame,
        area: Rect,
        title_prefix: &str,
        state: &UiState,
        is_input: bool,
    ) {
        let block = Block::default()
            .title(format!("{} Throughput", title_prefix))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Blue));

        if let Some(ref metrics) = state.metrics {
            let avg_throughput = if is_input {
                metrics.avg_tokens_per_second
            } else {
                metrics.avg_tokens_per_second
            };

            let lines = vec![
                Line::from(vec![
                    Span::styled("Avg: ", Style::default().fg(Color::Yellow)),
                    Span::raw(format!("{:.2} tokens/sec", avg_throughput)),
                ]),
                Line::from(""),
            ];

            let paragraph = Paragraph::new(lines).block(block);
            frame.render_widget(paragraph, area);
        } else {
            frame.render_widget(block, area);
        }
    }

    fn render_latency_panel(
        &self,
        frame: &mut Frame,
        area: Rect,
        title_prefix: &str,
        state: &UiState,
        is_input: bool,
    ) {
        let block = Block::default()
            .title(format!("{} Latency", title_prefix))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Yellow));

        if let Some(ref metrics) = state.metrics {
            let (avg, _min, _max, p50, p90, p99) = if is_input {
                (
                    metrics.avg_ttft_ms,
                    0.0, // Would need to track min/max separately
                    0.0,
                    metrics.p50_ttft_ms,
                    metrics.p95_ttft_ms, // Using P95 as P90 approximation
                    metrics.p99_ttft_ms,
                )
            } else {
                (
                    metrics.avg_total_time_ms,
                    0.0,
                    0.0,
                    metrics.p50_total_time_ms,
                    metrics.p95_total_time_ms,
                    metrics.p99_total_time_ms,
                )
            };

            let lines = vec![
                Line::from(vec![
                    Span::styled("Avg: ", Style::default().fg(Color::Yellow)),
                    Span::raw(format!("{:.2} ms", avg)),
                ]),
                Line::from(vec![
                    Span::styled("P50: ", Style::default().fg(Color::Blue)),
                    Span::raw(format!("{:.2} ms", p50)),
                ]),
                Line::from(vec![
                    Span::styled("P90: ", Style::default().fg(Color::Magenta)),
                    Span::raw(format!("{:.2} ms", p90)),
                ]),
                Line::from(vec![
                    Span::styled("P99: ", Style::default().fg(Color::Green)),
                    Span::raw(format!("{:.2} ms", p99)),
                ]),
            ];

            let paragraph = Paragraph::new(lines).block(block);
            frame.render_widget(paragraph, area);
        } else {
            frame.render_widget(block, area);
        }
    }

    fn render_histogram_panel(
        &self,
        frame: &mut Frame,
        area: Rect,
        title: &str,
        state: &UiState,
        is_input: bool,
    ) {
        let values = if is_input {
            &state.ttft_values
        } else {
            &state.total_time_values
        };

        if !values.is_empty() {
            let histogram = create_histogram(values, title, 10);
            frame.render_widget(histogram, area);
        } else {
            let border_color = if is_input { Color::Green } else { Color::Blue };
            let waiting = Paragraph::new("Waiting for data...")
                .style(Style::default().fg(Color::DarkGray))
                .alignment(Alignment::Center)
                .block(
                    Block::default()
                        .title(title)
                        .borders(Borders::ALL)
                        .border_style(Style::default().fg(border_color).add_modifier(Modifier::BOLD)),
                );
            frame.render_widget(waiting, area);
        }
    }

    fn render_scatter_plots(&self, frame: &mut Frame, area: Rect, state: &UiState) {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(area);

        // Left: TTFT vs Input Throughput
        if state.ttft_values.len() > 1 {
            let chart = create_scatter_chart(
                &state.input_throughput_values,
                &state.ttft_values,
                "Input Latency vs Input Throughput",
                "Throughput (tokens/sec)",
                "Latency (ms)",
                Color::Green,
            );
            frame.render_widget(chart, chunks[0]);
        } else {
            let waiting = Paragraph::new("Collecting data...")
                .style(Style::default().fg(Color::DarkGray))
                .alignment(Alignment::Center)
                .block(
                    Block::default()
                        .title("Input Latency vs Input Throughput")
                        .borders(Borders::ALL)
                        .border_style(Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
                );
            frame.render_widget(waiting, chunks[0]);
        }

        // Right: Output Latency vs Output Throughput
        if state.total_time_values.len() > 1 {
            let chart = create_scatter_chart(
                &state.output_throughput_values,
                &state.total_time_values,
                "Output Latency vs Output Throughput",
                "Throughput (tokens/sec)",
                "Latency (ms)",
                Color::Blue,
            );
            frame.render_widget(chart, chunks[1]);
        } else {
            let waiting = Paragraph::new("Collecting data...")
                .style(Style::default().fg(Color::DarkGray))
                .alignment(Alignment::Center)
                .block(
                    Block::default()
                        .title("Output Latency vs Output Throughput")
                        .borders(Borders::ALL)
                        .border_style(Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD)),
                );
            frame.render_widget(waiting, chunks[1]);
        }
    }

    fn render_footer(&self, frame: &mut Frame, area: Rect, state: &UiState) {
        let mut footer_text = vec![
            Span::styled("Press 'q' or 'ESC' to quit", Style::default().fg(Color::DarkGray)),
            Span::raw(" | "),
            Span::raw(format!("Elapsed: {:.1}s", state.elapsed_secs())),
        ];

        if let Some(eta) = state.eta_secs() {
            footer_text.push(Span::raw(" | "));
            footer_text.push(Span::styled(
                format!("ETA: {:.1}s", eta),
                Style::default().fg(Color::Yellow),
            ));
        }

        // Show error if present
        if let Some(ref error) = state.error_message {
            footer_text.push(Span::raw(" | "));
            footer_text.push(Span::styled(
                format!("Error: {}", error),
                Style::default().fg(Color::Red),
            ));
        }

        // Show completion status
        if state.is_complete {
            footer_text.push(Span::raw(" | "));
            footer_text.push(Span::styled(
                "Benchmark Complete!",
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            ));
        }

        let footer = Paragraph::new(Line::from(footer_text))
            .alignment(Alignment::Center)
            .block(Block::default().borders(Borders::ALL));

        frame.render_widget(footer, area);
    }
}

impl Default for Dashboard {
    fn default() -> Self {
        Self::new()
    }
}
