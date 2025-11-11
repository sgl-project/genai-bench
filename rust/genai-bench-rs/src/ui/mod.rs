//! Terminal User Interface (TUI) for live benchmarking dashboard

mod charts;
mod dashboard;
mod events;

pub use dashboard::Dashboard;
pub use events::EventHandler;

use crate::metrics::AggregatedMetrics;
use anyhow::Result;
use crossterm::{
    event::{DisableMouseCapture, EnableMouseCapture},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};
use std::io;
use tokio::sync::mpsc;

/// Message types for UI updates
#[derive(Debug, Clone)]
pub enum UiMessage {
    /// Update progress (completed, total)
    Progress { completed: usize, total: usize },
    /// Update metrics
    Metrics(AggregatedMetrics),
    /// Add a data point for charts
    DataPoint {
        ttft_ms: f64,
        total_time_ms: f64,
        input_throughput: f64,
        output_throughput: f64,
    },
    /// Benchmark completed
    Complete,
    /// Error occurred
    Error(String),
}

/// UI State
pub struct UiState {
    pub completed: usize,
    pub total: usize,
    pub metrics: Option<AggregatedMetrics>,
    pub is_complete: bool,
    pub error_message: Option<String>,
    pub start_time: std::time::Instant,
    pub ttft_values: Vec<f64>,
    pub total_time_values: Vec<f64>,
    pub input_throughput_values: Vec<f64>,
    pub output_throughput_values: Vec<f64>,
}

impl UiState {
    pub fn new(total: usize) -> Self {
        Self {
            completed: 0,
            total,
            metrics: None,
            is_complete: false,
            error_message: None,
            start_time: std::time::Instant::now(),
            ttft_values: Vec::new(),
            total_time_values: Vec::new(),
            input_throughput_values: Vec::new(),
            output_throughput_values: Vec::new(),
        }
    }

    pub fn update(&mut self, message: UiMessage) {
        match message {
            UiMessage::Progress { completed, total } => {
                self.completed = completed;
                self.total = total;
            }
            UiMessage::Metrics(metrics) => {
                self.metrics = Some(metrics);
            }
            UiMessage::DataPoint {
                ttft_ms,
                total_time_ms,
                input_throughput,
                output_throughput,
            } => {
                // Keep only last 100 data points for charts
                if self.ttft_values.len() >= 100 {
                    self.ttft_values.remove(0);
                    self.total_time_values.remove(0);
                    self.input_throughput_values.remove(0);
                    self.output_throughput_values.remove(0);
                }
                self.ttft_values.push(ttft_ms);
                self.total_time_values.push(total_time_ms);
                self.input_throughput_values.push(input_throughput);
                self.output_throughput_values.push(output_throughput);
            }
            UiMessage::Complete => {
                self.is_complete = true;
            }
            UiMessage::Error(msg) => {
                self.error_message = Some(msg);
            }
        }
    }

    pub fn elapsed_secs(&self) -> f64 {
        self.start_time.elapsed().as_secs_f64()
    }

    pub fn progress_percent(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            (self.completed as f64 / self.total as f64) * 100.0
        }
    }

    pub fn eta_secs(&self) -> Option<f64> {
        if self.completed == 0 || self.completed >= self.total {
            return None;
        }

        let elapsed = self.elapsed_secs();
        let rate = self.completed as f64 / elapsed;
        let remaining = self.total - self.completed;

        Some(remaining as f64 / rate)
    }
}

/// Run the TUI dashboard
pub async fn run_dashboard(
    mut receiver: mpsc::Receiver<UiMessage>,
    total_requests: usize,
) -> Result<()> {
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Create state
    let mut state = UiState::new(total_requests);
    let dashboard = Dashboard::new();

    // Event handler for keyboard input
    let mut events = EventHandler::new();

    // Main UI loop
    loop {
        // Draw UI
        terminal.draw(|f| dashboard.render(f, &state))?;

        // Check for keyboard events (non-blocking)
        if events.should_quit()? {
            break;
        }

        // Check for UI messages (non-blocking with timeout)
        match tokio::time::timeout(
            std::time::Duration::from_millis(100),
            receiver.recv(),
        )
        .await
        {
            Ok(Some(message)) => {
                let is_complete = matches!(message, UiMessage::Complete);
                state.update(message);
                if is_complete {
                    // Draw final state
                    terminal.draw(|f| dashboard.render(f, &state))?;
                    // Wait a moment so user can see completion
                    tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                    break;
                }
            }
            Ok(None) => break, // Channel closed
            Err(_) => {} // Timeout, continue loop
        }
    }

    // Restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    Ok(())
}
