//! Keyboard event handling for the TUI

use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEvent};
use std::time::Duration;

/// Handles keyboard events for the UI
pub struct EventHandler {
    quit: bool,
}

impl EventHandler {
    pub fn new() -> Self {
        Self { quit: false }
    }

    /// Check if user wants to quit
    pub fn should_quit(&mut self) -> Result<bool> {
        // Poll for events with a short timeout
        if event::poll(Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                self.handle_key(key);
            }
        }

        Ok(self.quit)
    }

    /// Handle keyboard input
    fn handle_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Char('q') | KeyCode::Char('Q') | KeyCode::Esc => {
                self.quit = true;
            }
            _ => {}
        }
    }
}

impl Default for EventHandler {
    fn default() -> Self {
        Self::new()
    }
}
