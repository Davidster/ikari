use anyhow::Result;
use std::time::{Duration, Instant};

const FRAME_TIME_HISTORY_SIZE: usize = 25;

pub struct Logger {
    recent_frame_times: Vec<Duration>,
    last_update_time: Option<Instant>,
    log_buffer: Vec<String>,
    terminal: console::Term,
}

impl Logger {
    pub fn new() -> Self {
        Logger {
            recent_frame_times: Vec::new(),
            last_update_time: None,
            log_buffer: Vec::new(),
            terminal: console::Term::stdout(),
        }
    }

    pub fn on_frame_completed(&mut self) {
        if let Some(last_update_time) = self.last_update_time {
            self.recent_frame_times.push(last_update_time.elapsed());
            if self.recent_frame_times.len() > FRAME_TIME_HISTORY_SIZE {
                self.recent_frame_times.remove(0);
            }

            self.recent_frame_times.last()
        } else {
            None
        };
        self.last_update_time = Some(Instant::now());
    }

    pub fn log(&mut self, text: &str) {
        self.log_buffer.push(text.to_string());
        dbg!(self.terminal.size());
        if self.log_buffer.len() > (self.terminal.size().0 - 2).into() {
            self.log_buffer.remove(0);
        }
    }

    pub fn write_to_term(&self) -> Result<()> {
        self.terminal.clear_screen()?;
        // TODO: exponential falloff instead of just checking the most recent frame time
        if let Some(avg_frame_time) = self.recent_frame_times.last() {
            self.terminal.write_line(&format!(
                "Frametime: {:?} ({:?}fps)",
                avg_frame_time,
                1_000_000 / avg_frame_time.as_micros()
            ))?;
        }

        // voodoo magic:
        // https://stackoverflow.com/questions/26368288/how-do-i-stop-iteration-and-return-an-error-when-iteratormap-returns-a-result
        let results: std::result::Result<Vec<_>, _> = self
            .log_buffer
            .iter()
            .map(|log| self.terminal.write_line(log))
            .collect();
        Ok(results.map(|_| ())?)
    }
}
