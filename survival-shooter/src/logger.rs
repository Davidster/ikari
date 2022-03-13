use anyhow::Result;
use std::time::{Duration, Instant};

const FRAME_TIME_HISTORY_SIZE: usize = 5000;

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
        }
        self.last_update_time = Some(Instant::now());
    }

    pub fn log(&mut self, text: &str) {
        self.log_buffer.push(text.to_string());
        if self.log_buffer.len() > (self.terminal.size().0 - 2).into() {
            self.log_buffer.remove(0);
        }
    }

    pub fn write_to_term(&self) -> Result<()> {
        self.terminal.clear_screen()?;
        let avg_frame_time_millis: Option<f64> = if self.recent_frame_times.len() != 0 {
            let alpha = 0.005;
            let mut frame_times_iterator = self
                .recent_frame_times
                .iter()
                .map(|frame_time| frame_time.as_nanos() as f64 / 1_000_000.0);
            let mut res = frame_times_iterator.next().unwrap(); // checked that length isnt 0
            for frame_time in frame_times_iterator {
                res = (1.0 - alpha) * res + (alpha * frame_time);
            }
            Some(res)
        } else {
            None
        };
        if let Some(avg_frame_time_millis) = avg_frame_time_millis {
            self.terminal.write_line(&format!(
                "Frametime: {:.2}ms ({:.2}fps)",
                avg_frame_time_millis,
                1_000.0 / avg_frame_time_millis
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
