use lazy_static::lazy_static;
use std::sync::Mutex;

const MAX_LOG_BUFFER_LENGTH: usize = 5000;

lazy_static! {
    pub static ref LOGGER: Mutex<Logger> = Mutex::new(Logger::new());
}

pub struct Logger {
    log_buffer: Vec<String>,
    terminal: console::Term,
}

pub fn logger_log(text: &str) {
    LOGGER.lock().unwrap().log(text);
}

// TODO: use env_logger instead of this bs
impl Logger {
    pub fn new() -> Self {
        Logger {
            log_buffer: Vec::new(),
            terminal: console::Term::stdout(),
        }
    }

    pub fn log(&mut self, text: &str) {
        self.log_buffer.push(text.to_string());
        if self.log_buffer.len() > MAX_LOG_BUFFER_LENGTH {
            self.log_buffer.remove(0);
        }
    }

    pub fn write_to_term(&mut self) -> anyhow::Result<()> {
        for log in &self.log_buffer {
            for log_line in log.split('\n') {
                self.terminal.write_line(log_line)?;
            }
        }
        self.log_buffer.clear();
        Ok(())
    }
}

impl Default for Logger {
    fn default() -> Self {
        Self::new()
    }
}
