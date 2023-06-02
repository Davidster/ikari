use crate::time::*;

#[derive(Debug, Copy, Clone)]
pub struct TimeTracker {
    first_frame_instant: Instant,
    last_frame_start_instant: Instant,
    current_frame_start_instant: Instant,
}

impl TimeTracker {
    pub fn new() -> Self {
        Self {
            first_frame_instant: now(),
            last_frame_start_instant: now(),
            current_frame_start_instant: now(),
        }
    }

    pub fn on_frame_started(&mut self) {
        self.last_frame_start_instant = self.current_frame_start_instant;
        self.current_frame_start_instant = now();
    }

    pub fn global_time_seconds(&self) -> f32 {
        self.first_frame_instant.elapsed().as_secs_f32()
    }

    pub fn last_frame_time_seconds(&self) -> f32 {
        self.last_frame_start_instant.elapsed().as_secs_f32()
    }
}

impl Default for TimeTracker {
    fn default() -> Self {
        Self::new()
    }
}
