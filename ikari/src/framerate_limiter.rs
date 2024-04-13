use crate::time_tracker::TimeTracker;

#[derive(Default, Debug, Copy, Clone, PartialEq)]
pub enum FramerateLimit {
    #[default]
    None,
    Monitor,
    Custom(f32),
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum FramerateLimitType {
    None,
    Monitor,
    Custom,
}

impl FramerateLimitType {
    pub const ALL: [FramerateLimitType; 3] = [
        FramerateLimitType::None,
        FramerateLimitType::Monitor,
        FramerateLimitType::Custom,
    ];
}

impl std::fmt::Display for FramerateLimitType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                FramerateLimitType::None => "None",
                FramerateLimitType::Monitor => "Monitor",
                FramerateLimitType::Custom => "Custom",
            }
        )
    }
}

impl From<FramerateLimit> for FramerateLimitType {
    fn from(value: FramerateLimit) -> Self {
        match value {
            FramerateLimit::None => FramerateLimitType::None,
            FramerateLimit::Monitor => FramerateLimitType::Monitor,
            FramerateLimit::Custom(_) => FramerateLimitType::Custom,
        }
    }
}

#[derive(Default, Debug, Copy, Clone, PartialEq)]
pub struct FrameRateLimiter {
    limit: FramerateLimit,
    current_sleep_start: Option<crate::time::Instant>,
    monitor_refresh_rate: Option<f32>,
}

impl FrameRateLimiter {
    pub fn framerate_limit(&self) -> FramerateLimit {
        self.limit
    }

    pub fn set_framerate_limit(&mut self, limit: FramerateLimit) {
        self.limit = limit;
    }

    pub(crate) fn set_monitor_refresh_rate(&mut self, monitor_refresh_rate: Option<f32>) {
        self.monitor_refresh_rate = monitor_refresh_rate;
    }

    /// updates the internal state of the limiter and returns true if we're sleeping
    pub(crate) fn update(&mut self, time_tracker: &TimeTracker) -> bool {
        let Some(last_frame_busy_time_secs) = time_tracker.last_frame_busy_time_secs() else {
            return false;
        };

        let Some(limit_hz) = (match self.limit {
            FramerateLimit::None => None,
            FramerateLimit::Monitor => self.monitor_refresh_rate,
            FramerateLimit::Custom(custom_limit) => Some(custom_limit),
        }) else {
            return false;
        };

        let limit_period_secs = 1.0 / limit_hz as f64;
        let sleep_period_secs = (limit_period_secs - last_frame_busy_time_secs).max(0.0) * 1.01;

        if sleep_period_secs <= 0.0 {
            return false;
        }

        match self.current_sleep_start {
            None => {
                self.current_sleep_start = Some(crate::time::Instant::now());
                true
            }
            Some(current_sleep_start) => {
                let remaining_sleep_time =
                    sleep_period_secs - current_sleep_start.elapsed().as_secs_f64();

                if remaining_sleep_time > 0.0 {
                    true
                } else {
                    self.current_sleep_start = None;
                    false
                }
            }
        }
    }
}
