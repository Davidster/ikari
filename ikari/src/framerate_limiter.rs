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

    /// starts and ends the sleep timer.
    /// call get_remaining_sleep_time immediately after this to know if we need should sleep
    pub(crate) fn update(&mut self, time_tracker: &TimeTracker) {
        let Some(sleep_period_secs) = self.get_sleep_period_secs(time_tracker) else {
            self.current_sleep_start = None;
            return;
        };

        let current_sleep_start = self
            .current_sleep_start
            .get_or_insert_with(|| crate::time::Instant::now());
        let remaining_sleep_time =
            (sleep_period_secs - current_sleep_start.elapsed().as_secs_f64()).max(0.0);

        if remaining_sleep_time <= 0.0 {
            self.current_sleep_start = None;
        }
    }

    /// duration will never be 0
    pub(crate) fn get_remaining_sleep_time(
        &self,
        time_tracker: &TimeTracker,
    ) -> Option<crate::time::Duration> {
        let Some(sleep_period_secs) = self.get_sleep_period_secs(time_tracker) else {
            return None;
        };

        let Some(current_sleep_start) = self.current_sleep_start else {
            return None;
        };

        let remaining_sleep_time =
            (sleep_period_secs - current_sleep_start.elapsed().as_secs_f64()).max(0.0);

        if remaining_sleep_time <= 0.0 {
            return None;
        }

        Some(crate::time::Duration::from_secs_f64(remaining_sleep_time))
    }

    /// returns None if sleeping is disabled
    pub fn get_sleep_period_secs(&self, time_tracker: &TimeTracker) -> Option<f64> {
        let Some(last_frame_busy_time_secs) = time_tracker.last_frame_busy_time_secs() else {
            return None;
        };

        let Some(limit_hz) = (match self.limit {
            FramerateLimit::None => None,
            FramerateLimit::Monitor => self.monitor_refresh_rate,
            FramerateLimit::Custom(custom_limit) => Some(custom_limit),
        }) else {
            return None;
        };

        let limit_period_secs = 1.0 / limit_hz as f64;
        let sleep_period = (limit_period_secs - last_frame_busy_time_secs).max(0.0) * 1.01;
        if sleep_period > 0.0 {
            Some(sleep_period)
        } else {
            None
        }
    }
}
