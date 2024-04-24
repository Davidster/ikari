use crate::time_tracker::TimeTracker;

#[derive(Default, Debug, Copy, Clone, PartialEq)]
pub struct FramerateLimiter {
    limit: FramerateLimit,
    current_sleep_start: Option<crate::time::Instant>,
    monitor_refresh_rate_hertz: Option<f32>,
}

impl FramerateLimiter {
    pub fn framerate_limit(&self) -> FramerateLimit {
        self.limit
    }

    pub fn set_framerate_limit(&mut self, limit: FramerateLimit) {
        self.limit = limit;
    }

    pub fn monitor_refresh_rate_hertz(&self) -> Option<f32> {
        self.monitor_refresh_rate_hertz
    }

    pub(crate) fn set_monitor_refresh_rate(&mut self, monitor_refresh_rate_hertz: Option<f32>) {
        self.monitor_refresh_rate_hertz = monitor_refresh_rate_hertz;
    }

    /// updates the internal state of the limiter and if a sleep is needed it will sleep.
    /// near the end of the sleep period (if applicable) it continuously returns true
    /// to allow the gameloop some time to process inputs at the last possible moment
    pub(crate) fn update_and_sleep(
        &mut self,
        time_tracker: &TimeTracker,
        is_vsync_on: bool,
    ) -> bool {
        if !Self::is_supported() {
            return false;
        }

        let Some(last_frame_busy_time_secs) = time_tracker.last_frame_busy_time_secs() else {
            return false;
        };

        let Some(limit_hz) = (match self.limit {
            FramerateLimit::None => None,
            FramerateLimit::Monitor => self.monitor_refresh_rate_hertz,
            FramerateLimit::Custom(custom_limit) => Some(custom_limit),
        }) else {
            return false;
        };

        let limit_period_secs = 1.0 / limit_hz as f64;
        let monitor_period_secs = self
            .monitor_refresh_rate_hertz
            .map(|rate| 1.0 / rate as f64);

        // sleep for 1% longer when the target is within 0.5ms or faster than vsync.
        // this is done to prevent us from accidentally outpacing the monitor and
        // eventually forcing the presentation engine to start queuing up frames, which
        // would cause latency and defeat the whole purpose of the framerate limiter
        let apply_vsync_sleep_time = is_vsync_on
            && (monitor_period_secs.is_none()
                || monitor_period_secs.unwrap() > limit_period_secs - 0.0005);
        let vsync_extra_sleep_time = if apply_vsync_sleep_time { 1.01 } else { 1.0 };

        let sleep_period_secs =
            (limit_period_secs - last_frame_busy_time_secs).max(0.0) * vsync_extra_sleep_time;

        if sleep_period_secs <= 0.0 {
            return false;
        }

        let current_sleep_start = *self
            .current_sleep_start
            .get_or_insert_with(crate::time::Instant::now);

        let remaining_sleep_time_secs =
            sleep_period_secs - current_sleep_start.elapsed().as_secs_f64();

        if remaining_sleep_time_secs > 0.0 {
            // delegate the last 1ms of sleep time to the gameloop during which we can process
            // player inputs and reduce input latency
            let remaining_sleep_time_excl_input_spin = remaining_sleep_time_secs - 0.001;

            if remaining_sleep_time_excl_input_spin > 0.0 {
                log::info!("Sleeping for {remaining_sleep_time_excl_input_spin:.6} seconds");

                crate::thread::sleep(crate::time::Duration::from_secs_f64(
                    remaining_sleep_time_excl_input_spin,
                ));
            }

            true
        } else {
            self.current_sleep_start = None;
            false
        }
    }

    pub fn is_supported() -> bool {
        #[cfg(target_arch = "wasm32")]
        {
            false
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            true
        }
    }
}

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
