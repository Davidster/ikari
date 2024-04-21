use crate::time::Duration;
use crate::time::Instant;

#[derive(Debug, Copy, Clone, Default)]
pub struct TimeTracker {
    first_frame_start: Option<Instant>,
    current_frame: Option<FrameInstants>,
    current_frame_index: usize,
    last_frame: Option<FrameInstants>,
}

#[derive(Debug, Copy, Clone)]
pub struct FrameInstants {
    pub start: Instant,
    pub sleep_and_inputs: Option<Instant>,
    pub update_done: Option<Instant>,
    pub render_done: Option<Instant>,
    pub get_surface_done: Option<Instant>,
}

#[derive(Debug, Copy, Clone)]
pub struct FrameDurations {
    pub total: Duration,
    pub sleep_and_inputs: Option<Duration>,
    pub update: Option<Duration>,
    pub render: Option<Duration>,
    pub get_surface: Option<Duration>,
}

impl FrameInstants {
    pub fn new(start: Instant) -> Self {
        Self {
            start,
            sleep_and_inputs: None,
            update_done: None,
            render_done: None,
            get_surface_done: None,
        }
    }
}

impl TimeTracker {
    pub(crate) fn on_frame_started(&mut self) {
        self.first_frame_start = Some(self.first_frame_start.unwrap_or_else(Instant::now));

        if let Some(current_frame) = self.current_frame {
            profiling::finish_frame!();
            self.last_frame = Some(current_frame);
            self.current_frame_index += 1;
        }

        self.current_frame = Some(FrameInstants::new(Instant::now()));
    }

    pub(crate) fn on_sleep_and_inputs_completed(&mut self) {
        if let Some(current_frame) = self.current_frame.as_mut() {
            current_frame.sleep_and_inputs = Some(Instant::now());
        }
    }

    pub(crate) fn on_update_completed(&mut self) {
        if let Some(current_frame) = self.current_frame.as_mut() {
            current_frame.update_done = Some(Instant::now());
        }
    }

    pub(crate) fn on_render_completed(&mut self) {
        if let Some(current_frame) = self.current_frame.as_mut() {
            current_frame.render_done = Some(Instant::now());
        }
    }

    pub(crate) fn on_get_surface_completed(&mut self) {
        if let Some(current_frame) = self.current_frame.as_mut() {
            current_frame.get_surface_done = Some(Instant::now());
        }
    }

    pub fn current_frame_index(&self) -> usize {
        self.current_frame_index
    }

    pub fn global_time(&self) -> Option<Duration> {
        self.first_frame_start.as_ref().map(Instant::elapsed)
    }

    pub fn last_frame_times(&self) -> Option<(FrameInstants, FrameDurations)> {
        match (self.last_frame, self.current_frame) {
            (Some(last_frame), Some(current_frame)) => Some((
                last_frame,
                FrameDurations {
                    total: current_frame.start - last_frame.start,
                    sleep_and_inputs: option_subtract(
                        last_frame.sleep_and_inputs,
                        Some(last_frame.start),
                    ),
                    update: option_subtract(last_frame.update_done, last_frame.sleep_and_inputs),
                    render: option_subtract(last_frame.render_done, last_frame.update_done),
                    get_surface: option_subtract(
                        last_frame.get_surface_done,
                        last_frame.render_done,
                    ),
                },
            )),
            _ => None,
        }
    }

    pub fn last_frame_busy_time_secs(&self) -> Option<f64> {
        self.last_frame_times()
            .and_then(|(_, last_frame_durations)| {
                last_frame_durations
                    .sleep_and_inputs
                    .map(|sleep_and_inputs_duration| {
                        // it's not a problem that the input processing isn't included in the 'busy' time
                        // as long as the input processing is cut off as soon as the sleep is complete
                        last_frame_durations.total.as_secs_f64()
                            - sleep_and_inputs_duration.as_secs_f64()
                    })
            })
    }
}

fn option_subtract(end: Option<Instant>, start: Option<Instant>) -> Option<Duration> {
    match (end, start) {
        (Some(end), Some(start)) => Some(end - start),
        _ => None,
    }
}
