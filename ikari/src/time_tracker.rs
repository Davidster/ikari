use crate::time::Duration;
use crate::time::Instant;

#[derive(Debug, Copy, Clone, Default)]
pub struct TimeTracker {
    first_frame_start: Option<Instant>,
    current_frame: Option<FrameInstants>,
    last_frame: Option<FrameInstants>,
}

#[derive(Debug, Copy, Clone)]
pub struct FrameInstants {
    pub start: Instant,
    pub update_done: Option<Instant>,
    pub get_surface_done: Option<Instant>,
    pub render_done: Option<Instant>,
}

#[derive(Debug, Copy, Clone)]
pub struct FrameDurations {
    pub total: Duration,
    pub get_surface: Option<Duration>,
    pub update: Option<Duration>,
    pub render: Option<Duration>,
}

impl FrameInstants {
    pub fn new(start: Instant) -> Self {
        Self {
            start,
            get_surface_done: None,
            update_done: None,

            render_done: None,
        }
    }
}

impl TimeTracker {
    pub(crate) fn on_frame_started(&mut self) {
        self.first_frame_start = Some(self.first_frame_start.unwrap_or_else(Instant::now));

        if let Some(current_frame) = self.current_frame {
            self.last_frame = Some(current_frame);
        }
        self.current_frame = Some(FrameInstants::new(Instant::now()));
    }

    pub(crate) fn on_update_completed(&mut self) {
        if let Some(current_frame) = self.current_frame.as_mut() {
            current_frame.update_done = Some(Instant::now());
        }
    }

    pub(crate) fn on_get_surface_completed(&mut self) {
        if let Some(current_frame) = self.current_frame.as_mut() {
            current_frame.get_surface_done = Some(Instant::now());
        }
    }

    pub(crate) fn on_render_completed(&mut self) {
        if let Some(current_frame) = self.current_frame.as_mut() {
            current_frame.render_done = Some(Instant::now());
        }
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
                    // update: last_frame
                    //     .update_done
                    //     .map(|update_done| update_done - last_frame.start),
                    // get_surface: match (last_frame.update_done, last_frame.get_surface_done) {
                    //     (Some(update_done), Some(get_surface_done)) => {
                    //         Some(get_surface_done - update_done)
                    //     }
                    //     _ => None,
                    // },
                    get_surface: last_frame
                        .get_surface_done
                        .map(|get_surface_done| get_surface_done - last_frame.start),
                    update: match (last_frame.get_surface_done, last_frame.update_done) {
                        (Some(get_surface_done), Some(update_done)) => {
                            Some(update_done - get_surface_done)
                        }
                        _ => None,
                    },
                    render: match (last_frame.update_done, last_frame.render_done) {
                        (Some(update_done), Some(render_done)) => Some(render_done - update_done),
                        _ => None,
                    },
                },
            )),
            _ => None,
        }
    }
}
