use super::*;

#[derive(Debug)]
pub struct GameState {
    pub scene: GameScene,
    time_tracker: Option<TimeTracker>,
}

impl GameState {
    pub fn new(scene: GameScene) -> Self {
        Self {
            scene,
            time_tracker: None,
        }
    }

    pub fn on_frame_started(&mut self) {
        self.time_tracker = self.time_tracker.or_else(|| TimeTracker::new().into());
        if let Some(time_tracker) = &mut self.time_tracker {
            time_tracker.on_frame_started();
        }
    }

    pub fn time(&self) -> TimeTracker {
        self.time_tracker.unwrap_or_else(|| {
            panic!("Must call GameState::on_frame_started at least once before getting the time")
        })
    }
}
