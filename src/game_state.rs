use super::*;

#[derive(Debug)]
pub struct GameState {
    pub scene: GameScene,
    time_tracker: Option<TimeTracker>,
}

#[derive(Debug)]
pub struct GameScene {
    pub nodes: Vec<GameNode>,
}

#[derive(Debug)]
pub struct GameNode {
    pub transform: crate::transform::Transform,
    pub render_node_index: usize,
}

impl GameState {
    pub fn init() -> Self {
        Self {
            scene: GameScene { nodes: vec![] },
            time_tracker: None,
        }
    }

    pub fn on_frame_started(&mut self) {
        self.time_tracker = self.time_tracker.or_else(|| TimeTracker::new().into());
        if let Some(time_tracker) = &mut self.time_tracker {
            time_tracker.on_frame_started();
        }
    }

    pub fn time(&mut self) -> TimeTracker {
        self.time_tracker.unwrap_or_else(|| {
            panic!("Must call GameState::on_frame_started at least once before getting the time")
        })
    }
}
