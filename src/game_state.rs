use super::*;

#[derive(Debug)]
pub struct GameState {
    pub scene: GameScene,
    pub time_tracker: Option<TimeTracker>,
    pub state_update_time_accumulator: f32,

    pub point_lights: Vec<PointLightComponent>,
    pub directional_lights: Vec<DirectionalLightComponent>,

    // store the previous state and next state and interpolate between them
    pub next_balls: Vec<BallComponent>,
    pub prev_balls: Vec<BallComponent>,
    pub actual_balls: Vec<BallComponent>,

    pub test_object_instances: Vec<MeshInstance>,
    pub plane_instances: Vec<MeshInstance>,
}

impl GameState {
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
