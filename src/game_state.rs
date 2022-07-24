use super::*;

pub struct GameState {
    pub scene: Scene,
    pub time_tracker: Option<TimeTracker>,
    pub state_update_time_accumulator: f32,
    pub is_playing_animations: bool,

    pub audio_manager: AudioManager,
    pub bgm_sound_index: usize,
    pub gunshot_sound_index: usize,
    pub gunshot_sound_data: SoundData,

    pub player_node_id: GameNodeId, // TODO: move this into player controller?

    pub point_lights: Vec<PointLightComponent>,
    pub point_light_node_ids: Vec<GameNodeId>,
    pub directional_lights: Vec<DirectionalLightComponent>,

    // store the previous state and next state and interpolate between them
    pub next_balls: Vec<BallComponent>,
    pub prev_balls: Vec<BallComponent>,
    pub actual_balls: Vec<BallComponent>,
    pub ball_node_ids: Vec<GameNodeId>,
    pub ball_pbr_mesh_index: usize,

    pub ball_spawner_acc: f32,

    pub test_object_node_id: GameNodeId,
    pub crosshair_node_id: GameNodeId,
    pub revolver: Revolver,

    pub bouncing_ball_node_id: GameNodeId,
    pub bouncing_ball_body_handle: RigidBodyHandle,

    pub physics_state: PhysicsState,

    pub physics_balls: Vec<PhysicsBall>,
    pub mouse_button_pressed: bool,

    pub character: Character,
    pub player_controller: PlayerController,
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

    pub fn toggle_animations(&mut self) {
        self.is_playing_animations = !self.is_playing_animations;
    }
}
