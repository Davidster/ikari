use crate::asset_loader::*;
use crate::audio::*;
use crate::ball::*;
use crate::character::*;
use crate::light::*;
use crate::mesh::*;
use crate::physics::*;
use crate::physics_ball::*;
use crate::player_controller::*;
use crate::revolver::*;
use crate::scene::*;
use crate::time_tracker::*;
use crate::ui_overlay::IkariUiOverlay;

use std::sync::{Arc, Mutex};

pub struct GameState {
    pub scene: Scene,
    pub time_tracker: Option<TimeTracker>,
    pub state_update_time_accumulator: f64,
    pub is_playing_animations: bool,

    pub audio_streams: AudioStreams,
    pub audio_manager: Arc<Mutex<AudioManager>>,
    pub bgm_sound_index: Option<usize>,
    pub gunshot_sound_index: Option<usize>,
    // pub gunshot_sound_data: SoundData,
    pub player_node_id: GameNodeId,

    pub point_lights: Vec<PointLightComponent>,
    pub point_light_node_ids: Vec<GameNodeId>,
    pub directional_lights: Vec<DirectionalLightComponent>,

    // store the previous state and next state and interpolate between them
    pub next_balls: Vec<BallComponent>,
    pub prev_balls: Vec<BallComponent>,
    pub actual_balls: Vec<BallComponent>,
    pub ball_node_ids: Vec<GameNodeId>,
    pub ball_pbr_mesh_index: usize,

    pub ball_spawner_acc: f64,

    pub test_object_node_id: GameNodeId,
    pub crosshair_node_id: Option<GameNodeId>,
    pub revolver: Option<Revolver>,

    pub bouncing_ball_node_id: GameNodeId,
    pub bouncing_ball_body_handle: RigidBodyHandle,

    pub physics_state: PhysicsState,

    pub physics_balls: Vec<PhysicsBall>,

    pub character: Option<Character>,
    pub player_controller: PlayerController,

    pub cube_mesh: BasicMesh,

    pub asset_loader: Arc<AssetLoader>,
    pub asset_binder: Arc<AssetBinder>,

    pub ui_overlay: IkariUiOverlay,
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
