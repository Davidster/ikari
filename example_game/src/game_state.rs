
use std::sync::{Arc, Mutex};

use ikari::asset_loader::AssetId;
use ikari::physics::rapier3d_f64::prelude::*;
use ikari::player_controller::PlayerController;
use ikari::scene::GameNodeId;
use ikari::ui::IkariUiContainer;

use crate::ui_overlay::UiOverlay;
use crate::{
    ball::BallComponent, character::Character, physics_ball::PhysicsBall, revolver::Revolver,
};

pub struct GameState {
    pub state_update_time_accumulator: f64,
    pub is_playing_animations: bool,

    pub bgm_sound_index: Option<usize>,
    pub gunshot_sound_index: Option<usize>,
    // pub gunshot_sound_data: SoundData,
    pub point_light_node_ids: Vec<GameNodeId>,

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

    pub physics_balls: Vec<PhysicsBall>,

    pub player_controller: PlayerController,
    pub character: Option<Character>,

    pub ui_overlay: IkariUiContainer<UiOverlay>,

    pub asset_ids: Arc<Mutex<AssetIds>>,
}

#[derive(Default)]
pub struct AssetIds {
    pub gun: Option<AssetId>,
    pub forest: Option<AssetId>,
    pub legendary_robot: Option<AssetId>,
    pub test_level: Option<AssetId>,
    pub anonymous_scenes: Vec<AssetId>,

    pub gunshot: Option<AssetId>,
    pub bgm: Option<AssetId>,

    pub skybox: Option<AssetId>,
}

impl ikari::gameloop::GameState<UiOverlay> for GameState {
    fn get_ui_container(&mut self) -> &mut IkariUiContainer<UiOverlay> {
        &mut self.ui_overlay
    }
}
